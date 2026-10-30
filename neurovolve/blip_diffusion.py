"""BLIP-Diffusion wrapper used for both Q-Former embedding extraction and image generation.

The pipeline lets us (1) read the 16x768 Q-Former query embedding of an image and
(2) decode an arbitrary (optimized) Q-Former embedding back into an image by injecting it in
place of the embedding computed from a reference image.
"""
from typing import List, Optional, Union

import PIL
import torch
from diffusers import BlipDiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"


class CustomBlipDiffusionPipeline(BlipDiffusionPipeline):
    """BlipDiffusionPipeline that can generate from a user-supplied Q-Former embedding."""

    def __init__(self, tokenizer, text_encoder, vae, unet, scheduler, qformer, image_processor,
                 ctx_begin_pos: int = 2, mean: List[float] = None, std: List[float] = None):
        super().__init__(tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, unet=unet,
                         scheduler=scheduler, qformer=qformer, image_processor=image_processor,
                         ctx_begin_pos=ctx_begin_pos, mean=mean, std=std)
        self._override_embed = None

    def set_custom_image_embed(self, embed: Optional[torch.Tensor]):
        """Use `embed` (1, 16, 768) instead of the Q-Former embedding of a reference image."""
        self._override_embed = embed

    def get_query_embeddings(self, input_image, src_subject):
        if self._override_embed is not None:
            return self._override_embed.to(device=self.device, dtype=self.unet.dtype)
        return self.qformer(image_input=input_image, text_input=src_subject, return_dict=False)

    def _get_text_embeddings(self, prompt, reference_image, source_subject_category,
                             target_subject_category, neg_prompt, guidance_scale,
                             prompt_strength, prompt_reps):
        device = self._execution_device
        if reference_image is not None:
            if isinstance(reference_image, PIL.Image.Image):
                reference_image = self.image_processor.preprocess(
                    reference_image, image_mean=self.config.mean, image_std=self.config.std, return_tensors="pt"
                )["pixel_values"]
            reference_image = reference_image.to(device)

        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(source_subject_category, str):
            source_subject_category = [source_subject_category]
        if isinstance(target_subject_category, str):
            target_subject_category = [target_subject_category]
        batch_size = len(prompt)

        prompt = self._build_prompt(prompts=prompt, tgt_subjects=target_subject_category,
                                    prompt_strength=prompt_strength, prompt_reps=prompt_reps)
        query_embeds = self.get_query_embeddings(reference_image, source_subject_category)
        text_embeddings = self.encode_prompt(query_embeds, prompt, device)

        if guidance_scale > 1.0:
            # classifier-free guidance: unconditional and conditional embeddings in one batch
            max_length = self.text_encoder.text_model.config.max_position_embeddings
            uncond_input = self.tokenizer([neg_prompt] * batch_size, padding="max_length",
                                          max_length=max_length, return_tensors="pt")
            uncond_embeddings = self.text_encoder(input_ids=uncond_input.input_ids.to(device),
                                                  ctx_embeddings=None)[0]
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: List[str],
        reference_image: Optional[PIL.Image.Image] = None,
        source_subject_category: List[str] = [""],
        target_subject_category: List[str] = [""],
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        neg_prompt: Optional[str] = "",
        prompt_strength: float = 1.0,
        prompt_reps: int = 20,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        show_progress_bar: bool = True,
        random_seed: int = 42,
    ):
        device = self._execution_device
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        text_embeddings = self._get_text_embeddings(
            prompt, reference_image, source_subject_category, target_subject_category,
            neg_prompt, guidance_scale, prompt_strength, prompt_reps)

        scale_down_factor = 2 ** (len(self.unet.config.block_out_channels) - 1)
        if generator is None:
            generator = torch.Generator(device=device)
        generator = generator.manual_seed(random_seed)
        latents = self.prepare_latents(
            batch_size=batch_size,
            num_channels=self.unet.config.in_channels,
            height=height // scale_down_factor,
            width=width // scale_down_factor,
            generator=generator,
            dtype=self.unet.dtype,
            device=device,
        )
        self.scheduler.set_timesteps(num_inference_steps)

        do_classifier_free_guidance = guidance_scale > 1.0
        timesteps = self.progress_bar(self.scheduler.timesteps) if show_progress_bar else self.scheduler.timesteps

        for t in timesteps:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            noise_pred = self.unet(
                latent_model_input,
                timestep=t,
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=None,
                mid_block_additional_residual=None,
            )["sample"]

            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]

        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        if image.requires_grad:
            image = image.detach().cpu()
        image = self.image_processor.postprocess(image, output_type=output_type)
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)
        return ImagePipelineOutput(images=image)


def init_pipe(torch_dtype=torch.float32):
    """Load BLIP-Diffusion (Salesforce/blipdiffusion) as a CustomBlipDiffusionPipeline.

    Optimization and generation use float32; encoder training extracted its Q-Former
    embeddings with a float16 pipeline.
    """
    pipe_base = BlipDiffusionPipeline.from_pretrained("Salesforce/blipdiffusion", torch_dtype=torch_dtype)
    pipe = CustomBlipDiffusionPipeline(
        tokenizer=pipe_base.tokenizer,
        text_encoder=pipe_base.text_encoder,
        vae=pipe_base.vae,
        unet=pipe_base.unet,
        scheduler=pipe_base.scheduler,
        qformer=pipe_base.qformer,
        image_processor=pipe_base.image_processor,
        ctx_begin_pos=pipe_base.ctx_begin_pos,
        mean=pipe_base.image_processor.image_mean,
        std=pipe_base.image_processor.image_std,
    ).to(device)
    return pipe


@torch.no_grad()
def get_blip_embed(pipe, image, category=[""]):
    """Q-Former query embedding (16, 768) of a PIL image or (3, H, W) image array/tensor."""
    pipe.set_custom_image_embed(None)
    pixel_values = pipe.image_processor.preprocess(
        image, image_mean=pipe.config.mean, image_std=pipe.config.std, return_tensors="pt"
    )["pixel_values"].to(device)
    return pipe.get_query_embeddings(pixel_values, category).squeeze(0)


def generate_image_from_embedding(pipe, embedding, random_seed, prompt=[""],
                                  num_inference_steps=50, guidance_scale=7.5, **kwargs):
    """Decode a Q-Former embedding (1, 16, 768) into a PIL image."""
    pipe.set_custom_image_embed(embedding)
    output = pipe(prompt=prompt, reference_image=None,
                  num_inference_steps=num_inference_steps, guidance_scale=guidance_scale,
                  random_seed=random_seed, **kwargs)
    return output.images[0]


def load_image(path):
    return Image.open(path).convert("RGB")
