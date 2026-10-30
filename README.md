# NeuroVolve

NeuroVolve generates images that drive a chosen brain region. Starting from a seed image, it optimizes the
image's BLIP-Diffusion Q-Former embedding under an fMRI encoding model, then decodes the
optimized embedding back into an image with BLIP-Diffusion. The models are trained on the
[Natural Scenes Dataset (NSD)](https://naturalscenesdataset.org/).

## Repository structure

```
neurovolve/
├── dataloader.py              data loading shared by both encoders (file layout, train/val/test split, NeuralDataset)
├── blip_diffusion.py          BLIP-Diffusion: Q-Former embedding extraction and image generation from an embedding
├── blip_encoder/              Q-Former (BLIP-based) encoder, used for optimization
│   ├── extract_embeddings.py  Q-Former embeddings of the stimuli (computed once, reused for training)
│   ├── model.py               BlipEncoder, voxel-wise and region-wise variants
│   └── train.py               training with early stopping on validation images, test evaluation
├── optimization/
│   └── optimize.py            embedding optimization under a mask over the encoder outputs, then image generation
├── clip_encoder/              CLIP validation encoder
│   ├── extract_features.py    CLIP ViT-H/14 features of the stimuli
│   ├── model.py               feature extractor and linear ROI readout
│   ├── train.py               training and held-out evaluation
│   └── predict.py             predicted ROI responses of generated (or any) images
└── clip_probe/
    └── probe.py               zero-shot CLIP category probe
scripts/                       shell wrappers for training, optimization, validation and the probe
```

## Method

```
seed image ──BLIP-Diffusion Q-Former──▶ embedding z (16 × 768)
                                             │
           ┌─────────────────────────────────┘
           ▼
  L2-normalize each token ─▶ flatten (12288) ─▶ MLP ─▶ readout ─▶ predicted responses
                                                    (frozen encoder)
           │
           ▼
  objective = mean predicted response of the outputs selected by a mask
  z ← Adam(z, ∇z objective)   300 steps, lr 0.01
           │
           ▼
  BLIP-Diffusion decodes z → image   (50 steps, guidance 7.5, empty prompt)
```

**Q-Former encoder** (`neurovolve/blip_encoder/model.py`). One model per subject. Each of the 16 Q-Former
tokens is L2-normalized, the tokens are flattened, and an MLP with hidden size h
(12288 → 4h → 2h → h, each layer followed by ReLU and dropout 0.1) feeds a readout. The Q-Former
stays frozen. Two variants are used:

| Variant | Outputs | h | Readout |
|---|---|---|---|
| `voxel` (visual-cortex experiments) | every voxel of the given ROIs | 512 | linear (h → n_voxels) |
| `region` (amygdala experiment) | voxel-averaged response of each given ROI | 256 | nonlinear (h → 2h → h → n_ROIs) |

Training minimizes MSE with AdamW (lr 3e-4, batch size 64, gradient-norm clipping 1.0), with the
order of the 16 Q-Former tokens randomly permuted for every training sample (the tokens are not
image patches, so their order carries no meaning; this keeps the optimization from exploiting
order-specific features). Each subject's images are split three ways:

| Split | Images | Used for |
|---|---|---|
| val | `--val_size` (default 500) images drawn at random (`--split_seed`) from the subject's non-shared images | early stopping (`--patience` 5, at most `--num_epochs` 50) |
| train | the remaining non-shared images | fitting |
| test | the NSD shared1000 images the subject viewed | final evaluation only |

The epoch with the best validation score (mean over ROIs of the median per-output correlation) is
kept and evaluated once on the test images. The validation images depend only on the subject,
`--val_size` and `--split_seed`, so all encoders of a subject trained with the same values (e.g. the
voxel-wise visual-cortex model and the region-wise amygdala model) share the same validation set.

**Optimization** (`neurovolve/optimization/optimize.py`). The seed image's Q-Former embedding is
updated with Adam to maximize (or minimize) the predicted response of the encoder outputs selected
by a mask that you define. Every iteration's embedding is saved. The seed, the first iteration
reaching the maximum and the final iteration are decoded by default.

**CLIP validation encoder** (`neurovolve/clip_encoder/`). This independent encoder checks that
generated images drive the target region in a different feature space. It is a linear model from
L2-normalized OpenCLIP ViT-H/14 (LAION-2B) image embeddings to the voxel-averaged response of each
ROI. Training uses AdamW (lr 3e-4 with exponential decay to 1.5e-4, weight decay 0.02) for a fixed
100 epochs on all of the subject's non-shared images (train + val, since there is no early
stopping), and the final epoch is used; it is evaluated on the same shared1000 test images.

**CLIP category probe** (`neurovolve/clip_probe/probe.py`). This step labels images as face, body,
scene or text, using zero-shot OpenCLIP CoCa ViT-L/14 against sets of text prompts. It reports,
per region, the fraction of images in the region's preferred category (FFA → face, EBA → body,
PPA → scene, mfs-words → text).

## Installation

```bash
git clone <this repo> && cd NeuroVolve
conda create -n neurovolve python=3.10 -y && conda activate neurovolve
pip install -r requirements.txt
```

Pretrained weights are downloaded from the Hugging Face Hub on first use:
`Salesforce/blipdiffusion`, `laion/CLIP-ViT-H-14-laion2B-s32B-b79K` and
`laion/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k`. A GPU is needed for training and generation.

Run every command from the repository root.

## Data

### What each step needs

| Step | Needs |
|---|---|
| Train the Q-Former encoder | response files, plus either precomputed Q-Former embeddings of the stimuli (fast) or the stimulus file to compute them from (below) |
| Train the CLIP validation encoder | the same response files, plus CLIP features extracted from the stimulus file |
| Optimize and generate images | a trained Q-Former encoder (`--model_dir`) and a folder of seed images |
| Score images with the CLIP encoder | a trained CLIP encoder checkpoint and the images |
| CLIP category probe | images only |

Pretrained BLIP-Diffusion and CLIP weights are downloaded automatically; no fMRI data is needed
after training.

### File layout

The fMRI data are not distributed with this repository (see
[NSD data access](https://naturalscenesdataset.org/)). All data loading is in
`neurovolve/dataloader.py` (shared by both encoders), which expects this layout under `NSD_ROOT` (edit that file to use other
data):

```
$NSD_ROOT/
├── responses/subj0{S}/avg_roi_response/
│   ├── avg_beta_order.npy      one per subject
│   └── avg_beta_{ROI}.npy      one per ROI you want to model
└── stimuli/
    └── S{S}_stimuli_227.h5py   one per subject
```

```bash
export NSD_ROOT=/path/to/nsd
```

**`avg_beta_order.npy`**: `int64`, shape `(n_images,)`. For each row of the response files, the
index (0–9999) of the image in the subject's list of 10,000 NSD images, in ascending order. Indices
0–999 are the NSD "shared1000" images; those the subject viewed form the test set, and all other
images are split into training and validation images. Subjects who completed fewer sessions have fewer rows
(e.g. 9,209 for subject 4).

**`avg_beta_{ROI}.npy`**: `float32`, shape `(n_images, n_voxels_in_ROI)`. Single-trial responses of
the ROI's voxels, averaged over the repeated presentations of each image (up to three); row `i`
corresponds to image `avg_beta_order[i]`. The ROI name in the file name is the name used with
`--rois` and in the trained models' `output_layout`. The voxel-wise encoder concatenates the voxels of
the given ROIs as its outputs; the region-wise and CLIP encoders use the mean over each ROI's voxels.

**`S{S}_stimuli_227.h5py`**: HDF5 dataset `stimuli`, `uint8`, shape `(10000, 3, 227, 227)`. The
subject's 10,000 NSD images in the same order as the indices in `avg_beta_order.npy`, resized to
227 × 227 pixels, channels first.

**Q-Former embeddings** (optional, any path, passed with `--embeddings`): the input the Q-Former
encoder is actually trained on. A `.pt` file (torch tensor) or `.npy` array of shape
`(10000, 16, 768)`: row `i` is the BLIP-Diffusion Q-Former embedding of image `i` of the stimulus
file. `python -m neurovolve.blip_encoder.extract_embeddings` writes it from the stimulus file
(`subj{S}_qformer_embeddings.pt`, float16). If you have it, training does not need the stimulus file or
the BLIP-Diffusion pipeline.

### How we built the files from NSD

1. **Single-trial betas.** `betas_fithrf_GLMdenoise_RR` in the 1.8 mm functional space, one file
   per session, divided by 300 (to percent signal change) and z-scored per voxel within each
   session.
2. **ROIs.** For each ROI, the voxels inside a binary ROI mask in the same space
   (e.g. NSD's `prf-visualrois` and `floc-*` labels) were extracted for every trial.
3. **Repeat averaging.** Using `masterordering` in `nsd_expdesign.mat` (trial → image index), the
   trials of each image were averaged; the image indices of the averaged rows form
   `avg_beta_order.npy`.
4. **Stimuli.** The subject's 10,000 images (`subjectim` in `nsd_expdesign.mat`) were taken from
   `nsd_stimuli.hdf5` and resized to 227 × 227.

## Usage

### 1. Train the Q-Former encoder

The encoder's input is the Q-Former embedding of each stimulus, which can be given in two ways:

| Input | How | Speed |
|---|---|---|
| precomputed embeddings (recommended) | `--embeddings FILE` | fast: training starts immediately, no BLIP-Diffusion pipeline needed |
| stimulus images | omit `--embeddings`; add `--save_embeddings FILE` to keep the result | slower: every run first passes all 10,000 images through BLIP-Diffusion on the GPU |

```bash
# 1a. once per subject: Q-Former embeddings of the stimuli (shared by both variants)
python -m neurovolve.blip_encoder.extract_embeddings --subject 1 --nsd_root $NSD_ROOT --output_dir features/qformer

# 1b. voxel-wise (visual cortex)
python -m neurovolve.blip_encoder.train --subject 1 --nsd_root $NSD_ROOT --level voxel \
    --rois V1v V1d V2v V2d hV4 FFA1 FFA2 EBA PPA mfswords \
    --output_dir checkpoints/encoder_voxel/subj1 --embeddings features/qformer/subj1_qformer_embeddings.pt
#     region-wise (amygdala)
python -m neurovolve.blip_encoder.train --subject 1 --nsd_root $NSD_ROOT --level region \
    --rois L_amygdala R_amygdala \
    --output_dir checkpoints/encoder_region/subj1 --embeddings features/qformer/subj1_qformer_embeddings.pt

# or directly from the images (no step 1a):
python -m neurovolve.blip_encoder.train --subject 1 --nsd_root $NSD_ROOT --level voxel --rois V1v V1d FFA1 FFA2 \
    --output_dir checkpoints/encoder_voxel/subj1 --save_embeddings features/qformer/subj1_qformer_embeddings.pt

# several subjects (extracts each subject's embeddings on the first run, then reuses them):
ROIS="V1v V1d FFA1 FFA2 PPA" bash scripts/train_encoders.sh 1 2 3 4 5 6 7 8
LEVEL=region ROIS="L_amygdala R_amygdala" bash scripts/train_encoders.sh 1 2 3 4 5 6 7 8
```

Training fits the MLP and readout with early stopping on the validation images. It writes:

- `model.chkpt`, `model_config.json`: the best epoch's encoder (the config's `output_layout` lists
  the outputs of each ROI);
- `eval.h5py`: test-set predictions, targets and per-output correlations for each ROI;
- `training_log.json`: validation score per epoch, best epoch and test score.

`--val_size` sets the number of validation images, `--split_seed` their random selection,
`--save_epochs N ...` additionally saves the model after epochs N, and `--no-shuffle_tokens` turns
off token shuffling.

### 2. Generate images

**Define the target with a mask.** The mask is a `.npy` array with one entry per encoder output
(`output_dim` in `model_config.json`); the optimizer drives the outputs where it is non-zero.
`output_layout` in `model_config.json` lists which outputs belong to each ROI the encoder was trained
on, so a mask for a region is a few lines of NumPy:

```python
import json
import numpy as np

config = json.load(open("checkpoints/encoder_voxel/subj1/model_config.json"))
mask = np.zeros(config["output_dim"], dtype=bool)
for roi in ["FFA1", "FFA2"]:                      # any subset of the training ROIs
    start, end = config["output_layout"][roi]
    mask[start:end] = True
np.save("ffa_mask.npy", mask)
```

Any subset of outputs works, e.g. individual voxels. For the region-wise encoder, `np.ones(2, bool)`
targets the mean of the left and right amygdala outputs. A real-valued mask weights the outputs: the
objective is then `sum(w * prediction) / sum(|w|)`, so e.g. `+1` on one region and `-1` on another
drives the first while suppressing the second.

**Run the optimization.**

```bash
python -m neurovolve.optimization.optimize \
    --model_dir checkpoints/encoder_voxel/subj1 --mask ffa_mask.npy \
    --image_path seed_images --output_dir results/s1_ffa \
    --num_iterations 300 --lr 0.01 --elbow_percentages 100
# same, with the paper settings:
bash scripts/run_optimization.sh checkpoints/encoder_voxel/subj1 ffa_mask.npy seed_images results/s1_ffa
```

For each seed image `{id}.png` this writes:

```
results/s1_ffa/maximize/{id}/{id}.png                       seed image
results/s1_ffa/maximize/{id}/results.pt                     embeddings (301, 1, 16, 768) + objective per iteration
results/s1_ffa/maximize/{id}/random_{seed}/initial_image.png
results/s1_ffa/maximize/{id}/random_{seed}/iter_{i:05d}_activation_{a:.4f}.png
results/s1_ffa/maximize/{id}/random_{seed}/optimization_progress.png
```

Useful options:

- `--mode minimize`: suppress the masked outputs instead of driving them.
- `--image_save_interval N`: decode every N iterations instead of the default iterations.
- `--num_seeds` / `--generation_seeds`: choose the diffusion seeds.

### 3. Train the CLIP validation encoder and score generated images

```bash
python -m neurovolve.clip_encoder.extract_features --subject 1 --nsd_root $NSD_ROOT --output_dir features/clip
python -m neurovolve.clip_encoder.train --subject 1 --nsd_root $NSD_ROOT --feature_dir features/clip \
    --rois V1v V1d V2v V2d hV4 FFA1 FFA2 EBA PPA mfswords L_amygdala R_amygdala \
    --output_dir checkpoints/clip_encoder/subj1
# several subjects:
ROIS="V1v V1d FFA1 FFA2 PPA L_amygdala R_amygdala" bash scripts/train_clip_encoders.sh 1 2 3 4 5 6 7 8
```

To score a run, predict the seed image and every decoded iteration (mean over the given ROIs):

```bash
python -m neurovolve.clip_encoder.predict --rois FFA1 FFA2 \
    --checkpoint checkpoints/clip_encoder/subj1/00100.chkpt \
    --run_dir results/s1_ffa/maximize/{id}/random_{seed}
# every run in an output directory:
bash scripts/validate_with_clip.sh checkpoints/clip_encoder/subj1/00100.chkpt results/s1_ffa FFA1 FFA2
# arbitrary images -> CSV:
python -m neurovolve.clip_encoder.predict --rois PPA --checkpoint ... --images my_images/ --output preds.csv
```

### 4. CLIP category probe

The probe expects one folder of images per image set, subject and region:

```
IMAGE_ROOT/{set}/s{S}_{ffa|eba|ppa|mfswords}/*.png     (files are ranked in sorted order)
```

For the paper the sets were the top-100 NeuroVolve images (`neurovolve`), the natural images with the
highest predicted response (`predicted`) and those with the highest measured response (`measured`).

```bash
python -m neurovolve.clip_probe.probe --image_root IMAGE_ROOT --output_dir results/clip_probe
# or:
bash scripts/run_clip_probe.sh IMAGE_ROOT results/clip_probe
```

This writes `clip_probe_per_image.csv`, `clip_probe_accuracy.csv` and `clip_probe_accuracy_wide.csv`.

## Reproducibility notes

- The encoder is trained on Q-Former embeddings from a float16 BLIP-Diffusion pipeline.
  Optimization and generation run in float32. If you compute the embeddings yourself for
  `--embeddings`, use the same pipeline and preprocessing (`get_blip_embed` in
  `neurovolve/blip_diffusion.py` with an empty category), so that they match the embeddings the
  optimizer starts from.
- Both encoders iterate over the training images in a fixed order, without shuffling, as in the
  original runs (`make_dataloader` in `neurovolve/dataloader.py`). Features and response targets are
  stored in float16.
- The CLIP encoder is linear, so each output (ROI) has its own weights: training it on more ROIs
  does not change the predictions for any one ROI (apart from the shared gradient-norm clipping).
- Diffusion seeds are drawn at random unless `--generation_seeds` is given. Each run's seed is
  recorded in its `random_{seed}` folder name, and `results.pt` keeps every embedding, so any
  iteration can be re-decoded exactly with `generate_image_from_embedding`. Exact reproduction
  needs the same device type: the initial latents come from a `torch.Generator` on the pipeline's
  device, and CUDA and CPU generators give different noise for the same seed.
- The validation encoder's CLIP features are computed from the 227 px NSD stimuli with
  `CLIPImageProcessor` defaults (224 px resize and center crop).

## Citation

```bibtex
@article{neurovolve,
  title  = {NeuroVolve},
  author = {},
  year   = {}
}
```

## License

See `LICENSE`.
