# Generative-Artificial-Intelligence-Condition-Image-Generation

This repository contains my solution for **HW6 – Text-guided Monster Image Generation** (Generative AI). The goal is to train a **conditional diffusion U-Net from scratch** to generate **256×256** monster images guided by text prompts, **while only using the officially allowed pretrained text encoder and VAE**. 

---

## Task Overview

You are given a monster dataset with text descriptions and action descriptions. Your model should learn to generate images conditioned on the input text. The competition evaluates your generated images using:

* **FID** (lower is better)
* **CLIP-I** (higher is better)
* **CLIP-T** (higher is better)

Official baselines: **FID 120↓**, **CLIP-I 0.70↑**, **CLIP-T 0.25↑**. 

---

## Rules (Important)

* You **can only** use:

  * The **pretrained text encoder** defined in the sample code
  * The **pretrained VAE** defined in the sample code
* **Any other pretrained model is forbidden.**
* All other components (especially **U-Net**) must be trained from scratch. 

The sample code uses:

* `openai/clip-vit-base-patch32`
* `CompVis/stable-diffusion-v1-4` (VAE subfolder) 

---

## Dataset Structure

Expected files/folders:

* `train_info.json`: text descriptions for images
* `train/`: **43,294** training images of ~2000 monsters, each with ≥5 actions
* `test.json`: **1,063** text prompts for generation
* You must generate **1,063** images at **256×256** following the filenames in `test.json`. 

---

## Environment

### Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

The provided requirement list includes:

* torch, torchvision, numpy, tqdm
* transformers, scipy
* open_clip_torch, tensorboard 

---

## Project Structure (Recommended)

For E3 submission, the expected zip layout is:

```
hw6_{student_id}/
 ├── code/        # training & inference code
 ├── model/       # U-Net weights (or cloud link if too large)
 ├── results/     # generated images
 └── report.pdf
```

Follow the official format to avoid penalties. 

---

## Code Files

* `train.py`: official sample training pipeline + dataloader + allowed pretrained modules 
* `test.py`: official sample inference loader for `test.json` (placeholder generation logic) 
* `My_Code.py`: my enhanced implementation (stronger U-Net, better augmentation, CFG, improved training/inference utilities) 

---

## My Method Summary

### Model

I used a customized `UNet2DConditionModel` with **4 Down/Up blocks** and channel progression:

* `(128, 256, 512, 512)`
* cross-attention dimension 512
* attention head dim 64
* linear projection enabled to control parameter count

This design follows the report’s optimized configuration. 

### Training

Key settings:

* Fixed pretrained **CLIP-B/32** and **SD v1-4 VAE**, both frozen
* Latent size: **32×32** (256 / 8)
* **Classifier-Free Guidance training** via empty caption probability (~15%)
* Optimizer: **AdamW**, lr = `1e-4`
* Large effective batch size using gradient accumulation
* Stronger augmentation (random crop, flip, ColorJitter)

These choices are consistent with the report and enhanced code.

### Inference

* Scheduler: **DDIM**
* Steps: **50**
* CFG scale: **~7.5** (tested 6–8 for balance)
* Latents rescaled by `1/0.18215` before VAE decode

See the report configuration. 

---

## How to Run

### 1. Prepare Data

Place the dataset as:

```
data/
 ├── train/
 ├── train_info.json
 └── test.json
```

(Adjust paths in code if your folder name differs.)

---

### 2. Train

Using the baseline script:

```bash
python train.py
```

Or my enhanced script:

```bash
python My_Code.py train
```

This will:

* load allowed pretrained CLIP + VAE
* initialize the custom U-Net
* train with augmentation and CFG-ready captions
* save checkpoints to the output folder

---

### 3. Generate Competition Images

Baseline-style usage requires you to load your trained U-Net in `test.py`. 

With my enhanced script:

```bash
python My_Code.py test outputs/ckpt_enhanced/best_model
```

Generated images will be saved to `results/` (or the directory specified in code). 

---

## Results (From My Report)

My reported performance:

* **FID:** 72.3964
* **CLIP-T:** 0.2935
* **CLIP-I:** 0.7890

All significantly better than the official baseline. 

---

## Tips From Official Hints

* Increase U-Net depth/width beyond the 1-block sample
* Implement **Classifier-Free Guidance**
* Train longer and compare checkpoints
* Use larger batch size or gradient accumulation

These are aligned with the official update notes. 

---

## Notes

* **No plagiarism.**
* Ensure your Codabench account uses the same email as E3 and register with your **student ID** as username (per rules). 
* Follow `test.json` filenames exactly for submissions.

---

## Acknowledgements

This implementation is based on the official HW6 sample pipeline and the allowed pretrained modules specified by the course staff.
