"""
Enhanced Conditional Image Generation Solution for Monster Dataset
Optimized for FID, CLIP-I, and CLIP-T scores
"""
import glob
import json
import os
import math
from random import random
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np

from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import CLIPTextModel, CLIPTokenizer

# ============================================================================
# Enhanced Dataset with Better Augmentation and Text Processing
# ============================================================================

class EnhancedTextImageDataset(Dataset):
    def __init__(self, data_root, caption_file, tokenizer, size=256, augment=True):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.size = size
        self.augment = augment
        
        with open(caption_file, 'r') as f:
            self.captions = json.load(f)
        
        self.image_files = glob.glob(os.path.join(data_root, "*.png"))
        
        # Enhanced augmentation pipeline
        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((size + 32, size + 32)),  # Larger for random crop
                transforms.RandomCrop((size, size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_file = self.image_files[idx]
        
        # Load and transform image
        image = Image.open(img_file).convert("RGB")
        image = self.transform(image)
        
        # Enhanced text processing
        key = img_file.split("/")[-1].split(".")[0]
        key = "_".join(key.split("_")[:-1])
        
        # More sophisticated caption generation
        given_descriptions = self.captions[key]['given_description']
        given_description = np.random.choice(given_descriptions)
        action_description = self.captions[key]['action_description']
        
        # Create varied caption formats
        caption_variants = [
            f"{given_description} {action_description}",
            f"{given_description}. {action_description}",
            f"{action_description} {given_description}",
            given_description,  # Sometimes just the description
        ]
        
        caption = np.random.choice(caption_variants)
        
        # Classifier-free guidance: 15% chance of empty caption
        caption = "" if random() < 0.15 else caption
        
        inputs = self.tokenizer(
            caption, 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        ).input_ids
        
        return {
            "pixel_values": image,
            "input_ids": inputs.squeeze(0),
        }

# ============================================================================
# Enhanced U-Net Architecture
# ============================================================================

def create_enhanced_unet():
    """Create a more sophisticated U-Net architecture"""
    return UNet2DConditionModel(
        sample_size=32,  # 32x32 latent space (256x256 / 8)
        in_channels=4,
        out_channels=4,
        layers_per_block=2,
        block_out_channels=(128, 256, 512, 512),  # Progressive channel increase
        down_block_types=(
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D", 
            "CrossAttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=512,
        attention_head_dim=64,  # Better attention mechanism
        use_linear_projection=True,  # More efficient
        class_embed_type=None,
        projection_class_embeddings_input_dim=None,
    )

# ============================================================================
# Advanced Training with Improved Generation
# ============================================================================

@torch.no_grad()
def generate_and_save_images(unet, vae, text_encoder, tokenizer, epoch, device, save_folder, 
                           guidance_scale=7.5, num_inference_steps=50, enable_bar=False):
    """Enhanced generation with classifier-free guidance"""
    unet.eval()
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    scheduler.set_timesteps(num_inference_steps)
    
    test_prompts = [
        "A red tree monster with a skull face and twisted branches.",
        "Blood-toothed monster with spiked fur, wielding an axe, and wearing armor. The monster is moving.",
        "Gray vulture monster with wings, sharp beak, and trident.",
        "Small, purple fish-like creature with large eye and pink fins. The monster is being hit.",
        "Green slime monster with tentacles and glowing eyes. The monster is idle.",
    ]
    
    batch_size = 1
    for i, prompt in enumerate(test_prompts):
        # Prepare text embeddings for classifier-free guidance
        text_inputs = tokenizer(
            [prompt], 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        ).input_ids.to(device)
        
        # Unconditional embeddings (empty prompt)
        uncond_inputs = tokenizer(
            [""], 
            return_tensors="pt", 
            padding="max_length", 
            truncation=True, 
            max_length=77
        ).input_ids.to(device)
        
        # Get embeddings
        cond_emb = text_encoder(text_inputs)[0]
        uncond_emb = text_encoder(uncond_inputs)[0]
        
        # Prepare latents
        latents = torch.randn((batch_size, 4, 32, 32)).to(device)
        latents = latents * scheduler.init_noise_sigma
        
        progress_bar = tqdm(scheduler.timesteps) if enable_bar else scheduler.timesteps
        
        for t in progress_bar:
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = scheduler.scale_model_input(latent_model_input, t)
            
            # Concatenate embeddings
            encoder_hidden_states = torch.cat([uncond_emb, cond_emb])
            
            # Predict noise
            noise_pred = unet(
                latent_model_input, 
                t, 
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute the previous noisy sample
            latents = scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode latents to image
        latents = latents / 0.18215
        images = vae.decode(latents, return_dict=False)[0]
        images = (images.clamp(-1, 1) + 1) / 2  # [-1,1] to [0,1]
        
        # Save image
        to_pil = transforms.ToPILImage()
        image = to_pil(images[0].cpu())
        image.save(os.path.join(save_folder, f"epoch_{epoch:03d}_{i}.png"))
    
    unet.train()

def cosine_lr_schedule(optimizer, step, total_steps, initial_lr, min_lr=1e-6):
    """Cosine learning rate schedule with warmup"""
    warmup_steps = total_steps * 0.1  # 10% warmup
    
    if step < warmup_steps:
        lr = initial_lr * step / warmup_steps
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    return lr

def train():
    # ========= Enhanced Hyperparameters ==========
    train_epochs = 20  # More epochs for better convergence
    batch_size = 8  # Larger batch size if GPU memory allows
    gradient_accumulation_steps = 2  # Effective batch size = 16
    initial_lr = 1e-4
    min_lr = 1e-6
    eval_freq = 2000
    save_freq = 5000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Training on device: {device}")
    print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
    
    # ========== Setup directories ==========
    ckpt_folder = "outputs/ckpt_enhanced"
    save_folder = "outputs/samples_enhanced"
    os.makedirs(ckpt_folder, exist_ok=True)
    os.makedirs(save_folder, exist_ok=True)
    
    # ========== Load Pretrained Models ==========
    pretrain_CLIP_path = "openai/clip-vit-base-patch32"
    pretrain_VAE_path = "CompVis/stable-diffusion-v1-4"
    
    # Load pre-trained CLIP
    tokenizer = CLIPTokenizer.from_pretrained(pretrain_CLIP_path)
    text_encoder = CLIPTextModel.from_pretrained(pretrain_CLIP_path).eval().to(device)
    text_encoder.requires_grad_(False)
    
    # Load pre-trained VAE
    vae = AutoencoderKL.from_pretrained(pretrain_VAE_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    # ========== Initialize Enhanced Model ==========
    unet = create_enhanced_unet().to(device)
    unet.train()
    
    # Enhanced optimizer with weight decay
    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=initial_lr, 
        betas=(0.9, 0.999), 
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Enhanced noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        variance_type="fixed_small",
        clip_sample=False,
    )
    
    # ========== Enhanced Dataset ==========
    dataset = EnhancedTextImageDataset("public_data/train", "public_data/train_info.json", tokenizer, augment=True)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches per epoch: {len(dataloader)}")
    
    total_steps = train_epochs * len(dataloader)
    print(f"Total training steps: {total_steps}")
    
    # Test initial generation
    generate_and_save_images(unet, vae, text_encoder, tokenizer, 0, device, save_folder)
    
    # ========== Enhanced Training Loop ==========
    step = 0
    best_loss = float('inf')
    
    for epoch in range(train_epochs):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{train_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            step += 1
            
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            
            # Encode text and images
            with torch.no_grad():
                encoder_hidden_states = text_encoder(input_ids)[0]
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * 0.18215  # Scaling factor
            
            # Sample noise and timestep
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, 
                (latents.shape[0],), 
                device=device
            ).long()
            
            # Add noise to latents
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
            # Predict noise
            noise_pred = unet(
                noisy_latents, 
                timesteps, 
                encoder_hidden_states=encoder_hidden_states
            ).sample
            
            # Calculate loss
            loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
            loss = loss / gradient_accumulation_steps
            loss.backward()
            
            epoch_loss += loss.item() * gradient_accumulation_steps
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(unet.parameters(), max_norm=1.0)
                
                optimizer.step()
                optimizer.zero_grad()
                
                # Update learning rate
                current_lr = cosine_lr_schedule(optimizer, step, total_steps, initial_lr, min_lr)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'LR': f"{current_lr:.6f}" if 'current_lr' in locals() else f"{initial_lr:.6f}"
            })
            
            # Generate test images
            if step % eval_freq == 0:
                print(f"\nGenerating test images at step {step}...")
                generate_and_save_images(unet, vae, text_encoder, tokenizer, step, device, save_folder)
            
            # Save checkpoint
            if step % save_freq == 0:
                print(f"\nSaving checkpoint at step {step}...")
                checkpoint_path = os.path.join(ckpt_folder, f"unet_step_{step}")
                unet.save_pretrained(checkpoint_path)
                
                # Save training state
                torch.save({
                    'step': step,
                    'epoch': epoch,
                    'model_state_dict': unet.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                }, os.path.join(ckpt_folder, f"training_state_step_{step}.pt"))
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            print(f"New best loss: {best_loss:.4f}. Saving best model...")
            unet.save_pretrained(os.path.join(ckpt_folder, "best_model"))
    
    # Final save
    print("Training completed. Saving final model...")
    unet.save_pretrained(os.path.join(ckpt_folder, "final_model"))

# ============================================================================
# Enhanced Test Generation
# ============================================================================

def generate_competition_images(model_path, output_dir="results", guidance_scale=7.5, num_inference_steps=50):
    """Generate images for competition submission"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load models
    pretrain_CLIP_path = "openai/clip-vit-base-patch32"
    pretrain_VAE_path = "CompVis/stable-diffusion-v1-4"
    
    tokenizer = CLIPTokenizer.from_pretrained(pretrain_CLIP_path)
    text_encoder = CLIPTextModel.from_pretrained(pretrain_CLIP_path).eval().to(device)
    text_encoder.requires_grad_(False)
    
    vae = AutoencoderKL.from_pretrained(pretrain_VAE_path, subfolder="vae").to(device)
    vae.requires_grad_(False)
    
    # Load trained U-Net
    unet = UNet2DConditionModel.from_pretrained(model_path).to(device)
    unet.eval()
    
    # Setup scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    scheduler.set_timesteps(num_inference_steps)
    
    # Load test prompts
    with open("public_data/test.json", "r") as f:
        test_data = json.load(f)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {len(test_data)} images...")
    
    with torch.no_grad():
        for key, value in tqdm(test_data.items(), desc="Generating images"):
            text_prompt = value["text_prompt"]
            image_name = value["image_name"]
            
            # Prepare text embeddings for classifier-free guidance
            text_inputs = tokenizer(
                [text_prompt], 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=77
            ).input_ids.to(device)
            
            uncond_inputs = tokenizer(
                [""], 
                return_tensors="pt", 
                padding="max_length", 
                truncation=True, 
                max_length=77
            ).input_ids.to(device)
            
            # Get embeddings
            cond_emb = text_encoder(text_inputs)[0]
            uncond_emb = text_encoder(uncond_inputs)[0]
            
            # Generate image
            latents = torch.randn((1, 4, 32, 32)).to(device)
            latents = latents * scheduler.init_noise_sigma
            
            for t in scheduler.timesteps:
                # Classifier-free guidance
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)
                
                encoder_hidden_states = torch.cat([uncond_emb, cond_emb])
                
                noise_pred = unet(
                    latent_model_input, 
                    t, 
                    encoder_hidden_states=encoder_hidden_states
                ).sample
                
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                
                latents = scheduler.step(noise_pred, t, latents).prev_sample
            
            # Decode to image
            latents = latents / 0.18215
            images = vae.decode(latents, return_dict=False)[0]
            images = (images.clamp(-1, 1) + 1) / 2
            
            # Save image
            to_pil = transforms.ToPILImage()
            image = to_pil(images[0].cpu())
            image = image.resize((256, 256), Image.LANCZOS)  # Ensure correct size
            image.save(os.path.join(output_dir, image_name))
    
    print(f"Generated images saved to {output_dir}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "train":
            print("Starting enhanced training...")
            train()
        elif sys.argv[1] == "test":
            model_path = sys.argv[2] if len(sys.argv) > 2 else "outputs/ckpt_enhanced/best_model"
            print(f"Generating competition images using model: {model_path}")
            generate_competition_images(model_path)
        else:
            print("Usage: python script.py [train|test] [model_path]")
    else:
        print("Starting enhanced training...")
        train()