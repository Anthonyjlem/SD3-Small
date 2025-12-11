import line_profiler
import os

import bitsandbytes as bnb
from diffusers import AutoencoderKL
import numpy as np
import open_clip
from PIL import Image
from safetensors.torch import save_file
import torch
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import wandb

from coco_dataloaders import build_dataloaders
from linear_warmup_scheduler import linear_warmup_scheduler
from model_utils import freeze_model, get_vae_latent, sample_time, print_model_size
from profiler_utils import VRAMProfiler
from sd3 import DiT


os.environ["WANDB_MODE"] = "online"


VALIDATION_AND_EMA = False
PROFILE_VRAM = False
BATCH_SIZE = 8
LR = 1e-4
SCHEDULER = "reduce"  # warmup (linear warmup) or reduce (reduce learning rate on plateau)
WARMUP_STEPS = 1000  # used if SCHEDULER is "warmup"
EPOCHS = 10000
SAVE_DIR = "D:/External Drive/Anthony/sd3_checkpoints"
CHECKPOINT = os.path.join(SAVE_DIR, "model.pt")


@line_profiler.profile
def train_one_epoch(model, epoch, num_epochs, train_loader, vae, clip, optimizer, loss_fn, scheduler, scaler, ema_model):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    if PROFILE_VRAM:
        vram_profiler = VRAMProfiler()
        vram_profiler.sample_vram()
    for i, data in enumerate(pbar):
        images, tokens = data
        images, tokens = images.to("cuda"), tokens.to("cuda")
        with autocast("cuda"):
            latents = get_vae_latent(images, vae)  # B x C x H x W
            text_embeddings = clip.forward_intermediates(text=tokens, text_indices=1, normalize_intermediates=True)
            ctx = text_embeddings["text_intermediates"][-1]
            ctx_pool = text_embeddings["text_features"]
        t = sample_time(latents.size(0), device="cuda")  # B
        t_unsqueeze = t[:,None,None,None]
        epsilon = torch.randn_like(latents)
        z = (1 - t_unsqueeze)*latents + t_unsqueeze*epsilon
        targets = epsilon - latents
        optimizer.zero_grad()
        with autocast("cuda"):
            velocities = model(z, ctx, ctx_pool, t)
            if PROFILE_VRAM:
                vram_profiler.sample_vram()
            loss = loss_fn(velocities, targets)
        if PROFILE_VRAM:
            vram_profiler.sample_vram()
        scaler.scale(loss).backward()
        if PROFILE_VRAM:
            vram_profiler.sample_vram()
        scaler.step(optimizer)
        if PROFILE_VRAM:
            vram_profiler.sample_vram()
        scaler.update()
        if SCHEDULER == "warmup":
            scheduler.step()
        else:
            scheduler.step(loss)
        loss = loss.detach().item()
        running_loss += loss
        if PROFILE_VRAM:
            print(f"Peak VRAM during training: {vram_profiler.get_peak_vram() / (1024**3):.2f} GB")
            vram_profiler.shutdown()
        pbar.set_postfix({'Train Loss': f'{loss:.4f}'})
        break
    if VALIDATION_AND_EMA:
        model = model.to("cpu")
        ema_model.update_parameters(model)

    return running_loss/(i+1)


def validate_one_epoch(model, epoch, num_epochs, val_loader, vae, clip, loss_fn):
    model.eval()
    running_loss = 0
    with torch.inference_mode():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for i, data in enumerate(pbar):
            images, tokens = data  # tokens are B x 77
            images, tokens = images.to("cuda"), tokens.to("cuda")
            with autocast("cuda"):
                latents = get_vae_latent(images, vae)  # B x C x H x W
                text_embeddings = clip.forward_intermediates(text=tokens, text_indices=1, normalize_intermediates=True)
                ctx = text_embeddings["text_intermediates"][-1]
                ctx_pool = text_embeddings["text_features"]
            t = sample_time(latents.size(0), device="cuda")  # B
            t_unsqueeze = t[:,None,None,None]
            epsilon = torch.randn_like(latents)  # B x 16 x 16 x 16
            z = (1 - t_unsqueeze)*latents + t_unsqueeze*epsilon
            targets = epsilon - latents
            with autocast("cuda"):
                velocities = model(z, ctx, ctx_pool, t)
                loss = loss_fn(velocities, targets).detach().item()
            running_loss += loss
            pbar.set_postfix({'Validation Loss': f'{loss:.4f}'})
            break

    return running_loss/(i+1)


def generate_sample(text, model, vae, clip, clip_tokenizer, num_steps=100):
    with torch.inference_mode():
        tokens = clip_tokenizer(text).to("cuda")
        tokens = tokens.repeat(1, 1)
        text_embeddings = clip.forward_intermediates(text=tokens, text_indices=1, normalize_intermediates=True)
        ctx = text_embeddings["text_intermediates"][-1]
        ctx_pool = text_embeddings["text_features"]
        latents = torch.randn((1, 16, 16, 16), device="cuda")
        timesteps = torch.linspace(1, 0, num_steps+1, device="cuda")
        dt = -1 / num_steps
        for step in tqdm(timesteps[:-1]):
            velocities = model(latents, ctx, ctx_pool, step[None])
            latents = latents + dt * velocities
        decoded_image_tensor = vae.decode(latents / vae.config.scaling_factor).sample  # undo scaling factor
        decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)*255  # denormalize to [0, 255]
        decoded_image = decoded_image_tensor.permute(0,2,3,1).cpu().numpy() # NCHW to NHWC
        return decoded_image[0].astype(np.uint8)


def train_loop(model, epochs, vae, clip, clip_tokenizer, train_loader, val_loader, optimizer, loss_fn, scheduler, lr, warmup_steps):
    config = {"learning_rate": lr,
              "warmup_steps": warmup_steps}
    wandb.init(project="SD3-Small",
               config=config)

    ema_model = torch.optim.swa_utils.AveragedModel(model,
                                                    device="cpu",
                                                    multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.99),
                                                    use_buffers=True)
    scaler = GradScaler("cuda")

    model = model.to("cuda")
    print("Evaluating initial parameters on train and validation sets...")
    avg_train_loss = validate_one_epoch(model, 0, 1, train_loader, vae, clip, loss_fn)
    if VALIDATION_AND_EMA:
        best_val_loss = validate_one_epoch(model, 0, 1, val_loader, vae, clip, loss_fn)
    else:
        best_train_loss = avg_train_loss
    text = "A dog sitting on the inside of a white boat."  # ID: 400
    img = generate_sample(text, model, vae, clip, clip_tokenizer, num_steps=5)
    Image.fromarray(img).save(os.path.join(SAVE_DIR, "sample_img_0.png"))
    log = {"Average Train Loss Per Epoch": avg_train_loss,
           "Epoch": 0,
           "Smaple Image": wandb.Image(os.path.join(SAVE_DIR, "sample_img_0.png"), caption=text)}
    if VALIDATION_AND_EMA:
        log["Average Validation Loss Per Epoch"] = best_val_loss
    wandb.log(log)

    print("Training...")
    for e in range(epochs):
        if VALIDATION_AND_EMA:
            ema_model = ema_model.to("cpu")
            model = model.to("cuda")
        avg_train_loss = train_one_epoch(model, e, epochs, train_loader, vae, clip, optimizer, loss_fn, scheduler, scaler, ema_model)
        checkpoint = {
            "epoch": e,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "avg_train_loss": avg_train_loss,
        }
        tmp_save_file = os.path.join(SAVE_DIR, "model_tmp.pt")
        torch.save(checkpoint, tmp_save_file)
        os.replace(tmp_save_file, os.path.join(SAVE_DIR, "model.pt"))
        if VALIDATION_AND_EMA:
            model = model.to("cpu")
            ema_model = ema_model.to("cuda")
            avg_val_loss = validate_one_epoch(ema_model, e, epochs, val_loader, vae, clip, loss_fn)
            checkpoint = {
                "epoch": e,
                "model_state_dict": ema_model.state_dict(),
                "avg_val_loss": avg_val_loss,
            }
            tmp_save_file = os.path.join(os.path.dirname(SAVE_DIR), "model_ema_tmp.pt")
            torch.save(checkpoint, tmp_save_file)
            os.replace(tmp_save_file, os.path.join(SAVE_DIR, "model_ema.pt"))
            img = generate_sample(text, ema_model, vae, clip, clip_tokenizer, num_steps=5)
        else:
            img = generate_sample(text, model, vae, clip, clip_tokenizer, num_steps=5)
        Image.fromarray(img).save(os.path.join(SAVE_DIR, "sample_img_{}.png".format(e+1)))
        log = {"Average Train Loss Per Epoch": avg_train_loss,
               "Epoch": e+1,
               "Smaple Image": wandb.Image(os.path.join(SAVE_DIR, "sample_img_{}.png".format(e+1)), caption=text)}
        if VALIDATION_AND_EMA:
            log["Average Validation Loss Per Epoch"] = avg_val_loss
        wandb.log(log)
        if VALIDATION_AND_EMA:
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                state_dict = ema_model.state_dict()
                tmp_save_file = os.path.join(SAVE_DIR, "model_{}_tmp.safetensors".format(e+1))
                save_file(state_dict, tmp_save_file)
                output_path = os.path.join(SAVE_DIR, "model_{}.safetensors".format(e+1))
                os.replace(tmp_save_file, output_path)
        else:
            if avg_train_loss < best_train_loss:
                best_train_loss = avg_train_loss
                chkpt_save_file = os.path.join(SAVE_DIR, "model_{}.pt".format(e+1))
                torch.save(checkpoint, chkpt_save_file)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2B-s34B-b88K', device="cuda")
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
    vae = AutoencoderKL.from_pretrained("AuraDiffusion/16ch-vae")
    vae = vae.to("cuda")
    freeze_model(clip)
    freeze_model(vae)
    clip.eval()
    vae.eval()
    model = DiT()
    dit_chkpt = None
    if CHECKPOINT:
        dit_chkpt = torch.load(CHECKPOINT, map_location="cpu")
        model.load_state_dict(dit_chkpt["model_state_dict"])
    if PROFILE_VRAM:
        print_model_size(model)

    train_loader, val_loader = build_dataloaders(BATCH_SIZE, clip_tokenizer)
    loss_fn = torch.nn.MSELoss()
    optimizer = bnb.optim.AdamW(model.parameters(), lr=LR)
    if CHECKPOINT:
        optimizer.load_state_dict(dit_chkpt["optimizer_state_dict"])
    if SCHEDULER == "warmup":
        scheduler = linear_warmup_scheduler(optimizer, WARMUP_STEPS)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
    train_loop(model, EPOCHS, vae, clip, clip_tokenizer, train_loader, val_loader, optimizer, loss_fn, scheduler, LR, WARMUP_STEPS)
