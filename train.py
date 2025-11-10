import line_profiler
import os

from diffusers import AutoencoderKL
import open_clip
from safetensors.torch import save_file
import torch
from torch.amp import GradScaler
from tqdm import tqdm
import wandb

from coco_dataloaders import build_dataloaders
from linear_warmup_scheduler import linear_warmup_scheduler
from model_utils import freeze_model, get_vae_latent, sample_time
from sd3 import DiT


os.environ["WANDB_MODE"] = "disabled"


BATCH_SIZE = 16
LR = 1e-4
WARMUP_STEPS = 1000
EPOCHS = 1000
TRAIN_MODEL_SAVE_PATH = "model.pt"
VAL_MODEL_SAVE_PATH = "model_ema.pt"


@line_profiler.profile
def train_one_epoch(model, epoch, num_epochs, train_loader, vae, clip, optimizer, loss_fn, scheduler, scaler, ema_model):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
    for i, data in enumerate(pbar):
        images, tokens = data
        images, tokens = images.to("cuda"), tokens.to("cuda")
        latents = get_vae_latent(images, vae)  # B x C x H x W
        text_embeddings = clip.forward_intermediates(text=tokens, text_indices=1, normalize_intermediates=True)
        ctx = text_embeddings["text_intermediates"][-1]
        ctx_pool = text_embeddings["text_features"]
        t = sample_time(latents.size(0)).to("cuda")  # B
        t_unsqueeze = t[:,None,None,None]
        epsilon = torch.randn_like(latents)
        z = (1 - t_unsqueeze)*latents + t_unsqueeze*epsilon
        targets = -t_unsqueeze/(1-t_unsqueeze)*z - t_unsqueeze*(-t_unsqueeze/(1-t_unsqueeze) - 1/t_unsqueeze)*epsilon
        optimizer.zero_grad()
        velocities = model(latents, ctx, ctx_pool, t)
        loss = loss_fn(velocities, targets)
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        scheduler.step()
        loss = loss.detach().item()
        running_loss += loss
        # if (i+1) % 1 == 0:
        #     cpu_model = model.to("cpu")
        #     ema_model.update_parameters(cpu_model)
        #     model = model.to("cuda")
        pbar.set_postfix({'Train Loss': f'{loss:.4f}'})

    del data
    del images
    del tokens
    del latents
    del text_embeddings
    del ctx
    del ctx_pool
    del t
    del t_unsqueeze
    del epsilon
    del z
    del targets
    del velocities
    del loss
    del cpu_model

    return running_loss/(i+1)


def validate_one_epoch(model, epoch, num_epochs, val_loader, vae, clip, loss_fn):
    model.eval()
    running_loss = 0
    with torch.inference_mode():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for i, data in enumerate(pbar):
            images, tokens = data  # tokens are B x 77
            images, tokens = images.to("cuda"), tokens.to("cuda")
            latents = get_vae_latent(images, vae)  # B x C x H x W
            text_embeddings = clip.forward_intermediates(text=tokens, text_indices=1, normalize_intermediates=True)
            ctx = text_embeddings["text_intermediates"][-1]
            ctx_pool = text_embeddings["text_features"]
            t = sample_time(latents.size(0)).to("cuda")  # B
            t_unsqueeze = t[:,None,None,None]
            epsilon = torch.randn_like(latents)  # B x 16 x 16 x 16
            z = (1 - t_unsqueeze)*latents + t_unsqueeze*epsilon
            targets = -t_unsqueeze/(1-t_unsqueeze)*z - t_unsqueeze*(-t_unsqueeze/(1-t_unsqueeze) - 1/t_unsqueeze)*epsilon
            velocities = model(latents, ctx, ctx_pool, t)
            loss = loss_fn(velocities, targets).detach().item()
            running_loss += loss
            pbar.set_postfix({'Validation Loss': f'{loss:.4f}'})
            break

    del data
    del images
    del tokens
    del latents
    del text_embeddings
    del ctx
    del ctx_pool
    del t
    del t_unsqueeze
    del epsilon
    del z
    del targets
    del velocities
    del loss

    return running_loss/(i+1)


def train_loop(model, epochs, vae, clip, train_loader, val_loader, optimizer, loss_fn, scheduler, lr, warmup_steps):
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
    best_val_loss = validate_one_epoch(model, 0, 1, val_loader, vae, clip, loss_fn)
    wandb.log({"Average Train Loss Per Epoch": avg_train_loss,
               "Average Validation Loss Per Epoch": best_val_loss,
               "Epoch": 0})

    print("Training...")
    for e in range(epochs):
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
        torch.save(checkpoint, TRAIN_MODEL_SAVE_PATH)
        model = model.to("cpu")
        ema_model = ema_model.to("cuda")
        avg_val_loss = validate_one_epoch(ema_model, e, epochs, val_loader, vae, clip, loss_fn)
        checkpoint = {
            "epoch": e,
            "model_state_dict": ema_model.state_dict(),
            "avg_val_loss": avg_val_loss,
        }
        torch.save(checkpoint, VAL_MODEL_SAVE_PATH)
        wandb.log({"Average Train Loss Per Epoch": avg_train_loss,
                   "Average Validation Loss Per Epoch": avg_val_loss,
                   "Epoch": e+1})
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            state_dict = model.state_dict()
            output_path = "model_{}.safetensors".format(e+1)
            save_file(state_dict, output_path)


if __name__ == "__main__":
    clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2B-s34B-b88K', device="cuda")
    vae = AutoencoderKL.from_pretrained("AuraDiffusion/16ch-vae")
    vae = vae.to("cuda")
    freeze_model(clip)
    freeze_model(vae)
    clip.eval()
    vae.eval()
    model = DiT()

    train_loader, val_loader = build_dataloaders(BATCH_SIZE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = linear_warmup_scheduler(optimizer, WARMUP_STEPS)
    train_loop(model, EPOCHS, vae, clip, train_loader, val_loader, optimizer, loss_fn, scheduler, LR, WARMUP_STEPS)
