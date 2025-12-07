from diffusers import AutoencoderKL
import numpy as np
import open_clip
from PIL import Image
import torch
from torch.amp import autocast

from coco_dataloaders import build_dataloaders
from model_utils import freeze_model, get_vae_latent


def vae_test(train_loader, vae):
    images, tokens = next(iter(train_loader))
    images, tokens = images.to("cuda"), tokens.to("cuda")
    with autocast("cuda"):
        latents = get_vae_latent(images, vae)  # B x C x H x W
        with torch.inference_mode():
            decoded_image_tensor = vae.decode(latents / vae.config.scaling_factor).sample  # undo scaling factor
            decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)*255  # denormalize to [0, 255]
            decoded_image = decoded_image_tensor.permute(0,2,3,1).cpu().numpy() # NCHW to NHWC
            decoded_image = decoded_image[0].astype(np.uint8)
            Image.fromarray(decoded_image).save("sample_img.png")


if __name__ == "__main__":
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
    vae = AutoencoderKL.from_pretrained("AuraDiffusion/16ch-vae")
    vae = vae.to("cuda")
    freeze_model(vae)
    vae.eval()
    train_loader, _ = build_dataloaders(1, clip_tokenizer)
    vae_test(train_loader, vae)