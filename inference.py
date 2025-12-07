from diffusers import AutoencoderKL
import numpy as np
import open_clip
from PIL import Image
import torch

from sd3 import DiT, SD3


MODEL_CHKPT_PATH = "D:/External Drive/Anthony/sd3_checkpoints/model.pt"


if __name__ == "__main__":
    torch.cuda.empty_cache()
    clip, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2B-s34B-b88K', device="cuda")
    clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
    vae = AutoencoderKL.from_pretrained("AuraDiffusion/16ch-vae")
    vae = vae.to("cuda")
    clip.eval()
    vae.eval()
    dit = DiT()
    dit_checkpoint = torch.load(MODEL_CHKPT_PATH)
    dit.load_state_dict(dit_checkpoint["model_state_dict"])
    model = SD3(clip, clip_tokenizer, vae, dit)
    text = "A dog sitting on the inside of a white boat."  # ID: 400
    img = (model(text)[0]*255).astype(np.uint8)
    Image.fromarray(img).save("img.png")
