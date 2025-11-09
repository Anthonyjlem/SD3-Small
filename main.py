from diffusers import AutoencoderKL
import open_clip
from safetensors.torch import save_file
import torch
from torch.amp import GradScaler
import torchvision.datasets as dset
import torchvision.transforms as transforms
from tqdm import tqdm
import wandb

from linear_warmup_scheduler import linear_warmup_scheduler
from sd3 import DiT


# XXX: Should test that the VAE can encode and decode images properly give the transforms
# I doubt it, but maybe the VAE input should be normalized from [-1, 1]
TRAIN_MODEL_SAVE_PATH = "model.pt"
VAL_MODEL_SAVE_PATH = "model_ema.pt"


transform = transforms.Compose([
    transforms.Resize(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))  # normalize to [-1, 1]
])
train_set = dset.CocoCaptions(root="../Datasets/COCO/train2017",
                              annFile="../Datasets/COCO/annotations_trainval2017/captions_train2017.json",
                              transform=transform)
val_set = dset.CocoCaptions(root="../Datasets/COCO/val2017",
                            annFile="../Datasets/COCO/annotations_trainval2017/captions_val2017.json",
                            transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1024, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1024, shuffle=False)

loss_fn = torch.nn.MSELoss()

clip, preprocess = open_clip.create_model_from_pretrained('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
clip_tokenizer = open_clip.get_tokenizer('hf-hub:laion/CLIP-ViT-B-16-laion2B-s34B-b88K')
vae = AutoencoderKL.from_pretrained("AuraDiffusion/16ch-vae")
for param in clip.parameters():
    param.requires_grad = False
for param in clip_tokenizer.parameters():
    param.requires_grad = False
for param in vae.parameters():
    param.requires_grad = False
clip.eval()
clip_tokenizer.eval()
vae.eval()
clip = clip.to("cuda")
clip_tokenizer = clip_tokenizer.to("cuda")
vae = vae.to("cuda")
model = DiT()

lr = 1e-4
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
warmup_steps = 1000
scheduler = linear_warmup_scheduler(optimizer, warmup_steps)


def get_vae_latent(image, vae):
    latent = vae.encode(image).latent_dist.mean
    latent *= vae.config.scaling_factor
    return latent


def train_one_epoch(model, epoch, num_epochs, train_loader, vae, clip, tokenizer, optimizer, loss_fn, scheduler, scaler, ema_model):
    model.train()
    running_loss = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    for i, data in enumerate(pbar):
        images, captions = data
        images, captions = images.to("cuda"), captions.to("cuda")
        with torch.autocast(device="cuda"), torch.no_grad():
            latents = get_vae_latent(images, vae)  # B x C x H x W
            ctx = tokenizer(captions)  # B x 77 x 768
            ctx_pool = clip.encode_text(ctx)
            ctx_pool /= ctx_pool.norm(dim=-1)  # B x 768
        t = model.sample_time(latents.size(0)).to("cuda")  # B
        epsilon = torch.randn_like(latents)
        z = (1 - t)*latents + t*epsilon
        targets = -t/(1-t)*z - t*(-t/(1-t) - 1/t)*epsilon
        optimizer.zero_grad()
        with torch.autocast(device="cuda"):
            velocities = model.forward(latents, ctx, ctx_pool, t)
            loss = loss_fn(velocities, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        loss = loss.detach().item()
        running_loss += loss
        if (i+1) % 100 == 0:
            cpu_model = model.to("cpu")
            ema_model.update_parameters(cpu_model)
        pbar.set_postfix({'Train Loss': f'{loss:.4f}'})
    return running_loss/(i+1)


def validate_one_epoch(model, epoch, num_epochs, val_loader, vae, clip, tokenizer, loss_fn):
    model.eval()
    running_loss = 0
    with torch.inference_mode():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for i, data in enumerate(pbar):
            images, captions = data
            images, captions = images.to("cuda"), captions.to("cuda")
            with torch.autocast(device="cuda"):
                latents = get_vae_latent(images, vae)  # B x C x H x W
                ctx = tokenizer(captions)  # B x 77 x 768
                ctx_pool = clip.encode_text(ctx)
                ctx_pool /= ctx_pool.norm(dim=-1)  # B x 768
            t = model.sample_time(latents.size(0)).to("cuda")  # B
            epsilon = torch.randn_like(latents)
            z = (1 - t)*latents + t*epsilon
            targets = -t/(1-t)*z - t*(-t/(1-t) - 1/t)*epsilon
            with torch.autocast(device="cuda"):
                velocities = model.forward(latents, ctx, ctx_pool, t)
                loss = loss_fn(velocities, targets).detach().item()
            running_loss += loss
            pbar.set_postfix({'Validation Loss': f'{loss:.4f}'})
    return running_loss/(i+1)


def train_loop(model, epochs, vae, clip, tokenizer, train_loader, val_loader, optimizer, loss_fn, scheduler, lr, warmup_steps):
    # Get regular training working first, then add in the EMA weights
    # Use the tokenizer and clip like so: https://github.com/mlfoundations/open_clip
    # Encoding and decoding with VAE:
        # Example: Load a PIL Image and convert to a PyTorch tensor
# image_path = "path/to/your/image.jpg" # Replace with your image path
# pil_image = Image.open(image_path).convert("RGB")

# # Preprocess the image: resize, convert to tensor, normalize to [-1, 1]
# # The exact preprocessing steps might vary slightly depending on the VAE's training
# image = np.array(pil_image.resize((512, 512))) / 255.0  # Resize and normalize to [0, 1]
# image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float() # HWC to NCHW
# image = image * 2 - 1 # Normalize to [-1, 1]
    # with torch.no_grad():
        # latent_dist = vae.encode(image).latent_dist
        # latents = latent_dist.sample() * vae.config.scaling_factor # Apply scaling factor
#     with torch.no_grad():
#         decoded_image_tensor = vae.decode(latents / vae.config.scaling_factor).sample # Undo scaling factor

# # Post-process the decoded image: denormalize and convert to PIL Image
# decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1) # Denormalize to [0, 1]
# decoded_image = decoded_image_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() # NCHW to HWC
# decoded_image = Image.fromarray((decoded_image * 255).astype(np.uint8))

# # Display or save the decoded image
# decoded_image.save("decoded_image.jpg")
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
    print("Epoch 0:")
    avg_train_loss = validate_one_epoch(model, 0, epochs, train_loader, vae, clip, tokenizer, loss_fn)
    best_val_loss = validate_one_epoch(model, 0, epochs, val_loader, vae, clip, tokenizer, loss_fn)
    wandb.log({"Average Train Loss Per Epoch": avg_train_loss,
               "Average Validation Loss Per Epoch": best_val_loss,
               "Epoch": 0})

    for e in range(epochs):
        print("Epcoh {}:".format(e+1))
        ema_model = ema_model.to("cpu")
        model = model.to("cuda")
        avg_train_loss = train_one_epoch(model, e, epochs, train_loader, vae, clip, tokenizer, optimizer, loss_fn, scheduler, scaler, ema_model)
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
        avg_val_loss = validate_one_epoch(ema_model, e, epochs, val_loader, vae, clip, tokenizer, loss_fn)
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


train_loop(model, 200, vae, clip, clip_tokenizer, train_loader, val_loader, optimizer, loss_fn, scheduler, lr, warmup_steps)

# print("Number of samples: ", len(cap))
# img, target = cap[0]
# print(target)
# with open('../Datasets/COCO/annotations_trainval2017/captions_val2017.json', 'r') as f:
#     data = json.load(f)
# print(len(data["annotations"]))
# print(data["annotations"][0])