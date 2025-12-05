import matplotlib.pyplot as plt
import torch


def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False


def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    print(f"Element size: {param.element_size()}")
    print(f"Dtype: {param.dtype}")
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    total_size_bytes = param_size + buffer_size
    total_size_mb = total_size_bytes / (1024**2)
    print(f"Total model size: {total_size_mb:.2f} MB")


def get_vae_latent(image, vae):
    latent = vae.encode(image).latent_dist.mean
    latent *= vae.config.scaling_factor
    return latent


def sample_time(num_samples, device="cpu"):
    with torch.no_grad():
        # Logit-Normal Sampling
        return torch.sigmoid(torch.randn(num_samples, device=device))


def show_first_image(data_loader):
    img, _ = next(iter(data_loader))
    image_np = img[0].detach().permute(1, 2, 0).numpy() / 2 + 0.5
    plt.imshow(image_np)
    plt.show()
    return img


def vae_reconstruction(img, vae):
    with torch.inference_mode():
        latent = vae.encode(img[0].unsqueeze(0)).latent_dist.mean * vae.config.scaling_factor
        decoded_image_tensor = vae.decode(latent / vae.config.scaling_factor).sample  # Undo scaling factor
    decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)  # Denormalize to [0, 1]
    decoded_image = decoded_image_tensor.squeeze(0).permute(1, 2, 0).cpu()  # NCHW to HWC
    return decoded_image


def test_vae_reconstruction_quality(data_loader, vae):
    img_tensor = show_first_image(data_loader)
    reconstructed_img = vae_reconstruction(img_tensor, vae).numpy()
    plt.imshow(reconstructed_img)
    plt.show()
