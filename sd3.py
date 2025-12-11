import math
import torch
from torch import nn
from tqdm import tqdm


class SinusoidalEncoding(nn.Module):

    def __init__(self, dim=320):
        super().__init__()
        self._dim = dim
        self._frequency = math.log(10000)

    def forward(self, x):
        # x: B
        device = x.device
        sin_embeddings = torch.arange(0, self._dim, 2, device=device) * self._frequency
        sin_embeddings = torch.sin(x[:, None] / torch.exp(sin_embeddings)[None, :])  # B x self._dim
        cos_embeddings = torch.arange(1, self._dim, 2, device=device) * self._frequency
        cos_embeddings = torch.cos(x[:, None] / torch.exp(cos_embeddings)[None, :])  # B x self._dim
        embeddings = torch.stack([sin_embeddings, cos_embeddings], dim=2).view(x.size(0), -1)
        return embeddings


class MLP(nn.Module):

    def __init__(self, dim_1, dim_2, dim_3):
        super().__init__()
        self._linear_1 = nn.Linear(dim_1, dim_2)
        self._linear_2 = nn.Linear(dim_2, dim_3)
        self._activation = nn.SiLU()

    def forward(self, x):
        x = self._linear_1(x)
        x = self._activation(x)
        x = self._linear_2(x)
        return x


class CustomModulatedLayerNorm(nn.Module):

    def __init__(self, x_dim, q_dim):
        super().__init__()
        self._ln = nn.LayerNorm(x_dim, elementwise_affine=False)
        self._linear = nn.Linear(x_dim, q_dim*3)
        self._rms_q = nn.RMSNorm([q_dim])
        self._rms_k = nn.RMSNorm([q_dim])

    def forward(self, x, a_x, b_x):
        # x: B x num_patches x embed
        # a_x: B x embed
        x = self._ln(x)
        x = a_x[:,None,:] * x + b_x[:,None,:]
        x = self._linear(x)
        q, k, v = x.chunk(3, dim=-1)
        q = self._rms_q(q)
        k = self._rms_k(k)
        return q, k, v


class CustomMLP(nn.Module):

    def __init__(self, dim_1, dim_2, dim_3):
        super().__init__()
        self._linear = nn.Linear(dim_1, dim_2)
        self._ln = nn.LayerNorm(dim_2, elementwise_affine=False)
        self._mlp = MLP(dim_2, dim_3, dim_2)

    def forward(self, xc, x, g, d, e, z):
        # xc: B x num_patches x embed
        # g: B x embed
        x_1 = self._linear(xc)
        x_1 = x_1 * g[:,None,:]
        x_1 = x_1 + x
        x_1 = self._ln(x_1)
        x_1 = d[:,None,:] * x_1 + e[:,None,:]
        x_1 = self._mlp(x_1)
        x_1 = x_1 * z[:,None,:]
        x = x_1 + x
        return x


class MMDiTBlock(nn.Module):

    def __init__(self, x_dim, y_dim, n_heads, head_dim, mlp_dim):
        super().__init__()
        self._activation = nn.SiLU()
        self._linear_y_x = nn.Linear(y_dim, 6*x_dim)
        self._linear_y_c = nn.Linear(y_dim, 6*x_dim)
        self._m_layer_norm_x = CustomModulatedLayerNorm(x_dim, n_heads*head_dim)
        self._m_layer_norm_c = CustomModulatedLayerNorm(x_dim, n_heads*head_dim)
        self._atten = nn.MultiheadAttention(n_heads*head_dim, n_heads, batch_first=True)
        self._mlp_x = CustomMLP(n_heads*head_dim, x_dim, mlp_dim)
        self._mlp_c = CustomMLP(n_heads*head_dim, x_dim, mlp_dim)

    def forward(self, xcy):
        # x: B x num_patches x embed
        # y: B x 320
        # c: B x 77 x 128
        x, c, y = xcy

        y_1 = self._activation(y)

        y_x = self._linear_y_x(y_1)  # B x 6*x_dim
        a_x, b_x, g_x, d_x, e_x, z_x = y_x.chunk(6, dim=-1)
        q_x, k_x, v_x = self._m_layer_norm_x(x, a_x, b_x)  # B x num_patches x head_dim*n_heads

        y_c = self._linear_y_c(y_1)
        a_c, b_c, g_c, d_c, e_c, z_c = y_c.chunk(6, dim=-1)
        q_c, k_c, v_c = self._m_layer_norm_c(c, a_c, b_c)  # B x 77 x head_dim*n_heads

        q = torch.cat([q_x, q_c], dim=1)
        k = torch.cat([k_x, k_c], dim=1)
        v = torch.cat([v_x, v_c], dim=1)
        xc = self._atten(q, k, v, need_weights=False)[0]  # B x (num_patches + 77) x head_dim*n_heads
        x_1, c_1 = torch.split(xc, [x.size(1), c.size(1)], dim=1)
        x = self._mlp_x(x_1, x, g_x, d_x, e_x, z_x)
        c = self._mlp_c(c_1, c, g_c, d_c, e_c, z_c)

        return x, c, y


class Modulation(nn.Module):

    def __init__(self, x_dim, y_dim):
        super().__init__()
        self._activation = nn.SiLU()
        self._linear = nn.Linear(y_dim, x_dim*2)

    def forward(self, x, y):
        # x: B x num_patches x x_dim
        # y: B x 320
        y = self._activation(y)
        y = self._linear(y)  # B x 2*x_dim
        a, b = y.chunk(2, dim=-1)
        x = a[:,None,:] * x + b[:,None,:]
        return x


class DiT(nn.Module):

    def __init__(self):
        super().__init__()
        # Image sizes will be 128 x 128
        self._x_dim = 128
        self._clip_dim = 512
        self._linear_1 = nn.Linear(self._clip_dim, self._x_dim)
        # latent is H/8 x W/8 x 16
        self._kernel_size = 2
        self._latent_dim = 16
        self._linear_2 = nn.Linear(self._kernel_size * self._kernel_size * self._latent_dim, self._x_dim)
        self._linear_3 = nn.Linear(self._x_dim, self._kernel_size * self._kernel_size * self._latent_dim)
        self._t_dim = 320
        self._sinusoidal_embedding = SinusoidalEncoding(self._t_dim)
        self._num_patches = 64
        self._positional_embedding = nn.Parameter(torch.zeros(1, self._num_patches, self._x_dim))
        self._patching = nn.Unfold(kernel_size=(self._kernel_size, self._kernel_size), stride=(self._kernel_size, self._kernel_size))
        self._latent_height = 16
        self._latent_width = 16
        self._unpatching = nn.Fold((self._latent_height, self._latent_width),
                                   (self._kernel_size, self._kernel_size),
                                   stride=(self._kernel_size, self._kernel_size))
        self._depth = 18
        self._mlp_dim = 4*64*self._depth
        self._mlp_1 = MLP(self._clip_dim, self._mlp_dim, self._t_dim)
        self._mlp_2 = MLP(self._t_dim, self._mlp_dim, self._t_dim)
        self._head_dim = 64
        self._mm_dit = nn.Sequential()
        for i in range(self._depth):
            self._mm_dit.append(MMDiTBlock(self._x_dim, self._t_dim, self._depth, self._head_dim, self._mlp_dim))
        self._modulation = Modulation(self._x_dim, self._t_dim)

    def forward(self, latent, ctx, ctx_pool, t):
        # latent is B x 16 x H/8 x W/8 (128/8 = 16)
        # 64 patches
        # ctx: B x 77
        c = self._linear_1(ctx)  # B x 77 x 128
        y = self._mlp_1(ctx_pool) + self._mlp_2(self._sinusoidal_embedding(t))  # B x 320
        divisor = self._unpatching(self._patching(torch.ones(latent.shape, dtype=latent.dtype, device=y.device)))
        latent = self._patching(latent).permute(0, 2, 1)  # B x num_patches x k^2 * 16
        latent = self._linear_2(latent)  # B x num_patches x embed
        x = self._positional_embedding + latent
        x, _, _ = self._mm_dit([x, c, y])
        x = self._modulation(x, y)  # B x num_patches x x_dim; B x 320; output: B x num_patches x x_dim
        x = self._linear_3(x)  # B x num_patches x k^2 * 16
        x = x.permute(0, 2, 1)  # B x k^2 * 16 x num_patches
        x = self._unpatching(x) * divisor
        return x


class SD3(nn.Module):

    def __init__(self, clip, tokenizer, vae, dit, device="cuda"):
        super().__init__()
        self._device = device
        self._clip = clip.to(self._device)
        self._tokenizer = tokenizer
        self._vae = vae.to(self._device)
        self._dit = dit.to(self._device)

    def forward(self, text, batch_size=1, num_steps=100):
        with torch.inference_mode():
            tokens = self._tokenizer(text).to(self._device)
            tokens = tokens.repeat(batch_size, 1)
            text_embeddings = self._clip.forward_intermediates(text=tokens, text_indices=1, normalize_intermediates=True)
            ctx = text_embeddings["text_intermediates"][-1]
            ctx_pool = text_embeddings["text_features"]
            latents = torch.randn((batch_size, 16, 16, 16), device=self._device)
            timesteps = torch.linspace(1, 0, num_steps+1, device=self._device)
            dt = -1 / num_steps
            for step in tqdm(timesteps[:-1]):
                velocities = self._dit(latents, ctx, ctx_pool, step[None])
                latents = latents + dt * velocities
            decoded_image_tensor = self._vae.decode(latents / self._vae.config.scaling_factor).sample  # undo scaling factor
            decoded_image_tensor = (decoded_image_tensor / 2 + 0.5).clamp(0, 1)  # denormalize to [0, 1]
            decoded_image = decoded_image_tensor.permute(0,2,3,1).cpu().numpy() # NCHW to NHWC
            return decoded_image
