import math
import torch
from torch import nn
from copy import deepcopy

from helper import ceiln, variance_scaling_init_


class TimeEmbedding(nn.Module):
    """
    improved based on
    https://github.com/lucidrains/denoising-diffusion-pytorch
    move some computations to init
    """

    def __init__(self, dim, max_step):
        super().__init__()

        assert dim % 2 == 0

        time_steps = torch.arange(max_step).unsqueeze(1)
        i = torch.arange(0, dim, 2).unsqueeze(0)
        embedding = time_steps / 10000 ** (i / dim)

        te = torch.empty((max_step, dim), requires_grad=False)
        te[:, 0::2] = embedding.sin()
        te[:, 1::2] = embedding.cos()
        self.register_buffer('te', te)  # register so that it will move to device

        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim * 4)

        self.act = nn.SiLU(inplace=True)

    def forward(self, t):
        embeddings = self.te[t]
        embeddings = self.act(self.linear1(embeddings))
        embeddings = self.linear2(embeddings)
        return embeddings


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim,
                 dropout=0., num_groups=32):
        super().__init__()

        assert in_channels % num_groups == out_channels % num_groups == 0

        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout(dropout, inplace=True)

        self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.embed_linear = nn.Linear(time_dim, out_channels * 2)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = None

    def forward(self, x, embedding, residual_queue=None):
        if residual_queue is not None:
            x = torch.cat((x, residual_queue.pop()), dim=1)

        shortcut = x
        if self.shortcut is not None:
            shortcut = self.shortcut(shortcut)
        x = self.act(self.norm1(x))
        x = self.conv1(x)

        embedding = self.embed_linear(embedding)[..., None, None]
        scale, shift = torch.chunk(embedding, 2, dim=1)
        x = self.norm2(x)
        x *= (scale + 1.)
        x += shift

        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)

        return x + shortcut


class Upsample(nn.Module):
    def __init__(self, channels, method='conv'):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        if method in ['conv', 'patch']:
            self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        else:
            self.conv = None

    def forward(self, x, embedding=None, residual_queue=None):
        x = self.up(x)
        if self.conv is not None:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, channels, method='conv'):
        super().__init__()

        if method == 'conv':
            self.down = nn.Conv2d(channels, channels, 3, stride=2, padding=1)
        elif method == 'patch':
            self.down = nn.Conv2d(channels, channels, 2, stride=2)
        elif method == 'avg':
            self.down = nn.AvgPool2d(2, stride=2)
        elif method == 'max':
            self.down = nn.MaxPool2d(2, stride=2)
        else:
            raise ValueError(method)

    def forward(self, x, embedding=None, residual_queue=None):
        return self.down(x)


class NonLocalSelfAttention(nn.Module):
    def __init__(self, in_channels, time_dim,
                 heads=4, head_dim=32, num_groups=32):
        super().__init__()

        self.norm = nn.GroupNorm(num_groups, in_channels)
        self.in_proj = nn.Conv2d(in_channels, head_dim * heads * 3, 1)
        self.out_proj = nn.Conv2d(head_dim * heads, in_channels, 1)
        self.softmax = nn.Softmax(-1)

        self.heads = heads
        self.head_dim = head_dim
        self.scale = 1. / math.sqrt(head_dim)

        self.embed_linear = nn.Linear(time_dim, in_channels)

    def forward(self, x, embedding, residual_queue=None):
        b, _, h, w = x.shape

        qkv = self.in_proj(self.norm(x))
        qkv = torch.chunk(qkv, 3, dim=1)
        q, k, v = map(lambda m: m.view(b, self.heads, self.head_dim, -1), qkv)

        qk = torch.einsum('bhcm,bhcn->bhmn', q, k)
        qk *= self.scale
        qk = self.softmax(qk)
        qkv = torch.einsum('bhmn,bhcn->bhcm', qk, v)
        qkv = qkv.view(b, self.heads, self.head_dim, h, w)
        qkv = qkv.reshape((b, self.heads * self.head_dim, h, w))

        qkv = self.out_proj(qkv)
        embedding = self.embed_linear(embedding)[..., None, None]
        qkv *= embedding

        return qkv + x


class DiffusionModel(nn.Module):
    def __init__(self,
                 max_step,
                 img_channels=3,
                 init_channels=64,
                 stages=(1, 2, 4, 8),
                 num_res=1,
                 cond_len=0,
                 cond_num_emb=0,
                 dropout=0.,
                 num_groups=32,
                 resize_method='conv'):
        super().__init__()

        up_channels = [ceiln(init_channels * s, num_groups) for s in stages]
        down_channels = list(reversed(up_channels))

        down_blocks = [nn.Conv2d(img_channels, init_channels, 3, padding=1)]
        c_temps = [init_channels]
        c_in = init_channels
        for i, c_out in enumerate(up_channels):
            for _ in range(num_res):
                down_blocks.append(ResBlock(c_in, c_out, init_channels * 4,
                                            dropout=dropout, num_groups=num_groups))
                c_temps.append(c_out)
                c_in = c_out
            if i != len(up_channels) - 1:
                down_blocks.append(Downsample(c_out, resize_method))
                c_temps.append(c_out)

        mid_blocks = []
        for _ in range(num_res):
            mid_blocks.append(ResBlock(c_in, c_in, init_channels * 4,
                                       dropout=dropout, num_groups=num_groups))
            mid_blocks.append(NonLocalSelfAttention(c_in, init_channels * 4,
                                                    heads=4, head_dim=64, num_groups=num_groups))

        up_blocks = []
        for i, c_out in enumerate(down_channels):
            for _ in range(num_res + 1):
                up_blocks.append(ResBlock(c_in + c_temps.pop(), c_out, init_channels * 4,
                                          dropout=dropout, num_groups=num_groups))
                c_in = c_out
            if i != len(down_channels) - 1:
                up_blocks.append(Upsample(c_out, resize_method))
        assert len(c_temps) == 0

        self.act = act = nn.SiLU(inplace=True)

        self.time_embedding = TimeEmbedding(init_channels, max_step)
        if cond_len:
            assert cond_num_emb > 0
            assert init_channels * 4 % cond_len == 0
            self.cond_embedding = nn.Sequential(
                nn.Embedding(cond_num_emb, init_channels * 4 // cond_len),
                nn.Flatten()
            )
        self.embedding = nn.Sequential(
            act,
            nn.Linear(init_channels * 4, init_channels * 4),
            act
        )
        self.down = nn.ModuleList(down_blocks)
        self.mid = nn.ModuleList(mid_blocks)
        self.up = nn.ModuleList(up_blocks)
        self.out = nn.Sequential(
            nn.GroupNorm(num_groups, c_in),
            act,
            nn.Conv2d(c_in, img_channels, 3, padding=1)
        )

        self.reset_weights()

    @torch.no_grad()
    def reset_weights(self):

        def _reset(module):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                variance_scaling_init_(module.weight)
                if getattr(module, 'bias') is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm) and module.affine:
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        self.apply(_reset)

        def _reset_special(module):
            if isinstance(module, ResBlock):
                variance_scaling_init_(module.conv2.weight, scale=1e-9)
            if isinstance(module, NonLocalSelfAttention):
                nn.init.zeros_(module.embed_linear.weight)

        self.apply(_reset_special)

        variance_scaling_init_(self.out[-1].weight, 1e-9)

    def forward(self, x, t, c=None):
        shape = x.shape
        embedding = self.time_embedding(t)
        if c is not None:
            embedding += self.cond_embedding(c)
        embedding = self.embedding(embedding)

        x = self.down[0](x)
        residual_queue = [x]
        for module in self.down[1:]:
            x = module(x, embedding=embedding)
            residual_queue.append(x)

        for module in self.mid:
            x = module(x, embedding=embedding)

        for module in self.up:
            x = module(x, embedding=embedding, residual_queue=residual_queue)

        x = self.out(x)

        assert len(residual_queue) == 0, len(residual_queue)
        assert x.shape == shape, x.shape

        return x


class EMA:
    def __init__(self, scale=0.9999):
        self.scale = scale
        self.shadow = None

    @torch.no_grad()
    def update(self, model):
        if self.shadow is not None:
            for k, v in model.named_parameters():
                v *= self.scale
                v += (1. - self.scale) * self.shadow[k]
        self.shadow = {k: v.detach().clone() for k, v in model.named_parameters()}
