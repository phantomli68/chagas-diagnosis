import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath
from functools import partial
from typing import Optional, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class LayerNorm1d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 2, 1).contiguous()
        return x

class StemLayer1D(nn.Module):
    def __init__(self, in_chans=12, out_chans=96, act_layer='GELU', norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=in_chans, out_channels=out_chans // 2,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = LayerNorm1d(out_chans // 2)
        self.act = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=out_chans // 2, out_channels=out_chans,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = LayerNorm1d(out_chans)
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x


class Heat1D(nn.Module):
    def __init__(self, infer_mode=False, res=2800, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

    def infer_init_heat1d(self, freq):
        weight_exp = self.get_decay_map_1d(self.res, device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, None], self.to_k(freq)), requires_grad=False)
        del self.to_k



    @staticmethod
    def get_cos_map(N=154, device=torch.device("cpu"), dtype=torch.float):
        # DCT matrix (N, N)
        x = (torch.arange(N, device=device, dtype=dtype) + 0.5) / N  # (N,)
        n = torch.arange(N, device=device, dtype=dtype).unsqueeze(1)  # (N, 1)
        cos_matrix = torch.cos(n * x * math.pi) * math.sqrt(2 / N)  # (N, N)
        cos_matrix[0, :] /= math.sqrt(2)
        return cos_matrix  # (N, N)


    @staticmethod
    def get_decay_map(length=154, device=torch.device("cpu"), dtype=torch.float):
        # exp(-(nπ/a)^2)
        # returns: (length,)
        weight = torch.linspace(0, torch.pi, length + 1, device=device, dtype=dtype)[:length]
        weight = torch.pow(weight, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, L = x.shape  #  [Batch, Channel, Length]

        x = self.dwconv(x)  # [B, hidden_dim, L]

        x = self.linear(x.permute(0, 2, 1).contiguous())  # -> [B, L, 2C]
        x, z = x.chunk(chunks=2, dim=-1)

        if (L == getattr(self, "__RES__", 0)) and (getattr(self, "__WEIGHT_COS__", None).device == x.device):
            weight_cos = getattr(self, "__WEIGHT_COS__")
            weight_exp = getattr(self, "__WEIGHT_EXP__")
        else:
            weight_cos = self.get_cos_map(L, device=x.device).detach()  # (L, L)
            weight_exp = self.get_decay_map(L, device=x.device).detach()  # (L,)
            setattr(self, "__RES__", L)
            setattr(self, "__WEIGHT_COS__", weight_cos)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N = weight_cos.shape[0]



        x = F.conv1d(x, weight_cos.view(N, L, 1))  # [B, C, N]
        x = x.permute(0, 2, 1).contiguous()  # [B, N, C]

        if self.infer_mode:
            x = x * self.k_exp
        else:

            weight = torch.pow(weight_exp[:, None], self.to_k(freq_embed))
            x = x.permute(0, 2, 1)

            x = x * weight

        x = F.conv1d(x, weight_cos.t().contiguous().view(L, N, 1))

        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 2, 1).contiguous()
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv1d, kernel_size=1, padding=0) if channels_first else nn.Linear

        self.fc1 = nn.Conv1d(in_channels=in_features, out_channels=hidden_features, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(in_channels=hidden_features, out_channels=out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class HeatBlock1D(nn.Module):
    def __init__(
            self,
            dim: int,
            res: int = 128,
            infer_mode=False,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm1d, eps=1e-6),
            use_checkpoint: bool = False,
            drop: float = 0.0,
            act_layer: nn.Module = nn.GELU,
            mlp_ratio: float = 4.0,
            post_norm=True,
            layer_scale=None,
            **kwargs,
    ):
        super().__init__()
        hidden_dim = hidden_dim or dim


        self.use_checkpoint = use_checkpoint
        self.norm1 = LayerNorm1d(hidden_dim)

        self.op = Heat1D(seq_len=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = LayerNorm1d(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
                channels_first=False
            )

        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        self.infer_mode = infer_mode

        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim), requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed=None):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x)))
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

        if self.post_norm:
            x = x + self.drop_path(self.gamma1[None, None, :] * self.norm1(self.op(x, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[None, None, :] * self.norm2(self.mlp(x)))
        else:
            x = x + self.drop_path(self.gamma1[None, None, :] * self.op(self.norm1(x), freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[None, None, :] * self.mlp(self.norm2(x)))
        return x

    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)

class vHeat1D(nn.Module):
    def __init__(self, patch_size=4, in_chans=12, num_classes=1,
                 depths=[2, 2, 18, 2], dims=[128, 256, 512, 1024], drop_path_rate=0.1,
                 patch_norm=True, post_norm=True, layer_scale=None,
                 use_checkpoint=False, mlp_ratio=4.0, seq_len=2800,
                 act_layer='GELU', infer_mode=False, **kwargs):
        super().__init__()
        self.infer_mode = infer_mode

        self.num_classes = num_classes
        self.num_layers = len(depths)

        if isinstance(dims, int):
            dims = [int(dims * 2 ** i) for i in range(self.num_layers)]

        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        self.depths = depths

        # 1D patch embedding layer
        self.patch_embed = StemLayer1D(in_chans=in_chans,
                                       out_chans=self.embed_dim,
                                       act_layer='GELU',
                                       norm_layer='LN')

        # 1D sequence resolution (length)
        res0 = seq_len // patch_size
        self.res = [math.ceil(res0), math.ceil(res0 // 2), math.ceil(res0 // 4), math.ceil(res0 // 8)+1]

        # drop path schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 1D frequency embedding
        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            self.freq_embed.append(nn.Parameter(torch.zeros(self.res[i], self.dims[i]), requires_grad=True))
            trunc_normal_(self.freq_embed[i], std=.02)

        # Heat blocks per layer
        self.layers = nn.ModuleList()
        for i in range(self.num_layers):
            self.layers.append(self.make_layer(
                res=self.res[i],
                dim=self.dims[i],
                depth=self.depths[i],
                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=nn.LayerNorm,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=self.make_downsample(self.dims[i], self.dims[i + 1]) if i < self.num_layers - 1 else nn.Identity(),
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode
            ))

        self.classifier = nn.Sequential(
            LayerNorm1d(self.num_features),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm1d):
        return nn.Sequential(
            nn.Conv1d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )

    @staticmethod
    def make_layer(res, dim, depth, drop_path, use_checkpoint,
                   norm_layer, post_norm, layer_scale, downsample, mlp_ratio, infer_mode):
        blocks = []
        for d in range(depth):
            blocks.append(HeatBlock1D(dim=dim,
                                      drop_path=drop_path[d],
                                      norm_layer=norm_layer,
                                      post_norm=post_norm,
                                      layer_scale=layer_scale,
                                      mlp_ratio=mlp_ratio,
                                      infer_mode=infer_mode))
        return AdditionalInputSequential(*blocks, downsample)

    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_heat1d(self.freq_embed[i])
        del self.freq_embed

    def forward_features(self, x):
        x = self.patch_embed(x)  # B x C x L
        if self.infer_mode:
            for layer in self.layers:
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, self.freq_embed[i])
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    from fvcore.nn import flop_count_table, flop_count_str, FlopCountAnalysis
    model = vHeat1D().cuda()
    input = torch.randn((1, 12, 2800), device=torch.device('cuda'))
    analyze = FlopCountAnalysis(model, (input,))
    print(flop_count_str(analyze))