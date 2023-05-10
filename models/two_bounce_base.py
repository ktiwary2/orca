from utils.print_fn import log
from utils.logger import Logger

import math
import numbers
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch import autograd
import torch.nn.functional as F

from .base import get_embedder, SirenLayer, DenseLayer

class LightFieldNetV1(nn.Module):
    def __init__(self,
        embedding_fn=None,
        D=4,
        W=256,
        skips=[],
        use_positional_embedding=True,
        embed_multires_view=4,
        embed_multires_pos=4,
        weight_norm=True,
        use_siren=False,
        final_act='sigmoid'):
        super().__init__()

        # input_ch_pts = 3
        input_ch_views = 3
        if use_siren:
            assert len(skips) == 0, "do not use skips for siren"
        self.skips = skips
        self.D = D
        self.W = W
        self.use_positional_embedding = use_positional_embedding

        if embedding_fn == None:
            self.embed_fn_view, input_ch_views = get_embedder(embed_multires_view)
            in_dim_0 = input_ch_views + 1
            self.sh_embed = False
            if self.use_positional_embedding:
                self.embed_fn_pos, input_ch_pos = get_embedder(embed_multires_pos)
                in_dim_pos_0 = input_ch_pos
            else:
                in_dim_pos_0 = 3
            # add all input dimensions up
            in_dim_0 += in_dim_pos_0
        else:
            self.embed_fn_view = embedding_fn 
            self.L = embed_multires_view
            in_dim_0 = 2**(self.L+2) + self.L - 1
            if self.use_positional_embedding:
                print("Embedding position with multires: {}".format(embed_multires_pos))
                self.embed_fn_pos, input_ch_pos = get_embedder(embed_multires_pos)
                in_dim_pos_0 = input_ch_pos 
            else:
                in_dim_pos_0 = 3
            # add all input dimensions up
            in_dim_0 += in_dim_pos_0
            self.sh_embed = True

        fc_layers = []
        print("Total input-embedding Dimension: {}".format(in_dim_0))
        # NOTE: as in IDR/NeuS, the network's has D+1 layers
        for l in range(D + 1):
            # decicde out_dim
            if l == D:
                out_dim = 3
            else:
                out_dim = W

            # decide in_dim
            if l == 0:
                in_dim = in_dim_0
            elif l in self.skips:
                in_dim = in_dim_0 + W
            else:
                in_dim = W

            if l != D:
                if use_siren:
                    layer = SirenLayer(in_dim, out_dim, is_first=(l==0))
                else:
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU(inplace=True))
            else:
                if final_act == 'sigmoid':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Sigmoid())
                elif final_act == 'swish':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.SiLU())
                elif final_act == 'relu':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.ReLU())
                elif final_act == 'identity':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Identity())
                elif final_act == 'softplus':
                    layer = DenseLayer(in_dim, out_dim, activation=nn.Softplus())

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            fc_layers.append(layer)

        self.layers = nn.ModuleList(fc_layers)

    def forward(
        self, 
        x, 
        view_dirs: torch.Tensor,
        roughness: torch.Tensor):

        if self.use_positional_embedding: 
            x = self.embed_fn_pos(x)

        if self.sh_embed:
            view_dirs_embed = self.embed_fn_view(dirs=view_dirs, roughness=roughness[...,0], L=self.L)
        else:
            view_dirs_embed = torch.cat([x, self.embed_fn_view(view_dirs), roughness], dim=-1)

        # print("lightfieldnet: x: {}, view: {}".format(x.shape, view_dirs_embed.shape))
        radiance_input = torch.cat([x, view_dirs_embed], dim=-1)
        # print("lightfieldnet: radiance: {}".format(radiance_input.shape))

        h = radiance_input
        for i in range(self.D+1):
            if i in self.skips:
                h = torch.cat([h, radiance_input], dim=-1)
            h = self.layers[i](h)
        return h