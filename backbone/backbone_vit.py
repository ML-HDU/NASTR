# Scene Text Recognition Model Hub
# Copyright 2022 Darwin Bautista
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from timm.models.vision_transformer import PatchEmbed, VisionTransformer

from functools import partial
from typing import Tuple
import torch
import torch.nn as nn


class VisionTransformerMAERec(VisionTransformer):
    """ Vision Transformer.

    Args:
        global_pool (bool): If True, apply global pooling to the output
            of the last stage. Default: False.
        patch_size (int): Patch token size. Default: 8.
        img_size (tuple[int]): Input image size. Default: (32, 128).
        embed_dim (int): Number of linear projection output channels.
            Default: 192.
        depth (int): Number of blocks. Default: 12.
        num_heads (int): Number of attention heads. Default: 3.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key,
            value. Default: True.
        norm_layer (nn.Module): Normalization layer. Default:
            partial(nn.LayerNorm, eps=1e-6).
        pretrained (str): Path to pre-trained checkpoint. Default: None.
    """

    def __init__(self,
                 global_pool: bool = False,
                 patch_size: int = 4,
                 img_size: Tuple[int, int] = (32, 128),
                 embed_dim: int = 192,
                 depth: int = 12,
                 num_heads: int = 3,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 pretrained: bool = None,
                 **kwargs):
        super(VisionTransformerMAERec, self).__init__(
            patch_size=patch_size,
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            **kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm
        self.reset_classifier(0)

        if pretrained:
            checkpoint = torch.load(pretrained, map_location='cpu')

            print("Load pre-trained checkpoint from: %s" % pretrained)

            checkpoint_model = checkpoint['model']
            state_dict = self.state_dict()
            for k in ['head.weight', 'head.bias']:
                if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint")
                    del checkpoint_model[k]
            # remove key with decoder
            for k in list(checkpoint_model.keys()):
                if 'decoder' in k:
                    del checkpoint_model[k]
            msg = self.load_state_dict(checkpoint_model, strict=False)
            print(msg)

    def forward_features(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(
            B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward(self, x):
        return self.forward_features(x)