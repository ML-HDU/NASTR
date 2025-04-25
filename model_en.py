from functools import partial
import math
import numpy as np

import torch
import torch.nn as nn

from backbone.backbone_vit import VisionTransformerMAERec
from backbone.backbone_svtr import svtr_tiny, svtr_small, svtr_base
from transformer import PositionalEncoding, TransformerDecoderLayer, TransformerDecoder, TransformerEncoderLayer, TransformerEncoder
from dataset.en_dataset import LabelConverter
from pathlib import Path


class NASTR(nn.Module):

    def __init__(self, common_kwargs, encoder_kwargs, decoder_kwargs):
        super(NASTR, self).__init__()

        # Basic configuration
        self.LabelConverter = LabelConverter(Path(common_kwargs['alphabet']), max_length=-1, ignore_over=False)

        self.imgH = common_kwargs['img_h']
        self.imgW = common_kwargs['img_w']

        self.padding_symbol = 0
        self.eos_symbol = 1

        self.n_head = decoder_kwargs['n_head']
        self.dimensions = decoder_kwargs['dimensions']
        self.dropout = decoder_kwargs['dropout']

        self.ITC = decoder_kwargs['ITC']
        if self.ITC:
            self.text_encoder_n_head = decoder_kwargs['text_encoder_kwargs']['n_head']
            self.text_encoder_dimensions = decoder_kwargs['text_encoder_kwargs']['dimensions']
            self.text_encoder_dropout = decoder_kwargs['text_encoder_kwargs']['dropout']
            self.text_encoder_stack = decoder_kwargs['text_encoder_kwargs']['stacks']

        self.nclass = self.LabelConverter.nclass

        if encoder_kwargs['type'] == 'svtr_base':
            self.Encoder = svtr_base(img_size=[self.imgH, self.imgW])
            self.dimensions = 384
        elif encoder_kwargs['type'] == 'vit':

            pretrained_path = r'path/to/pretrained_model.pth'
            # refer to https://github.com/Mountchicken/Union14M 5.2 Fine-tuning, MAERec-S

            self.Encoder = VisionTransformerMAERec(img_size=(32, 128), patch_size=4, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True, pretrained=None)
            self.dimensions = 384

        self.model_backbone = encoder_kwargs['type']

        if self.ITC:
            self.text_encoder_dimensions = self.dimensions

        self.TransformerDecoder = Decoder(self.dimensions, self.n_head, dim_feedforward=4 * self.dimensions,
                                          dropout=self.dropout, num_layers=decoder_kwargs['stacks'])

        if self.ITC:
            self.text_encoder = text_encoder(self.text_encoder_dimensions,
                                             self.text_encoder_n_head,
                                             dim_feedforward=4 * self.text_encoder_dimensions,
                                             dropout=self.text_encoder_dropout,
                                             num_layers=self.text_encoder_stack)

        # Coarse and Fine predictor
        self.fine_predictor = FinePredictor(self.dimensions, self.nclass)
        self.coarse_predictor = CoarsePredictor(embed_dims=self.dimensions, LabelConverter=self.LabelConverter)
        self.length_predictor = LengthPredictor(self.dimensions)

        # Embedding layer
        self.position = PositionalEncoding(self.dimensions, self.dropout)

        self.TextEmbedding = nn.Embedding(self.nclass, self.dimensions)
        self.sqrt_model_size = math.sqrt(self.dimensions)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params

    def forward(self, images, target_lengths, text_target=None):

        batch_max_length = max(target_lengths).cpu().item() + 1  # add "1" represents the <EOS> symbol

        encoder_features = self.Encoder(images)

        N = encoder_features.size()[0]

        if self.model_backbone == "svtr_base":

            _, E, h, w = encoder_features.size()

            seq_encoder_features = torch.flatten(encoder_features, start_dim=2).permute(0, 2, 1)
            seq_encoder_features = self.position(seq_encoder_features).permute(1, 0, 2)  # hw, n, E
            global_features = encoder_features.view(N, E, -1).permute(0, 2, 1).mean(dim=1)

        elif self.model_backbone == "vit":

            seq_encoder_features = encoder_features[:, 1:, :]
            seq_encoder_features = self.position(seq_encoder_features).permute(1, 0, 2)  # hw, n, E

            global_features = encoder_features[:, 0, :]

        length_logits = None

        fused_embedding = global_features.unsqueeze(1).repeat(1, batch_max_length, 1)
        coarse_logits = None

        target_embedding = self.position(fused_embedding).permute(1, 0, 2)  # T (batch_max_length), N, E;  batch_first = False

        seq_decoder_features = self.TransformerDecoder(tgt=target_embedding, memory=seq_encoder_features)
        seq_decoder_features = seq_decoder_features.squeeze().permute(1, 0, 2).contiguous()

        text_encoder_features = None
        if self.ITC:
            text_target_embeddings = self.TextEmbedding(text_target)
            text_target_embeddings = self.position(text_target_embeddings).permute(1, 0, 2)
            text_encoder_features = self.text_encoder(src=text_target_embeddings, text=text_target)
            text_encoder_features = text_encoder_features.squeeze().permute(1, 0, 2).contiguous()[:, :-1, :].mean(dim=1)

        fine_logits = self.fine_predictor(seq_decoder_features)

        outputs = {
            "coarse_logits": coarse_logits,
            'length_logits': length_logits,
            'fine_logits': fine_logits,
            'text_encoder_features': text_encoder_features,
            'vision_encoder_features': global_features
        }

        return outputs


class Decoder(nn.Module):

    def __init__(self, embed_dims, heads, dim_feedforward, dropout, num_layers):
        super().__init__()

        DecoderLayer = TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        self.TransformerDecoder = TransformerDecoder(
            decoder_layer=DecoderLayer,
            num_layers=num_layers
        )

    def forward(self, tgt, memory):
        out = self.TransformerDecoder(tgt, memory)

        return out


class text_encoder(nn.Module):

    def __init__(self, embed_dims, heads, dim_feedforward, dropout, num_layers):
        super().__init__()

        EncoderLayer = TransformerEncoderLayer(
            d_model=embed_dims,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.TransformerEncoder = TransformerEncoder(
            encoder_layer=EncoderLayer,
            num_layers=num_layers
        )

    def build_attention_mask(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, src, text):
        context_length = src.shape[0]

        src_key_padding_mask = text == 0

        out = self.TransformerEncoder(src, src_key_padding_mask=src_key_padding_mask)
        return out


class FinePredictor(nn.Module):
    """
    Define standard linear generation step.
    """

    def __init__(self, embed_dims, vocab_size):
        """
        :param embed_dims: dim of model
        :param vocab_size: size of vocabulary
        """
        super(FinePredictor, self).__init__()
        self.fc = nn.Linear(embed_dims, vocab_size)

    def forward(self, x):
        return self.fc(x)


class CoarsePredictor(nn.Module):
    def __init__(self, embed_dims=512, LabelConverter=None):
        super(CoarsePredictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(embed_dims, LabelConverter.nclass)
        )

    def forward(self, x, max_length_pred):
        logits = self.net(x)

        logits_detach = logits.detach()
        coarse_preds = torch.topk(torch.softmax(logits_detach, dim=-1), k=max_length_pred)

        return logits, coarse_preds[1]


class LengthPredictor(nn.Module):
    def __init__(self, embed_dims, max_length=120):
        super(LengthPredictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(embed_dims, max_length)
        )

    def forward(self, x):
        logits = self.net(x)

        logits_detach = logits.detach()
        length_preds = torch.argmax(torch.softmax(logits_detach, dim=-1), dim=-1)

        return logits, length_preds


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x