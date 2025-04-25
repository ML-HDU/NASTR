import os
import numpy as np
import torch


def predict(model, images, max_length):

    model.eval()
    with torch.no_grad():

        encoder_features = model.Encoder(images)
        
        N = encoder_features.size()[0]
        
        if model.model_backbone == 'svtr_base':

            _, E, h, w = encoder_features.size()

            seq_encoder_features = torch.flatten(encoder_features, start_dim=2).permute(0, 2, 1)
            seq_encoder_features = model.position(seq_encoder_features).permute(1, 0, 2)

            global_features = encoder_features.view(N, E, -1).permute(0, 2, 1).mean(dim=1)
        
        elif model.model_backbone == 'vit':
            
            seq_encoder_features = encoder_features[:, 1:, :]
            seq_encoder_features = model.position(seq_encoder_features).permute(1, 0, 2)

            global_features = encoder_features[:, 0, :]

        length_logits = None
        
        fused_embedding = global_features.unsqueeze(1).repeat(1, max_length, 1)
        coarse_preds = None

        target_embedding = model.position(fused_embedding).permute(1, 0, 2)  # T (max_length), N, E;  batch_first = False

        seq_decoder_features = model.TransformerDecoder(tgt=target_embedding, memory=seq_encoder_features)

        if len(seq_decoder_features.squeeze().size()) == 2:
            # batch size is 1 when testing, so we need unsqueeze the dimension of batch
            seq_decoder_features = seq_decoder_features.squeeze().unsqueeze(0)
            seq_decoder_features = seq_decoder_features.contiguous()
        else:
            seq_decoder_features = seq_decoder_features.squeeze().permute(1, 0, 2).contiguous()

        fine_logits = model.fine_predictor(seq_decoder_features)

        fine_predictions = torch.softmax(fine_logits, dim=-1)

        fine_confs, fine_preds = torch.max(fine_predictions, dim=-1)

        return fine_preds, coarse_preds, length_logits, seq_decoder_features
