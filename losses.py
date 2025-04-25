
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter


def get_loss(input, target, tau=0.2):
    def _flatten(sources, lengths):
        return torch.cat([t[:l + 1] for t, l in zip(sources, lengths)])

    length_preds = input['length']
    length_target = target['length']

    coarse_preds = input['coarse']
    fine_preds = input['fine']
    text_target = target['text']

    flatten_fine_preds = _flatten(fine_preds, length_target)
    flatten_text_target = _flatten(text_target, length_target)

    if length_preds is not None:
        loss_length = get_loss_for_length(length_preds, length_target)
    elif length_preds is None:
        loss_length = torch.tensor(0.).to(text_target.device)

    if coarse_preds is not None:
        soft_label = torch.zeros_like(coarse_preds)
        soft_label_prob = torch.stack([torch.tensor(1., device=text_target.device) / len(torch.unique(text_target[i, :length])) for i, length in enumerate(length_target)])

        for i, (target, length) in enumerate(zip(text_target, length_target)):
            soft_label_index = torch.unique(target[:length])

            counter = dict(Counter(target.tolist()))

            for idx in soft_label_index.tolist():
                soft_label[i, idx] = counter[idx] / length

        loss_coarse = F.kl_div(F.log_softmax(coarse_preds, dim=-1), soft_label, reduction='batchmean')

    elif coarse_preds is None:
        loss_coarse = torch.tensor(0.).to(text_target.device)

    loss_fine = get_loss_for_fine(flatten_fine_preds, flatten_text_target)

    loss_ITC = torch.tensor(0.).to(loss_fine.device)

    vision_global_features, text_global_features = input['vision_global_features'], input['text_global_features']

    if vision_global_features is not None and text_global_features is not None:
        loss_ITC = get_loss_for_ITC(vision_global_features, text_global_features, temperature=tau)

    return loss_fine, loss_coarse, loss_length, loss_ITC


def get_loss_for_length(input, target):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    return criterion(input, target)


def get_loss_for_fine(input, target):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    return criterion(input, target)


def get_loss_for_ITC(image_feat, text_feat, temperature):
    criterion = nn.CrossEntropyLoss(reduction='mean')

    image_feat = image_feat.squeeze()
    text_feat = text_feat.squeeze()

    image_feat = torch.nn.functional.normalize(image_feat, dim=1)
    text_feat = torch.nn.functional.normalize(text_feat, dim=1)

    logits = torch.matmul(image_feat, text_feat.t())
    logits /= temperature

    gt = torch.arange(logits.shape[0], device=logits.device)
    loss_1 = criterion(logits, gt)
    loss_2 = criterion(logits.t(), gt)

    return (loss_1 + loss_2) / 2