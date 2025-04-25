import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import collections
import os
import random
import time

import zhconv
import distance

import numpy as np
from pathlib import Path
import torch
import torch.utils.data

import model as CustomModel
import dataset.ch_dataset as CustomDataset
from dataset.ch_dataset import LabelConverter
from parse_config import ConfigParser
from utils import TensorboardWriter, AverageMetricTracker

from tester_STR import predict


def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248

        rstring += chr(inside_code)

    return rstring


def main(config: ConfigParser, logger=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    checkpoint_path = config['tester']['checkpoint_path']

    test_batch_size = config['tester']['test_batch_size']
    test_num_workers = config['tester']['test_num_workers']

    # setup  dataset and data_loader instances
    img_w = config['test_dataset']['args']['img_w']
    img_h = config['test_dataset']['args']['img_h']
    in_channels = config['model_arch']['args']['common_kwargs']['in_channels']
    convert_to_gray = False if in_channels == 3 else True

    test_dataset = config.init_obj('test_dataset', CustomDataset,
                                    transform=CustomDataset.CustomImagePreprocess(img_h, img_w, convert_to_gray),
                                    convert_to_gray=convert_to_gray)

    test_data_loader = config.init_obj('test_loader', torch.utils.data.dataloader,
                                        dataset=test_dataset,
                                        batch_size=test_batch_size,
                                        num_workers=test_num_workers,
                                        drop_last=False,
                                        shuffle=False)

    logger.info(f'Dataloader instances have finished. Test datasets: {len(test_dataset)} '
                f'Test_batch_size/gpu: {test_batch_size}')

    # build model architecture
    model = config.init_obj('model_arch', CustomModel)
    logger.info(f'Model created, trainable parameters: {model.model_parameters() / 1000000.0} MB.')

    exclude_prefix = ['text_encoder', 'coarse_predictor', 'length_predictor']
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  
            if not any(name.startswith(prefix) for prefix in exclude_prefix):
                total_params += np.prod(param.size())

    logger.info(f'Model parameters without any constraint components: {total_params / 1000000.0} MB.')

    logger.info("Loading checkpoint: {} ...".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    logger.info('Testing start...')

    LabelTransformer = LabelConverter(Path(config['model_arch']['args']['common_kwargs']['alphabet']), max_length=-1, ignore_over=False)
    test_metric_res_dict = _valid(config, model, test_data_loader, device, LabelTransformer)
    val_res = f"Word_acc: {test_metric_res_dict['word_acc']:.6f} " \
              f"Word_acc_case_ins: {test_metric_res_dict['word_acc_case_insensitive']:.6f} " \
              f"Edit_distance_acc: {test_metric_res_dict['edit_distance_acc']:.6f}"
    logger.info(val_res)

    logger.info('Testing end...')


def _valid(config: ConfigParser, model, test_data_loader, device, LabelTransformer):

    model.eval()

    logger = config.get_logger('tester', config['tester']['log_verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, config['tester']['tensorboard'])
    test_metrics = AverageMetricTracker('loss', 'word_acc', 'word_acc_case_insensitive', 'edit_distance_acc', writer=writer)
    test_metrics.reset()

    total_time = total_frame = 0

    count = 0

    for step_idx, input_data_item in enumerate(test_data_loader):

        if step_idx % 1000 == 0:
            logger.info('[Testing stage] Step: [{} / {}]'.format(step_idx, len(test_data_loader)))

        images = input_data_item[0]
        text_label = input_data_item[1]

        with torch.no_grad():
            images = images.to(device)

            if hasattr(model, 'module'):
                model = model.module
            else:
                model = model

            start = time.time()

            fine_logits, coarse_logits, length_logits, seq_decoder_features = predict(model, images, max_length=120)

            total_time += time.time() - start
            total_frame += images.size(0)

            correct = 0
            correct_case_ins = 0
            total_distance_ref = 0
            total_edit_distance = 0

            recognition_results = ''

            for index, (pred, text_gold) in enumerate(zip(fine_logits, text_label)):
                predict_text = ""
                for i in range(len(pred)):  # decode one sample
                    if pred[i] == LabelTransformer.EOS:
                        break

                    decoded_char = LabelTransformer.decode(pred[i])
                    predict_text += decoded_char

                predict_text = zhconv.convert(strQ2B(predict_text.lower().replace(' ', '')), 'zh-cn')
                text_gold = zhconv.convert(strQ2B(text_gold.lower().replace(' ', '')), 'zh-cn')

                recognition_results = f'ID: {step_idx} \t GT: {text_gold} \t PT: {predict_text} \t STATE: {predict_text == text_gold} \n'

                ref = max(len(text_gold), len(predict_text))      # The total_distance_ref computation of FuDan formula
                edit_distance = distance.levenshtein(text_gold, predict_text)
                total_distance_ref += ref
                total_edit_distance += edit_distance

                # calculate word accuracy
                if predict_text == text_gold:
                    correct += 1
                if predict_text.lower() == text_gold.lower():
                    correct_case_ins += 1
            batch_total = images.shape[0]
            # calculate accuracy directly, due to non-distributed
            word_acc = correct / batch_total
            word_acc_case_ins = correct_case_ins / batch_total
            edit_distance_acc = 1 - total_edit_distance / total_distance_ref

            logger.info(recognition_results)

        # update test metric and write to tensorboard,
        test_metrics.update('word_acc', word_acc, batch_total)
        test_metrics.update('word_acc_case_insensitive', word_acc_case_ins, batch_total)
        test_metrics.update('edit_distance_acc', edit_distance_acc, total_distance_ref)

    test_metric_res_dict = test_metrics.result()

    print("\n This test using : {:.5f} ms per images \n".format(total_time / total_frame * 1000))
    print("fps : {}".format(total_frame / total_time))

    print(count)

    return test_metric_res_dict


def parse_args():
    global config
    args = argparse.ArgumentParser(description='Evaluation the performance of NASTR')

    args.add_argument('-c', '--config', # default='path/to/config_file_TEST.json',
                      type=str, help='config file path (default: None)')

    args.add_argument('-r', '--resume',
                      default=None,
                      type=str, help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to be available (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags default type target help')
    options = [
        CustomArgs(['-dist', '--distributed'], default='false', type=str, target='distributed',
                   help='run distributed training, true or false, (default: true).'
                        ' turn off distributed mode can debug code on one gpu/cpu'),
        CustomArgs(['--local_world_size'], default=1, type=int, target='local_world_size',
                   help='the number of processes running on each node, this is passed in explicitly '
                        'and is typically either $1$ or the number of GPUs per node. (default: 1)'),
        CustomArgs(['--local_rank'], default=0, type=int, target='local_rank',
                   help='this is automatically passed in via torch.distributed.launch.py, '
                        'process will be assigned a local rank ID in [0,local_world_size-1]. (default: 0)'),
        CustomArgs(['--finetune'], default='false', type=str, target='finetune',
                   help='finetune mode will load resume checkpoint, but do not use previous config and optimizer '
                        '(default: false), so there has three running mode: normal, resume, finetune')
    ]

    config = ConfigParser.from_args(args, options)
    return config


if __name__ == '__main__':
    config = parse_args()
    logger = config.get_logger('test')
    main(config, logger)
