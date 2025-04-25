
import random
import re
from re import L
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import matplotlib.pyplot as plt
import numpy as np
import zhconv
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import lmdb                                     
import six
import sys
import math
import torch
import logging
from PIL import Image
from tqdm import tqdm

from collections import Counter


def set_random_seed(seed):
    if seed is not None:
        # ---- set random seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        logging.warning('You have chosen to seed training. '
                        'This will slow down your training!')


class lmdbDataset(Dataset):

    def __init__(self, root=None):
        super(lmdbDataset, self).__init__()
        self.env = lmdb.open(path=root,            
                             max_readers=16,       
                             readonly=True,       
                             lock=False,           
                             readahead=False,     
                             meminit=False)        

        if not self.env:
            print('cannot create lmdb from {}'.format(root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())
            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))

            try:
                img = Image.open(buf).convert('RGB')

                h, w = img.size[1], img.size[0]
                if h < 1 or w < 1:
                    print(label)

                img = self.transform(img)
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]
        return img, label


if __name__ == '__main__':
    set_random_seed(seed=42)

    dataset_path = r'path/to/dataset'

    dataset = lmdbDataset(root=dataset_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_labels = []
    for i, batch in enumerate(tqdm(dataloader)):
        img, label = batch

        print(label)