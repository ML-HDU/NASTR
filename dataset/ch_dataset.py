# including:
    # function: hierarchy_dataset
    # class: LmdbDataset
    # class: LabelConverter

import os
import io
import cv2
import math
import collections
collections.Iterable = collections.abc.Iterable
from PIL import Image
import numpy as np
import lmdb
from pathlib import Path

import torch
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler
import torch.distributed as dist


def hierarchy_dataset(root, select_data=None, img_w=256, img_h=32, transform=None,
                      target_transform=None,
                      case_sensitive=False, convert_to_gray=True, max_length=120):
    """
    combine lmdb data in sub directory (MJ_train vs Synthtext)
    change sub directory if you want in config file by format "sub_dir-sub_dir-..."
    """
    dataset_list = []
    if select_data is not None:
        select_data = select_data.split('-')
        for select_d in select_data:
            dataset = LmdbDataset(lmdb_dir_root=os.path.join(root, select_d), img_w=img_w,
                                  img_h=img_h, transform=transform, target_transform=target_transform,
                                  case_sensitive=case_sensitive, convert_to_gray=convert_to_gray, max_length=max_length)
            dataset_list.append(dataset)
    concatenated_dataset = ConcatDataset(dataset_list)
    return concatenated_dataset


class LmdbDataset(Dataset):
    """
    load lmdb dataset in deep text recognition benchmark [https://github.com/clovaai/deep-text-recognition-benchmark]
    Download link: [https://www.dropbox.com/sh/i39abvnefllx2si/AABX4yjNn2iLeKZh1OAwJUffa/data_lmdb_release.zip?dl=0]
    unzip and modify folder by:
    dataset/data_lmdb_release/
        training/
            MJ_train
            ST
        val/
            MJ_test
            MJ_valid
    """

    def __init__(self, lmdb_dir_root=None, transform=None, target_transform=None, img_w=256,
                 img_h=32, case_sensitive=False,
                 convert_to_gray=True, max_length=120):
        self.env = lmdb.open(lmdb_dir_root,
                             max_readers=32,
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        if not self.env:
            raise RuntimeError('Lmdb file cannot be open')

        self.transform = transform
        self.target_transform = target_transform

        self.case_sensitive = case_sensitive
        self.convert_to_gray = convert_to_gray
        self.img_w = img_w
        self.img_h = img_h
        self.max_length = max_length

        self.image_keys, self.labels = self.__get_images_and_labels()
        self.nSamples = len(self.image_keys)

    def __len__(self):
        return self.nSamples

    def __get_images_and_labels(self):
        image_keys = []
        labels = []
        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get(b"num-samples").decode())
            for i in range(nSamples):
                index = i + 1
                image_key = ('image-%09d' % index).encode()
                label_key = ('label-%09d' % index).encode()

                label = txn.get(label_key).decode()

                if len(label) > self.max_length != -1:
                    continue

                # if not 20 < len(label) <= 25:
                # if not 25 < len(label):
                #     continue

                image_keys.append(image_key)
                labels.append(label)
        return image_keys, labels

    def __getitem__(self, index):
        try:
            image_key = self.image_keys[index]

            with self.env.begin(write=False) as txn:
                imgbuf = txn.get(image_key)
                buf = io.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)

                try:
                    if self.convert_to_gray:
                        img = Image.open(buf).convert('L')
                    else:
                        img = Image.open(buf).convert('RGB')
                except IOError:
                    print('Error Image for {}'.format(image_key))

                # rotate vertical images by measure the aspect ratio
                width, height = img.size
                if height > width:
                    img = img.transpose(Image.ROTATE_90)

                if self.transform is not None:
                    img, width_ratio = self.transform(img)

                label = self.labels[index]

                if self.target_transform is not None:
                    label = self.target_transform(label)

            if not self.case_sensitive:
                label = label.lower()
            return img, label

        except Exception as read_e:
            return self.__getitem__(np.random.randint(self.__len__()))


class LabelConverter:
    def __init__(self, classes, max_length=-1, masked_size=1, ignore_over=False):
        """

        :param classes: alphabet(keys), key string or text vocabulary
        :param max_length:  max_length is mainly for controlling the statistics' stability,
         due to the layer normalisation. and the model can only predict max_length text.
         -1 is for fixing the length, the max_length will be computed dynamically for one batch.
         Empirically, it should be maximum length of text you want to predict.
        :param ignore_over:  (bool, default=False): whether to ignore over max length text.
        """

        cls_list = None
        if isinstance(classes, str):
            cls_list = list(classes)
        if isinstance(classes, Path):
            p = Path(classes)
            if not p.exists():
                raise RuntimeError('Key file is not found')
            with p.open(encoding='utf8') as f:
                classes = f.read()
                classes = classes.strip()
                cls_list = list(classes)
        elif isinstance(classes, list):
            cls_list = classes

        self.alphabet = cls_list
        self.alphabet_mapper = {'<EOS>': 1, '<PAD>': 0, '<SOS>': 2}
        nSymbol = len(self.alphabet_mapper)
        # end of sequence
        for i, item in enumerate(self.alphabet):
            self.alphabet_mapper[item] = i + nSymbol      # Add <EOS>, <PAD>, <SOS>
        self.alphabet_inverse_mapper = {v: k for k, v in self.alphabet_mapper.items()}

        self.EOS = self.alphabet_mapper['<EOS>']
        self.PAD = self.alphabet_mapper['<PAD>']
        self.SOS = self.alphabet_mapper['<SOS>']

        self.nclass = len(self.alphabet) + nSymbol
        self.max_length = max_length
        self.masked_size = masked_size
        self.ignore_over = ignore_over

    def encode(self, text):
        """ convert text to label index, add <EOS>, and do max_len padding
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.LongTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.alphabet_mapper[item] for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]  # encode

            if self.max_length == -1:
                local_max_length = max([len(x) for x in text])  # padding
                self.ignore_over = True
            else:
                local_max_length = self.max_length

            nb = len(text)

            targets = torch.zeros(nb, (local_max_length + 1))
            targets[:, :] = self.PAD

            for i in range(nb):
                if not self.ignore_over:
                    if len(text[i]) > local_max_length:
                        raise RuntimeError('Text is larger than {}: {}'.format(local_max_length, len(text[i])))

                targets[i][:len(text[i])] = text[i]
                targets[i][len(text[i])] = self.EOS

            text = targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def AR_encode(self, text):
        """ convert text to label index, add <EOS>, and do max_len padding
        Args:
            text (str or list of str): texts to convert.
        Returns:
            torch.LongTensor targets:max_length × batch_size
        """
        if isinstance(text, str):
            text = [self.alphabet_mapper[item] for item in text]
        elif isinstance(text, collections.Iterable):
            text = [self.encode(s) for s in text]  # encode

            if self.max_length == -1:
                local_max_length = max([len(x) for x in text])  # padding
                self.ignore_over = True
            else:
                local_max_length = self.max_length

            nb = len(text)

            AR_targets = torch.zeros(nb, (local_max_length) + 2)  # add sos and eos to sequence
            AR_targets[:, :] = self.PAD
            AR_targets[:, 0] = self.SOS     # Take the same symbol for SOS (start of sentence) and SOS

            for i in range(nb):
                if not self.ignore_over:
                    if len(text[i]) > local_max_length:
                        raise RuntimeError('Text is larger than {}: {}'.format(local_max_length, len(text[i])))

                AR_targets[i][1: len(text[i]) + 1] = text[i]
                AR_targets[i][len(text[i]) + 1] = self.EOS

            text = AR_targets.transpose(0, 1).contiguous()
            text = text.long()
        return torch.LongTensor(text)

    def decode(self, t):
        """Decode encoded texts back into strs.
        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        Raises:
            AssertionError: when the texts and its length does not match.
        Returns:
            text (str or list of str): texts to convert.
        """

        if isinstance(t, torch.Tensor):
            texts = self.alphabet_inverse_mapper[t.item()]
        else:
            texts = self.alphabet_inverse_mapper[t]
        return texts

    def get_length(self, text):
        """ get lengths of batch texts by filtering the <EOS> and <PAD> symbols
        :param text: converted text (torch.LongTensor) max_length x batch_size
        :return: lengths (torch.LongTensor) batch_size
        """
        nonzero_index = torch.nonzero(text, as_tuple=True)
        unique_count = torch.unique(nonzero_index[0], return_counts=True)
        lengths = unique_count[1] - 1

        return lengths


class DistCollateFn(object):
    '''
    fix bug when len(data) do not be divided by batch_size, on condition of distributed validation
    avoid error when some gpu assigned zero samples
    '''

    def __init__(self, training=True):
        self.training = training

    def __call__(self, batch):

        batch_size = len(batch)
        if batch_size == 0:
            return dict(batch_size=batch_size, images=None, labels=None)

        if self.training:
            images, labels = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        labels=labels)
        else:
            images, file_names = zip(*batch)
            image_batch_tensor = torch.stack(images, dim=0).float()
            # images Tensor: (bs, c, h, w), file_names tuple: (bs,)
            return dict(batch_size=batch_size,
                        images=image_batch_tensor,
                        file_names=file_names)


class ResizeWeight(object):

    def __init__(self, size, interpolation=Image.BILINEAR, gray_format=True):
        self.w, self.h = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()
        self.gray_format = gray_format

    def __call__(self, img):
        img_w, img_h = img.size

        if self.gray_format:
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:self.h, 0] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[0:self.h, 0:new_w, 0] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 1), dtype=np.uint8)
                resize_img[:, :, 0] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w
        else:  # RGB format
            if img_w / img_h < 1.:
                img = img.resize((self.h, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:self.h, :] = img
                img = resize_img
                width = self.h
            elif img_w / img_h < self.w / self.h:
                ratio = img_h / self.h
                new_w = int(img_w / ratio)
                img = img.resize((new_w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[0:self.h, 0:new_w, :] = img
                img = resize_img
                width = new_w
            else:
                img = img.resize((self.w, self.h), self.interpolation)
                resize_img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                img = np.array(img, dtype=np.uint8)  # (w,h) -> (h,w,c)
                resize_img[:, :, :] = img
                img = resize_img
                width = self.w

            img = self.toTensor(img)
            img.sub_(0.5).div_(0.5)
            return img, width / self.w


class CustomImagePreprocess:
    def __init__(self, _target_height, _target_width, _is_gray):
        self.target_height, self.target_width = _target_height, _target_width
        self.is_gray = _is_gray

    def __call__(self, _img: Image.Image):
        if self.is_gray:
            img = _img.convert('L')
        else:
            img = _img
        img_np = np.asarray(img)
        h, w = img_np.shape[:2]
        resized_img = cv2.resize(img_np, (self.target_width, self.target_height))
        full_channel_img = resized_img[..., None] if len(resized_img.shape) == 2 else resized_img
        resized_img_tensor = torch.from_numpy(np.transpose(full_channel_img, (2, 0, 1))).to(torch.float32)
        resized_img_tensor.sub_(127.5).div_(127.5)
        return resized_img_tensor, w / self.target_width


class DistValSampler(Sampler):
    # DistValSampler distributes batches equally (based on batch size) to every gpu (even if there aren't enough samples)
    # This instance is used as batch_sampler args of validation dtataloader,
    # to guarantee every gpu validate different samples simultaneously
    # WARNING: Some baches will contain an empty array to signify there aren't enough samples
    # distributed=False - same validation happens on every single gpu
    def __init__(self, indices, batch_size, distributed=True):
        self.indices = indices
        self.batch_size = batch_size
        if distributed:
            self.world_size = dist.get_world_size()
            self.global_rank = dist.get_rank()
        else:
            self.global_rank = 0
            self.world_size = 1

        # expected number of batches per process. Need this so each distributed gpu validates on same number of batches.
        # even if there isn't enough data to go around
        self.expected_num_steps = math.ceil(len(self.indices) / self.world_size / self.batch_size)

        # num_samples = total samples / world_size. This is what we distribute to each gpu
        self.num_samples = self.expected_num_steps * self.batch_size

    def __iter__(self):
        current_rank_offset = self.num_samples * self.global_rank
        current_sampled_indices = self.indices[
                                  current_rank_offset:min(current_rank_offset + self.num_samples, len(self.indices))]

        for step in range(self.expected_num_steps):
            step_offset = step * self.batch_size
            # yield sampled_indices[offset:offset + self.batch_size]
            yield current_sampled_indices[step_offset:min(step_offset + self.batch_size, len(current_sampled_indices))]

    def __len__(self):
        return self.expected_num_steps

    def set_epoch(self, epoch):
        return


