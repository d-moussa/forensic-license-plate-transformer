import os
import sys
import pickle
import numpy as np
sys.path.append('src/models/seq2seq_transformer/')
import torch
from torch.utils.data import Dataset
from src.models.seq2seq_transformer.transformer_helpers import translate_to_codes
from src.data_loading.augmentations import JpegCompression
import matplotlib.pyplot as plt

class SyntheticDataGenerator(Dataset):

    def __init__(self, data_dir, parameters, mode='train', transform=None, target_transform=None):

        # init
        self.parameters = parameters
        self.transforms = transform

        # generate training/validation and test data
        self.data_dir = data_dir
        self.data_batches_names = os.listdir(data_dir)
        self.num_batches = len(self.data_batches_names)  # num. of pickled data chunks


        # get correct generator
        assert mode in ['train', 'valid', 'test'], print("mode hast to be one of ['train', 'valid', 'test'], given: {}".format(mode))
        if mode == 'train':
            self.dataset = self._train_generator()
        elif mode == 'valid':
            self.dataset = self._valid_generator()
        else:
            self.dataset = self._test_generator()

    def __getitem__(self, item):

        x,y = next(self.dataset)
        k = np.inf  # knowledge potentially returned by transforms

        # apply data transforms
        if self.transforms is not None:

            for transform in self.transforms:
                if isinstance(transform, JpegCompression):
                    k, x = transform(x)
                else:
                    x = transform(x)
                # plt.imshow(x)
                # plt.show()

        # plt.imshow(x, cmap='gray')
        # plt.show()
        # plt.imsave('/home/moussa/inf4former/saved_models/jpeg1-res35.png', x)
        x = torch.from_numpy(x)

        y_num = translate_to_codes(caption=y, parameters=self.parameters)
        y_num = torch.from_numpy(y_num)

        return k, x, y_num

    def __len__(self):
        return self.parameters.items_per_set * self.num_batches

    def _train_generator(self):

        for data_file in range(1, self.num_batches + 1):
            data_dict = self.unpack_data(data_file, data_dir=self.data_dir, status='train')

            data = data_dict['data']
            labels = data_dict['labels']

            num_samples = data.shape[0]

            for sample_idx in range(num_samples):
                yield data[sample_idx], labels[sample_idx][1:],

    def _valid_generator(self):
        for data_file in range(1, self.num_batches + 1):
            data_dict = self.unpack_data(data_file, data_dir=self.data_dir, status='valid')

            data = data_dict['data']
            labels = data_dict['labels']

            num_samples = data.shape[0]

            for sample_idx in range(num_samples):
                yield data[sample_idx], labels[sample_idx][1:],

    def _test_generator(self):
        for data_file in range(1, self.num_batches + 1):
            data_dict = self.unpack_data(data_file, data_dir=self.data_dir, status='test')

            data = data_dict['data']
            labels = data_dict['labels']

            num_samples = data.shape[0]

            for sample_idx in range(num_samples):
                yield data[sample_idx], labels[sample_idx][1:],

    def unpack_data(self, package_num, data_dir, status='train'):
        # check if status is valid
        assert (status in ['train', 'valid', 'test'])

        batch_name = "train_batch_"
        if status == 'test':
            batch_name = "test_batch_"
        elif status == 'valid':
            batch_name = "valid_batch_"

        file_name = os.path.join(data_dir, batch_name + str(package_num))

        # check if file exists
        if not os.path.exists(file_name):
            return None

        data_dict = None
        with open(file_name, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')

        return data_dict



