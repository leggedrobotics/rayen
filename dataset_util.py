import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
import gzip
import codecs


class MNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.
    This file extends the PyTorch implementation by a three-way data split in
    train, validation and test set and speeds up the getitem() method by
    avoiding to go through PIL.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``,
            ``processed/validation.pt`` and  ``processed/test.pt`` exist.
        partition (str, optional): Data partition, 'train', 'val' or 'test'.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    val_file = 'validation.pt'
    test_file = 'test.pt'

    def __init__(self, root, partition='train', transform=None,
                 target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.partition = partition

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.partition == 'train':
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
            if self.transform is not None:
                self.train_data = self.transform(self.train_data)
            if self.target_transform is not None:
                self.train_labels = self.target_transform(train_labels)
            self.train_data = self.train_data.float()
            self.train_labels = self.train_labels.long()
        elif self.partition == 'val':
            self.val_data, self.val_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.val_file))
            if self.transform is not None:
                self.val_data = self.transform(self.val_data)
            if self.target_transform is not None:
                self.val_labels = self.target_transform(val_labels)
            self.val_data = self.val_data.float()
            self.val_labels = self.val_labels.long()
        elif self.partition == 'test':
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))
            if self.transform is not None:
                self.test_data = self.transform(self.test_data)
            if self.target_transform is not None:
                self.test_labels = self.target_transform(test_labels)
            self.test_data = self.test_data.float()
            self.test_labels = self.test_labels.long()
        else:
            raise ValueError('Partition {} not known.'.format(self.partition))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.partition == 'train':
            img, target = self.train_data[index], self.train_labels[index]
        elif self.partition == 'val':
            img, target = self.val_data[index], self.val_labels[index]
        elif self.partition == 'test':
            img, target = self.test_data[index], self.test_labels[index]
        else:
            raise ValueError('Partition {} not known.'.format(self.partition))
        return img, target

    def __len__(self):
        if self.partition == 'train':
            return len(self.train_data)
        elif self.partition == 'val':
            return len(self.val_data)
        elif self.partition == 'test':
            return len(self.test_data)
        else:
            raise ValueError('Partition {} not known.'.format(self.partition))

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.val_file)) and \
            os.path.exists(
            os.path.join(
                self.root,
                self.processed_folder,
                self.test_file))

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        train_val_set = (
            read_image_file(
                os.path.join(
                    self.root,
                    self.raw_folder,
                    'train-images-idx3-ubyte')),
            read_label_file(
                os.path.join(
                    self.root,
                    self.raw_folder,
                    'train-labels-idx1-ubyte'))
        )
        training_set = (train_val_set[0][:-1000], train_val_set[1][:-1000])
        validation_set = (
            train_val_set[0][-1000:], train_val_set[1][-1000:])
        test_set = (
            read_image_file(
                os.path.join(
                    self.root,
                    self.raw_folder,
                    't10k-images-idx3-ubyte')),
            read_label_file(
                os.path.join(
                    self.root,
                    self.raw_folder,
                    't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.val_file), 'wb') as f:
            torch.save(validation_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.partition
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp,
                                     self.transform.__repr__().replace('\n',
                                                                       '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp,
                                   self.target_transform.__repr__().replace('\n',
                                                                            '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)
