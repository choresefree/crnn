import re
import warnings
import PIL
import lmdb
import pandas as pd
import six
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import transforms
from utils import str2num


class ImageConcatDataset(ConcatDataset):
    """ Splice datasets of the same type. """
    def __getattr__(self, k):
        return getattr(self.datasets[0], k)


# coding: utf-8
class SimpleDataset(Dataset):
    """ A very simple way to load text image data.
    Args:
        csv_dir (str): Path of CSV file which save the image paths, labels and their corresponding.
        alphabet (str): A dictionary that associates numeric labels with text labels.
    """
    def __init__(self, csv_dir, alphabet='abc'):
        data = pd.read_csv(csv_dir, header=None, low_memory=False).values.tolist()[1:]
        self.data = data
        self.alphabet = alphabet

    def __getitem__(self, index):
        img_dir, text = self.data[index]
        text = re.sub('[^0-9a-zA-Z]+', '', text)
        if len(self.alphabet) == 37:
            text = text.lower()
        image = Image.open(img_dir).convert('RGB')
        label = str2num(self.alphabet, text)[0]
        label_len = len(label)
        return image, label, label_len

    def __len__(self):
        return len(self.data)


# coding: utf-8
class LmdbDataset(Dataset):
    """ Dataset for loading datasets in LMDB format.
    Import from https://github.com/ayumiymk/aster.pytorch. Websites for downloading training set and
    test set in LMDB format can be found from https://github.com/FangShancheng/ABINet.
    Args:
        lmdb_dir (str): Path of LMDB file.
        alphabet (str): A dictionary that associates numeric labels with text labels.
    """
    def __init__(self, lmdb_dir, alphabet='abc'):
        print(f'load dataset from {lmdb_dir}')
        self.env = lmdb.open(lmdb_dir, readonly=True, lock=False, readahead=False, meminit=False)
        self.img_texts = []
        self.alphabet = alphabet
        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('num-samples'.encode()))

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            image_key, label_key = f'image-{index + 1:09d}', f'label-{index + 1:09d}'
            text = str(txn.get(label_key.encode()), 'utf-8')
            text = re.sub('[^0-9a-zA-Z]+', '', text)
            if len(self.alphabet) == 37:
                text = text.lower()
            img_buf = txn.get(image_key.encode())
            buf = six.BytesIO()
            buf.write(img_buf)
            buf.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # EXIF warning from TiffPlugin
                image = PIL.Image.open(buf).convert('RGB')
        label = str2num(self.alphabet, text)
        label_len = len(label)

        return image, label, text, label_len

    def __len__(self):
        return self.length


class ResizeNormalize(object):
    """ Convert RGB images to the input format of the network.
    RGB image is adopted by default and use 0.5 as sequence of means and standard deviations for each channel.
    Args:
        size (tuple): Image size input to the network.
    """
    def __init__(self, size):
        self.img_tfs = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, ], [0.5, ])
        ])

    def __call__(self, img):
        img = self.img_tfs(img)
        return img


class AlignCollate(object):
    """ Rewriting function during data loading.
    Args:
        img_height (int): Input image height.
        img_width (int): Input image weight.
    """
    def __init__(self, img_height=32, img_width=100):
        self.img_height = img_height
        self.img_width = img_width

    def __call__(self, batch):
        images, labels, b_texts, lengths = zip(*batch)
        labels = [i for j in labels for i in j]
        transform = ResizeNormalize((self.img_height, self.img_width))
        images = [transform(image) for image in images]
        b_images = torch.stack(images)
        b_labels = torch.IntTensor(labels)
        b_label_lens = torch.IntTensor(lengths)

        return b_images, b_labels, b_texts, b_label_lens
