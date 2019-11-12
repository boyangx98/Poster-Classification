"""
EECS 445 - Introduction to Machine Learning
Fall 2019 - Project 2
Posters Dataset
    Class wrapper for interfacing with the dataset of movie poster images
"""
import os
import numpy as np
import pandas as pd
import torch 
from matplotlib import pyplot as plt
from imageio import imread
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utils import config

def get_train_val_test_loaders(num_classes):
    tr, va, te, _ = get_train_val_dataset(num_classes=num_classes)

    batch_size = config('cnn.batch_size')
    tr_loader = DataLoader(tr, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va, batch_size=batch_size, shuffle=False)
    te_loader = DataLoader(te, batch_size=batch_size, shuffle=False)

    return tr_loader, va_loader, te_loader, tr.get_semantic_label

def get_train_val_dataset(num_classes=10):
    tr = PostersDataset('train', num_classes)
    va = PostersDataset('val', num_classes)
    te = PostersDataset('test', num_classes)

    # Resize
    tr.X = resize(tr.X)
    va.X = resize(va.X)
    te.X = resize(te.X)

    # Standardize
    standardizer = ImageStandardizer()
    standardizer.fit(tr.X)
    tr.X = standardizer.transform(tr.X)
    va.X = standardizer.transform(va.X)
    te.X = standardizer.transform(te.X)

    # Transpose the dimensions from (N,H,W,C) to (N,C,H,W)
    tr.X = tr.X.transpose(0,3,1,2)
    va.X = va.X.transpose(0,3,1,2)
    te.X = te.X.transpose(0,3,1,2)

    return tr, va, te, standardizer

def resize(X):
    """
    Resizes the data partition X to the size specified in the config file.
    Uses bicubic interpolation for resizing.

    Returns:
        the resized images as a numpy array.
    """
    # TODO: Complete this function
    resized = []
    image_dim = config('image_dim')
    N = X.shape[0]
    #num_channel = X.shape[3]
    for img in X:
        new_img = np.array(Image.fromarray(img).resize((image_dim, image_dim), Image.BICUBIC))
        resized.append(new_img)
    return resized

class ImageStandardizer(object):
    """
    Channel-wise standardization for batch of images to mean 0 and variance 1.
    The mean and standard deviation parameters are computed in `fit(X)` and
    applied using `transform(X)`.

    X has shape (N, image_height, image_width, color_channel)
    """
    def __init__(self):
        super().__init__()
        self.image_mean = None
        self.image_std = None

    def fit(self, X):
        # TODO: Complete this function
        self.image_mean = np.mean(X, axis = (0,1,2))
        self.image_std = np.std(X, axis = (0,1,2))
        print(self.image_mean, self.image_std)
        #

    def transform(self, X):
        # TODO: Complete this function
        N = np.shape(X)[0]
        image_height = np.shape(X)[1]
        image_width = np.shape(X)[2]
        num_channel = np.shape(X)[3]
        trans_X = np.zeros([N, image_height, image_width, num_channel])
        for img_num in range(N):
            for i in range(image_height):
                for j in range(image_width):
                    for ch_num in range(num_channel):
                        trans_X[img_num][i][j][ch_num] = X[img_num][i][j][ch_num] - self.image_mean[ch_num]
                        trans_X[img_num][i][j][ch_num] /= self.image_std[ch_num]
        return trans_X
        #

class PostersDataset(Dataset):

    def __init__(self, partition, num_classes=10):
        """
        Reads in the necessary data from disk.
        """
        super().__init__()

        if partition not in ['train', 'val', 'test']:
            raise ValueError('Partition {} does not exist'.format(partition))

        np.random.seed(0)
        self.partition = partition
        self.num_classes = num_classes

        # Load in all the data we need from disk
        self.metadata = pd.read_csv(config('csv_file'))
        self.X, self.y = self._load_data()

        self.semantic_labels = dict(zip(
            self.metadata['numeric_label'],
            self.metadata['semantic_label']
        ))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]).float(), torch.tensor(self.y[idx]).long()

    def _load_data(self):
        """
        Loads a single data partition from file.
        """
        print("loading %s..." % self.partition)

        if self.partition == 'test':
            if self.num_classes == 5:
                df = self.metadata[self.metadata.partition == self.partition]
            elif self.num_classes == 10:
                df = self.metadata[self.metadata.partition.isin([self.partition, ' '])]
            else:
                raise ValueError('Unsupported test partition: num_classes must be 5 or 10')
        else:
            df = self.metadata[
                (self.metadata.numeric_label < self.num_classes) &
                (self.metadata.partition == self.partition)
            ]

        X, y = [], []
        for i, row in df.iterrows():
            label = row['numeric_label']
            image = imread(os.path.join(config('image_path'), row['filename']), pilmode = 'RGB')
            X.append(image)
            y.append(row['numeric_label'])

        return np.array(X), np.array(y)

    def get_semantic_label(self, numeric_label):
        """
        Returns the string representation of the numeric class label (e.g.,
        the numberic label 1 maps to the semantic label 'miniature_poodle').
        """
        return self.semantic_labels[numeric_label]

if __name__ == '__main__':
    ## Future note: check scipy imread and imresize
    import warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    np.set_printoptions(precision=3)
    tr, va, te, standardizer = get_train_val_dataset()
    print("Train:\t", len(tr.X))
    print("Val:\t", len(va.X))
    print("Test:\t", len(te.X))
    print("Mean:", standardizer.image_mean)
    print("Std: ", standardizer.image_std)
