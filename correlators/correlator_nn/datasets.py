import torch
from torchvision.transforms import ToTensor, CenterCrop, Compose
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data import SequentialSampler
from torchvision.datasets import MNIST, FashionMNIST


class MNISTDataset:
    '''
    Loads testing and training MNIST datasets and optionally trim
    retention_radius pixels from edge of image to remove some black parts.
    '''

    def __init__(self, directory='data/', retention_radius=28, images_of=7):
        self.directory = directory
        self.retention_radius = retention_radius
        self.retention_transform = Compose([CenterCrop(self.retention_radius),
                                            ToTensor()])
        self.images_of = images_of

    def get_training_data(self, batch_size=64, batches=0, batch_offset=0):
        ''''
        Arguments:
            batch_size (int): Size of batches the dataset is split into.
            batches (int): Default None. Number of batches to split the
                           dataset into.
        Returns:
            trimmed_image (DataLoader object):
                MNIST dataset with the each image array sliced
                to remove retention_radius.
        '''
        # Load MNIST datasets

        # Obtain the training dataset
        self.mnist_train = MNIST(root=self.directory, train=True,
                                 transform=self.retention_transform,
                                 download=True)

        # Filter out the desired training digits
        self.train_digits = [(img, label) for img, label in
                             self.mnist_train if label == self.images_of]

        self.dataset_length = len(self.train_digits)

        # Initialise the dataset parameters
        self.train_batch_size = batch_size

        # Number of training batches. We can set this to 0 to use all data
        # and dynamically find max number of batches
        self.train_batches = batches

        # If we want a batch number greater than 1
        if self.train_batches != 0:
            subset_sampler = SequentialSampler(
                range(batch_offset * self.train_batch_size,
                      self.train_batches * self.train_batch_size)
            )

        else:
            subset_sampler = SequentialSampler(
                range(batch_offset * self.train_batch_size,
                      self.dataset_length)
            )

        train_dataloader = DataLoader(self.train_digits,
                                      batch_size=self.train_batch_size,
                                      shuffle=False, num_workers=3,
                                      sampler=subset_sampler)
        return train_dataloader

    def get_testing_data(self, batch_size=64, batches=None):
        ''''
        Arguments:
            batch_size (int): Size of batches the dataset is split into.
            batches (int): Default None. Number of batches to split the
                           dataset into.

        Returns:
            trimmed_image (DataLoader object):
                MNIST dataset with the each image array sliced
                to remove data outside retention_radius.
        '''
        self.mnist_test = MNIST(root=self.directory, train=False,
                                transform=self.retention_transform,
                                download=True)

        self.test_digits = [(img, label) for img, label in
                            self.mnist_test if label == self.images_of]

        # Number of test batches. We can use this to do a quick run on a
        # subset of the dataset
        self.test_batch_size = batch_size
        self.test_batches = batches

        if self.test_batches is not None:
            subset_sampler = SubsetRandomSampler(range(self.test_batches *
                                                       self.test_batch_size))
            test_dataloader = DataLoader(self.test_digits,
                                         batch_size=self.test_batch_size,
                                         num_workers=3, sampler=subset_sampler)
        else:
            test_dataloader = DataLoader(self.test_digits,
                                         batch_size=self.test_batches,
                                         shuffle=True, num_workers=3)
        return test_dataloader


class FashionMNISTDataset:
    '''
    Loads testing and training MNIST datasets and optionally trim
    retention_radius pixels from edge of image to remove some black parts.


    images_of key = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }

    '''

    def __init__(self, directory='data/', retention_radius=28, images_of=8):
        self.directory = directory
        self.retention_radius = retention_radius
        self.retention_transform = Compose([CenterCrop(self.retention_radius),
                                            ToTensor()])
        self.images_of = images_of

    def get_training_data(self, batch_size=64, batches=0, batch_offset=0):
        ''''
        Arguments:
            batch_size (int): Size of batches the dataset is split into.
            batches (int): Default None. Number of batches to split the
                           dataset into.
        Returns:
            trimmed_image (DataLoader object):
                MNIST dataset with the each image array sliced
                to remove retention_radius.
        '''
        # Load MNIST datasets

        # Obtain the training dataset
        self.fashion_mnist_train = FashionMNIST(root=self.directory, train=True,
                                 transform=self.retention_transform,
                                 download=True)

        # Filter out the desired training digits
        self.train_digits = [(img, label) for img, label in
                             self.fashion_mnist_train if label == self.images_of]

        self.dataset_length = len(self.train_digits)

        # Initialise the dataset parameters
        self.train_batch_size = batch_size

        # Number of training batches. We can set this to 0 to use all data
        # and dynamically find max number of batches
        self.train_batches = batches

        # If we want a batch number greater than 1
        if self.train_batches != 0:
            subset_sampler = SequentialSampler(
                range(batch_offset * self.train_batch_size,
                      self.train_batches * self.train_batch_size)
            )

        else:
            subset_sampler = SequentialSampler(
                range(batch_offset * self.train_batch_size,
                      self.dataset_length)
            )

        train_dataloader = DataLoader(self.train_digits,
                                      batch_size=self.train_batch_size,
                                      shuffle=False, num_workers=3,
                                      sampler=subset_sampler)
        return train_dataloader

    def get_testing_data(self, batch_size=64, batches=None):
        ''''
        Arguments:
            batch_size (int): Size of batches the dataset is split into.
            batches (int): Default None. Number of batches to split the
                           dataset into.

        Returns:
            trimmed_image (DataLoader object):
                MNIST dataset with the each image array sliced
                to remove data outside retention_radius.
        '''
        self.mnist_test = FashionMNIST(root=self.directory, train=False,
                                transform=self.retention_transform,
                                download=True)

        self.test_digits = [(img, label) for img, label in
                            self.mnist_test if label == self.images_of]

        # Number of test batches. We can use this to do a quick run on a
        # subset of the dataset
        self.test_batch_size = batch_size
        self.test_batches = batches

        if self.test_batches is not None:
            subset_sampler = SubsetRandomSampler(range(self.test_batches *
                                                       self.test_batch_size))
            test_dataloader = DataLoader(self.test_digits,
                                         batch_size=self.test_batch_size,
                                         num_workers=3, sampler=subset_sampler)
        else:
            test_dataloader = DataLoader(self.test_digits,
                                         batch_size=self.test_batches,
                                         shuffle=True, num_workers=3)
        return test_dataloader



class MNISTFisherDataset(Dataset):
    '''
    The NNGeometry package is slightly picky about datasets, so create
    a separate class for the data used to calculate the FIM.
    '''
    def __init__(self, mnistdataset, batch_size):
        # large_data = torch.arange(1, 785, dtype=torch.float32).view(1, 784)
        mnist_fisher = MNIST(root=mnistdataset.directory, train=False,
                             transform=mnistdataset.retention_transform,
                             download=True)

        fisher_digits = [(img, label) for img, label in
                         mnist_fisher if label == mnistdataset.images_of]

        # Select batch_size random images (use randperm since w/o replacement)
        selected_indices = torch.randperm(len(fisher_digits))[:batch_size]
        
        # self.data = (torch.squeeze(torch.stack([fisher_digits[i][0].flatten()
        #                                         for i in selected_indices])), )
        self.data = torch.empty(784 * batch_size)
        for i, data_index in enumerate(selected_indices):
            self.data[784*i:784*(i+1)] = fisher_digits[data_index][0].flatten()

        # Deal with NNGeometry's input form expectation
        self.data = (self.data, )

    def __getitem__(self, index):
        return (self.data[index], )

    def __len__(self):
        return len(self.data)


class FashionMNISTFisherDataset(Dataset):
    '''
    The NNGeometry package is slightly picky about datasets, so create
    a separate class for the data used to calculate the FIM.
    '''
    def __init__(self, mnistdataset, batch_size):
        # large_data = torch.arange(1, 785, dtype=torch.float32).view(1, 784)
        fashion_mnist_fisher = FashionMNIST(root=mnistdataset.directory, train=False,
                             transform=mnistdataset.retention_transform,
                             download=True)

        fisher_digits = [(img, label) for img, label in
                         fashion_mnist_fisher if label == mnistdataset.images_of]

        # Select batch_size random images (use randperm since w/o replacement)
        selected_indices = torch.randperm(len(fisher_digits))[:batch_size]
        
        # self.data = (torch.squeeze(torch.stack([fisher_digits[i][0].flatten()
        #                                         for i in selected_indices])), )
        self.data = torch.empty(784 * batch_size)
        for i, data_index in enumerate(selected_indices):
            self.data[784*i:784*(i+1)] = fisher_digits[data_index][0].flatten()

        # Deal with NNGeometry's input form expectation
        self.data = (self.data, )

    def __getitem__(self, index):
        return (self.data[index], )

    def __len__(self):
        return len(self.data)
