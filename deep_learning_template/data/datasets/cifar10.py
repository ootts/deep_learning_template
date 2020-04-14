from torchvision.datasets import cifar


class CIFAR10(cifar.CIFAR10):
    def __init__(self, root, train, transforms, ds_len=-1):
        super().__init__(root, train=train, transform=None, target_transform=None, download=True)
        self.transforms2 = transforms
        self.ds_len = ds_len

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        x, y = self.transforms2(x, y)
        return x, y

    def __len__(self):
        if self.ds_len < 0:
            return super().__len__()
        else:
            return self.ds_len
