import os


class DatasetCatalog(object):
    DATA_DIR = os.path.join(os.environ['HOME'], 'Datasets')
    DATASETS = {
        "MNIST_TRAIN": {
            "root": "mnist",
            "train": True,
            "download": True
        },
        "MNIST_TEST": {
            "root": "mnist",
            "train": False,
            "download": True
        },

    }

    @staticmethod
    def get(name):
        if name in ['MNIST_TRAIN', "MNIST_TEST"]:
            attrs = DatasetCatalog.DATASETS[name]
            root = os.path.join(DatasetCatalog.DATA_DIR, attrs['root'])
            attrs['root'] = root
            return dict(
                factory='MNIST',
                args=attrs
            )
        raise RuntimeError("Dataset not available: {}".format(name))
