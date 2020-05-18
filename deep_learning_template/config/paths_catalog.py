import os


class DatasetCatalog(object):
    default_data_dir = os.path.expanduser('~/Datasets')
    DATA_DIR = os.environ.get('DATASET_HOME', default_data_dir)
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
        "CIFAR10_TRAIN": {
            "root": "cifar10",
            "train": True,
        },
        "CIFAR10_TEST": {
            "root": "cifar10",
            "train": False,
        },
        "KITTI_ROI_Z_TRAIN": {
            "root": "/home/linghao/Datasets/kitti/object/training/roi_z",
            "split": 'train'
        },
        "KITTI_ROI_Z_VAL": {
            "root": "/home/linghao/Datasets/kitti/object/training/roi_z",
            "split": 'val'
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
        if name in ['CIFAR10_TRAIN', 'CIFAR10_TEST']:
            attrs = DatasetCatalog.DATASETS[name]
            root = os.path.join(DatasetCatalog.DATA_DIR, attrs['root'])
            attrs['root'] = root
            return dict(
                factory='CIFAR10',
                args=attrs
            )
        if name in ['KITTI_ROI_Z_TRAIN', 'KITTI_ROI_Z_VAL']:
            attrs = DatasetCatalog.DATASETS[name]
            root = os.path.join(DatasetCatalog.DATA_DIR, attrs['root'])
            attrs['root'] = root
            return dict(
                factory='ROI_Z_DS',
                args=attrs
            )
        raise RuntimeError("Dataset not available: {}".format(name))
