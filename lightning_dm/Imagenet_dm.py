import os
import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from lightning import LightningDataModule
import presets
import utils
from sampler import RASampler


def _get_cache_path(filepath):
    import hashlib

    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path
class IMNETDataModule(LightningDataModule):
    def __init__(self, traindir, valdir, args):
        super().__init__()
        self.traindir = traindir
        self.valdir = valdir
        self.args = args

    def setup(self, stage=None):
        val_resize_size, val_crop_size, train_crop_size = (
            self.args.val_resize_size,
            self.args.val_crop_size,
            self.args.train_crop_size,
        )
        interpolation = InterpolationMode(self.args.interpolation)

        # Load training data
        print("Loading training data")
        self.train_dataset = self._load_dataset(
            self.traindir, train_crop_size, interpolation, train=True
        )

        # Load validation data
        print("Loading validation data")
        self.val_dataset = self._load_dataset(
            self.valdir, val_crop_size, interpolation, train=False
        )

    def _load_dataset(self, directory, crop_size, interpolation, train=True):
        cache_path = _get_cache_path(directory)
        if self.args.cache_dataset and os.path.exists(cache_path):
            print(f"Loading dataset from {cache_path}")
            dataset, _ = torch.load(cache_path, weights_only=False)
        else:
            if train:
                dataset = torchvision.datasets.ImageFolder(
                    directory,
                    presets.ClassificationPresetTrain(
                        crop_size=crop_size,
                        interpolation=interpolation,
                        auto_augment_policy=getattr(self.args, "auto_augment", None),
                        random_erase_prob=getattr(self.args, "random_erase", 0.0),
                        ra_magnitude=getattr(self.args, "ra_magnitude", None),
                        augmix_severity=getattr(self.args, "augmix_severity", None),
                        backend=self.args.backend,
                        use_v2=self.args.use_v2,
                    ),
                )
            else:
                preprocessing = presets.ClassificationPresetEval(
                    crop_size=crop_size,
                    resize_size=self.args.val_resize_size,
                    interpolation=interpolation,
                    backend=self.args.backend,
                    use_v2=self.args.use_v2,
                )
                dataset = torchvision.datasets.ImageFolder(
                    directory,
                    preprocessing,
                )

            if self.args.cache_dataset:
                print(f"Saving dataset to {cache_path}")
                utils.mkdir(os.path.dirname(cache_path))
                utils.save_on_master((dataset, directory), cache_path)

        return dataset

    def train_dataloader(self):
        if hasattr(self.args, "ra_sampler") and self.args.ra_sampler:
            train_sampler = RASampler(self.train_dataset, shuffle=True, repetitions=self.args.ra_reps)
        else:
            train_sampler = torch.utils.data.RandomSampler(self.train_dataset)

        return DataLoader(
            self.train_dataset,
            sampler=train_sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        test_sampler = torch.utils.data.SequentialSampler(self.val_dataset)
        return DataLoader(
            self.val_dataset,
            sampler=test_sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
        )
