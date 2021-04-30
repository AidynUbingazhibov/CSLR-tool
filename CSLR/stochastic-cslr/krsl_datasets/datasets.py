import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from functools import partial
from PIL import Image
from torchvision import transforms
from collections import defaultdict

class defaultdict_with_warning(defaultdict):
    warned = set()
    warning_enabled = False

    def __getitem__(self, key):
        if key == "text" and key not in self.warned and self.warning_enabled:
            print(
                'Warning: using batch["text"] to obtain label is deprecated, '
                'please use batch["label"] instead.'
            )
            self.warned.add(key)

        return super().__getitem__(key)


class VideoTextDataset(Dataset):
    Corpus = None

    def __init__(
        self,
        root,
        split,
        p_drop=0,
        random_drop=True,
        random_crop=True,
        base_size=[256, 256],
        crop_size=[224, 224],
        vocab=None,
    ):
        assert 0 <= p_drop <= 1, f"p_drop value {p_drop} is out of range."
        assert (
            self.Corpus is not None
        ), f"Corpus is not defined in the derived class {self.__class__.__name__}."

        self.corpus = self.Corpus(root)
        self.random_drop = random_drop
        self.p_drop = p_drop

        self.data_frame = self.corpus.load_data_frame(split)
        self.vocab = vocab or self.corpus.create_vocab()
        self.transform = transforms.Compose(
            [
                transforms.Resize(base_size),
                transforms.RandomCrop(crop_size)
                if random_crop
                else transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def sample_indices(self, n):
        p_kept = 1 - self.p_drop
        if self.random_drop:
            indices = np.arange(n)
            np.random.shuffle(indices)
            indices = indices[: int(n * p_kept)]
            indices = sorted(indices)
        else:
            indices = np.arange(0, n, 1 / p_kept)
            indices = np.round(indices)
            indices = np.clip(indices, 0, n - 1)
            indices = indices.astype(int)
        return indices

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, index):
        sample = {**self.data_frame.iloc[index].to_dict()}  # copy
        frames = self.corpus.get_frames(sample)

        indices = self.sample_indices(len(frames))
        frames = [Image.fromarray(frames[i].asnumpy(), 'RGB') for i in indices]
        frames = map(self.transform, frames)
        frames = np.stack(list(frames))

        label = list(map(self.vocab, sample["annotation"]))

        sample.update(
            video=frames,
            label=label,
        )
        return sample

    @staticmethod
    def collate_fn(batch):
        collated = defaultdict_with_warning(list)

        for sample in batch:
            collated["video"].append(torch.tensor(sample["video"]).float())
            collated["label"].append(torch.tensor(sample["label"]).long())
            collated["annotation"].append(sample["annotation"])

        collated.warning_enabled = True

        return dict(collated)
