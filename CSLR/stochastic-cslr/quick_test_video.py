from tqdm import tqdm
import numpy as np
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from krsl_datasets import PhoenixVideoTextDataset, PhoenixEvaluator
import Levenshtein as Lev
import stochastic_cslr
import pandas as pd
from decord import VideoReader
from decord import cpu, gpu
from pathlib import Path
from PIL import Image
from torchvision import transforms
from krsl_datasets.utils import LookupTable
import os
import glob

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default="data/phoenix-2014-multisigner")
parser.add_argument("--split", default="val")
parser.add_argument("--model", choices=["dfl", "sfl"], default="sfl")
parser.add_argument("--epoch", default=100)
parser.add_argument("--device", default="cpu")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--nj", type=int, default=4)
parser.add_argument("--beam-width", type=int, default=10)
parser.add_argument("--prune", type=float, default=0.01)
parser.add_argument("--use-lm", action="store_true", default=False)
args = parser.parse_args()

base_size = [256, 256]
crop_size = [224, 224]
random_crop = False
p_drop = 0.5
random_drop = False

def sample_indices(n, p_drop, random_drop):
    p_kept = 1 - p_drop

    if random_drop:
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

#def get_frames(video_path):
    # frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
#    vr = VideoReader(video_path, ctx=cpu(0))

#    return vr

def get_frames():
    frames = glob.glob("video1/*")
    return sorted(frames)

def load_data_frame(split):
    """Load corpus."""
    path = f"{split}.csv"
    df = pd.read_csv(path, sep="|")
    df["annotation"] = df["annotation"].apply(str.split)
    df = df.sort_values("id")

    return df

def create_vocab():
    df = load_data_frame("train_ger")
    sentences = df["annotation"].to_list()
    return LookupTable(
        [gloss for sentence in sentences for gloss in sentence],
        allow_unk=True,
    )

vocab = create_vocab()

transform = transforms.Compose(
[
    transforms.Resize(base_size),
    transforms.RandomCrop(crop_size)
    if random_crop
    else transforms.CenterCrop(crop_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]
)

video_path = "video.avi"
#frames = get_frames(video_path="video.avi")
frames = get_frames()
indices = sample_indices(n=len(frames), p_drop=p_drop, random_drop=random_drop)
frames = [frames[i] for i in indices]
frames = map(Image.open, frames)
frames = map(transform, frames)
frames = np.stack(list(frames))

#indices = sample_indices(n=len(frames), p_drop=p_drop, random_drop=random_drop)
#frames = [Image.fromarray(frames[i].asnumpy(), 'RGB') for i in indices]
#frames = map(transform, frames)
#frames = np.stack(list(frames))
model = stochastic_cslr.load_model(args.model == "sfl", epoch=args.epoch, lang="German")
model.to(args.device)
model.eval()

lpis = model([torch.tensor(frames).to(args.device)])
prob = []

prob += [lpi.exp().detach().cpu().numpy() for lpi in lpis]
hyp = model.decode(prob=prob, beam_width=args.beam_width, prune=args.prune, nj=args.nj)

def sup(preds):
    res = []
    for i in range(len(preds)):
        if preds[i] == 0 or (i > 0 and preds[i] == preds[i - 1]):
            continue

        res.append(preds[i])
    return res
print(hyp)
hyp = [sup(h) for h in hyp]
hyp2 = [" ".join([vocab[i] for i in hi]) for hi in hyp]
print(hyp2[0])
#print(os.path.join(video_path.split("/")[-2], video_path.split("/")[-1]))
#ground_truth = list(df[df["video"] == os.path.join(video_path.split("/")[-2], video_path.split("/")[-1])]["annotation"])
#print(ground_truth)
