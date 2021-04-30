from tqdm import tqdm
import numpy as np
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from krsl_datasets import PhoenixVideoTextDataset, PhoenixEvaluator
import Levenshtein as Lev
import stochastic_cslr

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", default="data/phoenix-2014-multisigner")
parser.add_argument("--split", default="val")
parser.add_argument("--model", choices=["dfl", "sfl"], default="sfl")
parser.add_argument("--epoch", default=10)
parser.add_argument("--device", default="cuda")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--nj", type=int, default=4)
parser.add_argument("--beam-width", type=int, default=10)
parser.add_argument("--prune", type=float, default=0.01)
parser.add_argument("--use-lm", action="store_true", default=False)
args = parser.parse_args()

dataset = PhoenixVideoTextDataset(
    root=args.data_root,
    split=args.split,
    p_drop=0.5,
    random_drop=False,
    random_crop=False,
    crop_size=[224, 224],
    base_size=[256, 256],
)

data_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    shuffle=False,  # result should be strictly ordered for evaluation.
    num_workers=args.nj,
    collate_fn=dataset.collate_fn,
)

model = stochastic_cslr.load_model(args.model == "sfl", epoch=args.epoch)
model.to(args.device)
model.eval()

prob = []
ref = []
j = 0

with torch.no_grad():
    for batch in tqdm(data_loader):
        ref += batch["label"]
        video = list(map(lambda v: v.to(args.device), batch["video"]))
        prob += [lpi.exp().cpu().numpy() for lpi in model(video)]

hyp = model.decode(prob=prob, beam_width=args.beam_width, prune=args.prune, nj=args.nj)

def sup(preds):
    res = []
    for i in range(len(preds)):
        if preds[i] == 0 or (i > 0 and preds[i] == preds[i - 1]):
            continue

        res.append(preds[i])
    return res

hyp = [sup(h) for h in hyp]

hypes = []
gts = []

for sent in hyp:
    hypes += sent

for gt in ref:
    gts += gt

hyp2 = [" ".join([dataset.vocab[i] for i in hi]) for hi in hyp]
#hyp = [" ".join([chr(i) for i in hi]) for hi in hyp]
#ref = [" ".join([chr(i) for i in re]) for re in ref]

hyp = "".join([chr(x) for x in hypes])
ref = "".join([chr(x) for x in gts])

print(Lev.distance(hyp, ref) / len(ref) * 100)
