import pandas as pd
from pathlib import Path
from pandarallel import pandarallel
from functools import partial

from .utils import LookupTable
from .language_model import SRILM
import os

from decord import VideoReader
from decord import cpu, gpu

pandarallel.initialize(verbose=0)


class Corpus:
    def __init__(self, root):
        self.root = Path(root)

    def load_data_frame(self, split):
        raise NotImplementedError

    def create_vocab(self):
        df = self.load_data_frame("train")
        sentences = df["annotation"].to_list()
        return LookupTable(
            [gloss for sentence in sentences for gloss in sentence],
            allow_unk=True,
        )


class PhoenixCorpus(Corpus):
    mean = [0.53724027, 0.5272855, 0.51954997]
    std = [1, 1, 1]

    def __init__(self, root):
        super().__init__(root)

    def load_alignment(self):
        dirname = self.root / "annotations" / "automatic"

        # important to literally read NULL instead read it as nan
        read = partial(pd.read_csv, sep=" ", na_filter=False)
        ali = read(dirname / "train.alignment", header=None, names=["id", "classlabel"])
        cls = read(dirname / "trainingClasses.txt")

        df = pd.merge(ali, cls, how="left", on="classlabel")
        del df["classlabel"]

        df["gloss"] = df["signstate"].apply(lambda s: s.rstrip("012"))

        df["id"] = df["id"].parallel_apply(lambda s: "/".join(s.split("/")[3:-2]))
        grouped = df.groupby("id")

        gdf = grouped["gloss"].agg(" ".join)
        sdf = grouped["signstate"].agg(" ".join)

        df = pd.merge(gdf, sdf, "inner", "id")

        assert (
            len(df) == 5671
        ), f"Alignment file is not correct, expect to have 5671 entries but got {len(df)}."

        return df

    def load_data_frame(self, split, aligned_annotation=False):
        """Load corpus."""
        #path = self.root / "annotations" / "manual" / f"{split}.corpus.csv"
        path = f"/home/asandygulova/dock/krsl_173_1708/annotation_phrase_split/{split}.csv"
        df = pd.read_csv(path)
        df["annotation"] = df["annotation"].apply(str.split)

        return df

    def get_frames(self, sample):
        #frames = (self.root / "features" / type / sample["folder"]).glob("*.png")
        video_path = os.path.join("/home/asandygulova/dock/krsl_173_1708/videos/", sample["video"])
        vr = VideoReader(video_path, ctx=cpu(0))

        return vr

    def create_lm(self):
        path = self.root / "models" / "LanguageModel" / "MS-train-4gram.sri.lm.gz"
        return SRILM(path, self.create_vocab())


class PhoenixTCorpus(PhoenixCorpus):
    def __init__(self):
        # TODO:
        raise NotImplementedError
