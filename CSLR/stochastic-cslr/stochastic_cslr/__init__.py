from torch.utils.model_zoo import load_url

from .model import Model
import torch

model_urls = {
    "dfl": "https://github.com/zheniu/stochastic-cslr-ckpt/raw/main/dfl.pth",
    "sfl": "https://github.com/zheniu/stochastic-cslr-ckpt/raw/main/sfl.pth",
}


def load_model(use_sfl=True, epoch=30, lang="Russian"):
    if lang == "Russian":
        vocab_size = 314
    else:
        vocab_size = 1232

    model = Model(
        vocab_size=vocab_size,
        dim=512,
        max_num_states=5 if use_sfl else 2,
        use_sfl=use_sfl
    )

    if epoch==100:
        print("Loading the model ...")
        model.load_state_dict(torch.load(f"/app/CSLR/stochastic-cslr/{epoch}.pth", map_location=torch.device('cpu')))
    else:
        model.load_state_dict(torch.load(f"/app/CSLR/stochastic-cslr/{epoch}.pth", map_location=torch.device('cpu'))["model"])

    return model
