# Adapted from https://github.com/jmhessel/clipscore/blob/1036465276513621f77f1c2208d742e4a430781f/clipscore.py
"""
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
"""



import collections
import warnings

import clip
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor


class CLIPCapDataset(torch.utils.data.Dataset):

    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(c_data, truncate=True).squeeze()
        return {"caption": c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose(
            [
                Resize(n_px, interpolation=Image.BICUBIC),
                CenterCrop(n_px),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(
                    (0.48145466, 0.4578275, 0.40821073),
                    (0.26862954, 0.26130258, 0.27577711),
                ),
            ]
        )

    def __getitem__(self, idx):
        c_data = self.data[idx]
        if isinstance(c_data, str):  # If it's a file path
            image = Image.open(c_data)
        else:  # If it's already a PIL image
            image = c_data
        image = self.preprocess(image)
        return {'image':image}
    

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size,
        shuffle=False,
    )
    all_text_features = []
    with torch.no_grad():
        for b in data:
            b = b["caption"].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size,
        shuffle=False,
    )
    all_image_features = []
    with torch.no_grad():
        for b in data:
            b = b["image"].to(device)
            if device == "cuda":
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    """
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    """
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    images = images / np.sqrt(np.sum(images**2, axis=1, keepdims=True))
    candidates = candidates / np.sqrt(np.sum(candidates**2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates



def clip_metrics(
    clip_model, images, device, batch_size, prompts_A
):

    image_feats = extract_all_images(images, clip_model, device, batch_size=batch_size)
    _, prompt_A_sim, _ = get_clip_score(
        clip_model,
        image_feats,
        prompts_A,
        device,
    )
    
    return np.array(prompt_A_sim).mean()
    