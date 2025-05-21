from pathlib import Path
import torch
from torch.utils.data import Dataset
import scipy.io as sio
from PIL import Image
import clip

class YourDataset(Dataset):
    """
    Loads echocardiography frames and text prompts for distillation.
    Returns (image, text_tokens, index).
    """
    def __init__(self, img_dir, label_mat, split_txt, preprocess, mtl=0):
        self.img_dir = Path(img_dir)
        self.labels = sio.loadmat(label_mat)[f"MC6D_{split_txt.stem}"]
        self.samples = [line.strip() for line in Path(split_txt).read_text().splitlines()]
        self.preprocess = preprocess
        self._prepare_texts()

    def _prepare_texts(self):
        classes = ['PLAX','A2C','A3C','A4C','A5C','PSAX-PM','PSAX-AV']
        trans   = ['others','feet','head','chest','back','upward','downward']
        rot     = ['others','left','right','up','down','clockwise','counterclockwise']
        self.texts = []
        for idx in range(len(self.samples)):
            c,p,r = self.labels[idx][[0,2,3]]
            txt = (f"a photo of a cardiac ultrasound {classes[c-1]} view "
                   f"with the probe moving {trans[p]} and rotating {rot[r]}")
            self.texts.append(txt)
        self.tokens = clip.tokenize(self.texts)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path = self.img_dir / f"{self.samples[idx]}.jpg"
        img = Image.open(path).convert('RGB')
        img = self.preprocess(img)
        return img, self.tokens[idx], idx