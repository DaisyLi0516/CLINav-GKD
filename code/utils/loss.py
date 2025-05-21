from torchvision.transforms import Compose, ToTensor, Normalize, RandomHorizontalFlip
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import torch as t
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


class AdaptiveResize(object):
    """Resize the input PIL Image to the given size adaptively.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=InterpolationMode.BILINEAR, image_size=None):
        assert isinstance(size, int)
        self.size = size
        self.interpolation = interpolation
        if image_size is not None:
            self.image_size = image_size
        else:
            self.image_size = None

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        h, w = img.size

        if self.image_size is not None:
            if h < self.image_size or w < self.image_size:
                return transforms.Resize(self.image_size, self.interpolation)(img)

        if h < self.size or w < self.size:
            return img
        else:
            return transforms.Resize(self.size, self.interpolation)(img)


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _preprocess_self():
    return Compose([
        _convert_image_to_rgb,
        AdaptiveResize(768),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def TopoLa_t(anc, pos, lambda_val, iters=20):
    """Original TopoLa: given two sets of embeddings anc, pos,
    returns U @ diag(S') @ U^T, where S' = f(S, lambda_val)."""
    def _calc_s_prime(S, l):
        # Newton-series approximation of (I + l S^2)^{-1}
        S_ = S**2
        for i in range(iters):
            S_ -= (-1)**i * l**(i+1) * S_ * S**2
        return t.nan_to_num(S_)

    # Stack anchor & positive embeddings, do SVD
    combined = t.cat([anc, pos], dim=0)
    U, S, _ = t.svd(t.nan_to_num(combined))
    U = t.nan_to_num(U)
    S = t.nan_to_num(S)

    # Compute the “topological spectrum”
    S_p = _calc_s_prime(S, lambda_val)
    return U @ t.diag(S_p) @ U.t()

def Topola(anc_t, pos_t, anc_s, pos_s, lambda_val):
    """
    anc_t, pos_t : teacher embeddings (two halves of the batch)
    anc_s, pos_s : student embeddings
    lambda_val   : your TopoLa hyperparameter
    """
    # 1) build the teacher’s topological map
    topo_t = TopoLa_t(anc_t, pos_t, lambda_val)
    # 2) build the student’s topological map
    topo_s = TopoLa_t(anc_s, pos_s, lambda_val)
    # 3) penalize their difference
    return F.mse_loss(topo_s, topo_t)

