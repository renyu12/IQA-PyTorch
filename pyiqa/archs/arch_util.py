from collections import OrderedDict
import collections.abc
from itertools import repeat
import numpy as np

import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
import torchvision.transforms as T

from .constants import OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from pyiqa.utils.download_util import load_file_from_url


# --------------------------------------------
# IQA utils
# --------------------------------------------

def dist_to_mos(dist_score: torch.Tensor) -> torch.Tensor:
    """Convert distribution prediction to mos score.
    For datasets with detailed score labels, such as AVA

    Args:
        dist_score (tensor): (*, C), C is the class number

    Output:
        mos_score (tensor): (*, 1)
    """
    num_classes = dist_score.shape[-1]
    mos_score = dist_score * torch.arange(1, num_classes + 1).to(dist_score)
    mos_score = mos_score.sum(dim=-1, keepdim=True)
    return mos_score

def random_crop(input_list, crop_size, crop_num):
    """
    Randomly crops the input tensor(s) to the specified size and number of crops.

    Args:
        input_list (list or tensor): List of input tensors or a single input tensor.
        crop_size (int or tuple): Size of the crop. If an int is provided, a square crop of that size is used.
        If a tuple is provided, a crop of that size is used.
        crop_num (int): Number of crops to generate.

    Returns:
        tensor or list of tensors: If a single input tensor is provided, a tensor of cropped images is returned.
            If a list of input tensors is provided, a list of tensors of cropped images is returned.
    """
    if not isinstance(input_list, collections.abc.Sequence):
        input_list = [input_list]

    b, c, h, w = input_list[0].shape
    ch, cw = to_2tuple(crop_size)

    if min(h, w) <= crop_size:
        scale_factor = (crop_size + 1) / min(h, w)
        input_list = [
            F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            for x in input_list
        ]
        b, c, h, w = input_list[0].shape

    crops_list = [[] for i in range(len(input_list))]
    for i in range(crop_num):
        sh = np.random.randint(0, h - ch + 1)
        sw = np.random.randint(0, w - cw + 1)
        for j in range(len(input_list)):
            crops_list[j].append(input_list[j][..., sh : sh + ch, sw : sw + cw])

    # renyu: 返回时不改变B C H W的格式，只是B变成B*crop_num
    for i in range(len(crops_list)):
        crops_list[i] = torch.stack(crops_list[i], dim=1).reshape(
            b * crop_num, c, ch, cw
        )

    if len(crops_list) == 1:
        crops_list = crops_list[0]
    return crops_list


def uniform_crop(input_list, crop_size, crop_num):
    """
    Crop the input_list of tensors into multiple crops with uniform steps according to input size and crop_num.

    Args:
        input_list (list or torch.Tensor): List of input tensors or a single input tensor.
        crop_size (int or tuple): Size of the crops. If int, the same size will be used for height and width.
            If tuple, should be (height, width).
        crop_num (int): Number of crops to generate.

    Returns:
        torch.Tensor or list of torch.Tensor: Cropped tensors. If input_list is a list, the output will be a list
            of cropped tensors. If input_list is a single tensor, the output will be a single tensor.
    """
    if not isinstance(input_list, collections.abc.Sequence):
        input_list = [input_list]

    b, c, h, w = input_list[0].shape
    ch, cw = to_2tuple(crop_size)

    if min(h, w) <= crop_size:
        scale_factor = (crop_size + 1) / min(h, w)
        input_list = [
            F.interpolate(x, scale_factor=scale_factor, mode="bilinear")
            for x in input_list
        ]
        b, c, h, w = input_list[0].shape
    
    step_h = (h - ch) // int(np.sqrt(crop_num))
    step_w = (w - cw) // int(np.sqrt(crop_num))

    # renyu: 返回时不改变B C H W的格式，只是B变成B*crop_num
    crops_list = []
    for idx, inp in enumerate(input_list):
        tmp_list = []
        for i in range(int(np.ceil(np.sqrt(crop_num)))):
            for j in range(int(np.ceil(np.sqrt(crop_num)))):
                sh = i * step_h
                sw = j * step_w
                tmp_list.append(inp[..., sh : sh + ch, sw : sw + cw])
        crops_list.append(
            torch.stack(tmp_list[:crop_num], dim=1).reshape(b * crop_num, c, ch, cw)
        )

    if len(crops_list) == 1:
        crops_list = crops_list[0]
    return crops_list


def clip_preprocess_tensor(x: torch.Tensor, model):
    """clip preprocess function with tensor input.

    NOTE: Results are slightly different with original preprocess function with PIL image input, because of differences in resize function.
    """
    # Bicubic interpolation
    x = T.functional.resize(
        x,
        model.visual.input_resolution,
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    )
    # Center crop
    x = T.functional.center_crop(x, model.visual.input_resolution)
    x = T.functional.normalize(x, OPENAI_CLIP_MEAN, OPENAI_CLIP_STD)
    return x


# --------------------------------------------
# Common utils
# --------------------------------------------


def clean_state_dict(state_dict):
    # 'clean' checkpoint by removing .module prefix from state dict if it exists from parallel training
    cleaned_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        cleaned_state_dict[name] = v
    return cleaned_state_dict


def load_pretrained_network(net, model_path, strict=True, weight_keys=None):
    if model_path.startswith("https://") or model_path.startswith("http://"):
        model_path = load_file_from_url(model_path)
    print(f"Loading pretrained model {net.__class__.__name__} from {model_path}")
    state_dict = torch.load(model_path, map_location=torch.device("cpu"), weights_only=False)
    if weight_keys is not None:
        state_dict = state_dict[weight_keys]
    state_dict = clean_state_dict(state_dict)
    net.load_state_dict(state_dict, strict=strict)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """
    Initializes the weights of the given module(s) using Kaiming Normal initialization.

    Args:
        module_list (list or nn.Module): List of modules or a single module to initialize.
        scale (float, optional): Scaling factor for the weights. Default is 1.
        bias_fill (float, optional): Value to fill the biases with. Default is 0.
        **kwargs: Additional arguments for the Kaiming Normal initialization.

    Returns:
        None

    Example:
        >>> import torch.nn as nn
        >>> from arch_util import default_init_weights
        >>> model = nn.Sequential(
        >>>     nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
        >>>     nn.ReLU(),
        >>>     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
        >>>     nn.ReLU(),
        >>>     nn.Linear(64 * 32 * 32, 10)
        >>> )
        >>> default_init_weights(model, scale=0.1, bias_fill=0.01)
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
