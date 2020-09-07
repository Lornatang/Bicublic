# Copyright 2020 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from math import exp

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image


def gaussian(kernel_size, sigma):
    r"""Implementation of Gaussian kernel function.

    Args:
        kernel_size (int): Gaussian kernel size.
        sigma (float): Penalty coefficient.

    ..math:
        K(x, y)=e^{-\gamma|| x-y||^{2}}

    Returns:
        Linear separable data processed by Gaussian kernel.

    """
    gauss = torch.Tensor(
        [exp(-(x - kernel_size // 2) ** 2 / float(2 * sigma ** 2)) for x in
         range(kernel_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    r"""Create a 2D flat window.

    Args:
        window_size (int): Flat window size.
        channel (int): The number of channels of the image.

    Returns:
        2D flat window.
    """
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size,
                               window_size).contiguous()
    return window


def cal_ssim(img1, img2, window_size=11, size_average=True):
    r"""The structural similarity between the two images was calculated.

    Args:
        img1 (str): Address of fake high resolution image.
        img2 (str): Address of real high resolution image.
        window_size (int): Flat window size. Default: 11.
        size_average (int): The pixel size of the fill window.

    ..math:
        \operatorname{SSIM}(x, y)=\frac{\left(2 \mu_{x} \mu_{y}+c_{1}\right)
        \left(\sigma_{x y}+c_{2}\right)}{\left(\mu_{x}^{2}+\mu_{y}^{2}+c_{1}
        \right)\left(\sigma_{x}^{2}+\sigma_{y}^{2}+c_{2}\right)}

    Returns:
        Structural similarity of two images.

    """
    prediction = Image.open(img1)
    target = Image.open(img2)

    if prediction.size != target.size:
        # Make sure the two pictures are the same size
        target = target.resize((prediction.size[0], prediction.size[1]))

    # Convert pictures to tensor format
    pil_to_tensor = transforms.ToTensor()
    with torch.no_grad():
        prediction = pil_to_tensor(prediction).unsqueeze(0)
        target = pil_to_tensor(target).unsqueeze(0)

    (_, channel, _, _) = prediction.size()
    window = create_window(window_size, channel)

    if prediction.is_cuda:
        window = window.cuda(prediction.get_device())
    window = window.type_as(prediction)

    mu1 = F.conv2d(prediction, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(prediction * prediction, window,
                         padding=window_size // 2,
                         groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2,
                         groups=channel) - mu2_sq
    sigma12 = F.conv2d(prediction * target, window, padding=window_size // 2,
                       groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
