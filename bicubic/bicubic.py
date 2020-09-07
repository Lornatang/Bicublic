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
import math

import numpy as np


def bibubic(x):
    r"""Different weights of 16 pixels are generated

    Args:
        x (float): Image pixel value.

    Returns:
        Pixel weight
    """
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0


def bicubic_interpolation(src_img, dst_height, dst_width):
    r"""Implementation of bicubic linear interpolation algorithm in Python.

    Args:
        src_img (PIL.Image.open): Image read by pillow Library.
        dst_height (int): Height of target image.
        dst_width (int): Width of target image.

    Returns:
        Image after bicubic linear interpolation.

    """
    src_h, src_w, _ = src_img.shape
    img = np.zeros((dst_height, dst_width, 3), dtype=np.uint8)
    for height in range(dst_height):
        for width in range(dst_width):
            scr_x = height * (src_h / dst_height)
            scr_y = width * (src_w / dst_width)
            x = math.floor(scr_x)
            y = math.floor(scr_y)
            u = scr_x - x
            v = scr_y - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= src_h or y + jj >= src_w:
                        continue
                    tmp += src_img[x + ii, y + jj] * bibubic(ii - u) * bibubic(
                        jj - v)
            img[height, width] = np.clip(tmp, 0, 255)
    return img
