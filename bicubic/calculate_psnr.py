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

import cv2
import numpy as np


def cal_psnr(img1, img2):
    r"""Python simply calculates the maximum signal noise ratio.

    Args:
        img1 (str): Address of fake high resolution image.
        img2 (str): Address of real high resolution image.

    ..math:
        10 \cdot \log _{10}\left(\frac{MAX_{I}^{2}}{MSE}\right)

    Returns:
        Maximum signal to noise ratio between two images.
    """
    prediction = cv2.imread(img1)
    target = cv2.imread(img2)

    if prediction.shape != target.shape:
        # Make sure the two pictures are the same size
        target = cv2.resize(target, (prediction.shape[1], prediction.shape[0]),
                            interpolation=cv2.INTER_CUBIC)

    mse = np.mean((prediction - target) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)
