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
import os

import numpy as np
from PIL import Image

from bicublic import bicubic_interpolation

try:
    os.makedirs("./data/BSDS300/X2/data/")
    os.makedirs("./data/BSDS300/X3/data/")
    os.makedirs("./data/BSDS300/X4/data/")
except OSError:
    pass

# Make data
for filename in os.listdir("./data/BSDS300/val/"):
    src_img = np.array(Image.open("./data/BSDS300/val/" + filename))
    dst_height = src_img.shape[0] // 3
    dst_width = src_img.shape[1] // 3
    dst_img = bicubic_interpolation(src_img, dst_height, dst_width)
    dst_img = Image.fromarray(dst_img.astype("uint8")).convert("RGB")
    dst_img.save("./data/BSDS300/X3/data/" + filename)

# Make target
for filename in os.listdir("./data/BSDS300/val/"):
    src_img = np.array(Image.open("./data/BSDS300/val/" + filename))
    dst_height = src_img.shape[0] // 3 * 3
    dst_width = src_img.shape[1] // 3 * 3
    dst_img = bicubic_interpolation(src_img, dst_height, dst_width)
    dst_img = Image.fromarray(dst_img.astype("uint8")).convert("RGB")
    dst_img.save("./data/BSDS300/X3/target/" + filename)
