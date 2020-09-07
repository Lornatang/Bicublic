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
import argparse
import numpy as np
from PIL import Image

from bicublic import bicubic_interpolation
from bicublic import cal_psnr
from bicublic import cal_ssim

parser = argparse.ArgumentParser()
parser.add_argument("--dataroot", type=str, default="./data/Set5",
                    help="The directory address where the image needs "
                         "to be processed. (default: `./data/Set5`).")
parser.add_argument("--scale-factor", type=int, default=4, choices=[2, 3, 4],
                    help="Image scaling ratio. (default: `4`).")
args = parser.parse_args()

# Evaluate algorithm performance
total_psnr_value = 0.0
total_ssim_value = 0.0
# Count the number of files in the directory
total_file = 0

dataroot = f"{args.dataroot}/X{args.scale_factor}/data"
target = f"{args.dataroot}/X{args.scale_factor}/target"
scale_factor = args.scale_factor

for filename in os.listdir(dataroot):
    src_img = np.array(Image.open(f"{dataroot}/{filename}"))
    dst_height = src_img.shape[0] * scale_factor
    dst_width = src_img.shape[1] * scale_factor
    dst_img = bicubic_interpolation(src_img, dst_height, dst_width)
    dst_img = Image.fromarray(dst_img.astype("uint8")).convert("RGB")
    dst_img.save(f"results/{filename}")

    total_psnr_value += cal_psnr(f"results/{filename}",
                                 f"{target}/{filename}")
    total_ssim_value += cal_ssim(f"results/{filename}",
                                 f"{target}/{filename}")

    total_file += 1

print(f"Avg PSNR: {total_psnr_value / total_file:.2f} "
      f"Avg SSIM: {total_ssim_value / total_file:.4f}")
