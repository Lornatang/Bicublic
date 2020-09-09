# Bicublic

## Requirements

```bash
pip install -r requirements.txt
```

## Test benckmark
```bash
# Help information
usage: test.py [-h] [--dataroot DATAROOT] [--scale-factor {2,3,4}]

optional arguments:
  -h, --help            show this help message and exit
  --dataroot DATAROOT   The directory address where the image needs to be
                        processed. (default: `./data/Set5`).
  --scale-factor {2,3,4}
                        Image scaling ratio. (default: `4`).
# Example
python test.py --dataroot ./data/Set --scale-factor 4
```

## Test image

It will generate images based on bicubic linear interpolation.
```bash
# Example
python test_image.py
```
