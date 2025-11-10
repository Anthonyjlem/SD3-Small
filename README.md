# SD3-Small
This is a work in progress.

This is an implementation of a small version of Stable Diffusion 3 from scratch trained on the COCO dataset (original architecture seen below from https://arxiv.org/abs/2403.03206). The architecture has been modeified so that the three text encoders have been replaced by a single CLIP-B/16 text encoder. The entire full-precision model is designed to fit within 2 GB of memory, including the text encoder and VAE. Output image generations are 128 x 128 pixels.

<img width="1116" height="791" alt="image" src="https://github.com/user-attachments/assets/a90665a9-78e8-4a98-b3cb-bd3b968a4876" />

## Usage
```
python train.py
```

## To Do
- [ ] Optimize training time
