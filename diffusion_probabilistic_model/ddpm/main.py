# -*- coding: utf-8 -*-
# Copyright (c) XiMing Xing. All rights reserved.
# Author: XiMing Xing
# Description:

import sys
import os
import os.path as osp
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

rootPath = os.path.abspath(os.path.join(os.getcwd(), "../.."))
sys.path.append(rootPath)

from deep_nn.utils import set_random_seed

from ddpm import DenoiseDiffusion
from unet import UNet
from dataset import MNISTDataset

PROJECT_ROOT_DIR = "."
IMAGES_PATH = osp.join(PROJECT_ROOT_DIR, "images")
os.makedirs(IMAGES_PATH, exist_ok=True)


def sample(img_prefix, diffusion, n_steps, n_samples, image_channels, image_size, device):
    """Sample images"""
    with torch.no_grad():
        # $x_T \sim p(x_T) = \mathcal{N}(x_T; \mathbf{0}, \mathbf{I})$
        x = torch.randn([n_samples, image_channels, image_size, image_size], device=device)

        # Remove noise for $T$ steps
        for t_ in range(n_steps):
            # $t$
            t = n_steps - t_ - 1
            # Sample from $\textcolor{lightgreen}{p_\theta}(x_{t-1}|x_t)$
            x = diffusion.p_sample(x, x.new_full((n_samples,), t, dtype=torch.long))

        name = random.sample('zyxwvutsrqponmlkjihgfedcba', 5)
        torch.save(x, str(img_prefix + "".join(name)))


def main(args):
    device = torch.device("cuda:{}".format(args.device) if torch.cuda.is_available() else "cpu")

    eps_model = UNet(
        image_channels=1,
        n_channels=64,
        ch_mults=[1, 2, 2, 4],
        is_attn=[False, False, False, True]
    ).to(device)
    diffusion = DenoiseDiffusion(eps_model, args.n_steps, device)

    optimizer = torch.optim.Adam(eps_model.parameters(), args.lr)

    if args.data_path is None or '':
        DATA_PATH = osp.join(PROJECT_ROOT_DIR, "data")
        os.makedirs(DATA_PATH, exist_ok=True)
    else:
        DATA_PATH = args.data_path

    dataset = MNISTDataset(DATA_PATH, args.image_size)
    image_channels = 1
    dataloader = DataLoader(dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=False, pin_memory=True)

    pbar = tqdm(range(args.epochs))
    for epoch in pbar:
        for idx, (img, target) in enumerate(dataloader):
            img = img.to(device)

            loss = diffusion.loss(img)
            pbar.set_description(f"loss: {loss.item():.3f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            img_prefix = osp.join(IMAGES_PATH, f"_{epoch}_{idx}_")  # image prefix
            sample(img_prefix, diffusion, args.n_steps, args.n_samples, image_channels, args.image_size, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="DDPM Training")

    parser.add_argument('--data-path', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=32)
    parser.add_argument('--n-steps', type=int, default=1000)
    parser.add_argument('--n-samples', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-5)

    args = parser.parse_args()

    set_random_seed(seed=args.seed)
    main(args)
