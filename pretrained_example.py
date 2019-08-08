# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for generating an image using pre-trained StyleGAN generator."""

import argparse
import os
import pickle
from tqdm import tqdm
import numpy as np
import PIL.Image
import dnnlib.tflib as tflib
import config
from models.ModelRetriever import ModelRetriever


def do_parsing():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Use pretrained models to generate random images")
    parser.add_argument("--file", required=True, type=str, help="Pretrained model file to load")
    parser.add_argument("--output_dir", required=False, type=str, default='output/', help="Output directory")
    parser.add_argument("--seed", required=False, type=int, default=42, help="Random seed for seeds generation")
    parser.add_argument("--n_images", required=False, type=int, default=100, help="Number of images to generate")
    parser.add_argument("--truncation_psi", required=False, type=float, default=0.7, help="Truncation psi")
    args = parser.parse_args()
    return args


def main():

    # Load params
    args = do_parsing()
    print(args)

    # Initialize TensorFlow.
    tflib.init_tf()

    # Load pre-trained network, already on file system
    ###model_filepath = ModelRetriever().get_model_filepath(args.model_name)

    with open(args.file, "rb") as f:
        _G, _D, Gs = pickle.load(f)

        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    # Print network details.
    Gs.print_layers()

    # Pick latent vector.
    seed = args.seed
    rnd = np.random.RandomState(seed)
    latents = rnd.randn(args.n_images, Gs.input_shape[1])

    ###output_dir = os.path.join(config.result_dir, args.model_name, "seed_" + str(seed))
    output_dir = args.output_dir

    for index, latent in tqdm(enumerate(latents), desc="image"):

        # Generate image.
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(np.expand_dims(latent, axis=0), None, truncation_psi=args.truncation_psi, randomize_noise=True, output_transform=fmt)

        # Save image.
        os.makedirs(output_dir, exist_ok=True)
        png_filename = os.path.join(output_dir, 'example_' + str(index) + '.png')
        PIL.Image.fromarray(images[0], 'RGB').save(png_filename)


if __name__ == "__main__":
    main()
