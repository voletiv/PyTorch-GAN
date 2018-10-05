import argparse
import datetime
import math
import numpy as np
import os
import sys

import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch

import utils

from wgan_gp_models import *


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = Variable(Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dset", type=str, default="mnist", help="Directory of images dataset")
    parser.add_argument("--n_epochs", type=int, default=10000, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
    # parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")
    parser.add_argument("--save_images_every_n_batches", type=int, default=100, help="interval betwen image samples")
    parser.add_argument("--save_weights_every_n_batches", type=int, default=100, help="interval betwen weight samples")
    parser.add_argument("--keep_last_n_weights", type=int, default=20, help="interval betwen weight samples")
    parser.add_argument("--dont_shuffle", action="store_true", help="dont_shuffle")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    opt = parser.parse_args()

    if opt.dont_shuffle:
        opt.shuffle = False
    else:
        opt.shuffle = True

    model_name = "{0:%Y%m%d_%H%M%S}_WGAN_GP_{1}".format(datetime.datetime.now(), os.path.basename(opt.dset.rstrip('/')))
    opt.model_name = model_name

    print(opt)

    # Setup logging
    model_dir, fig_dir = utils.setup_dirs(opt)

    # Configure dataloader
    dataloader = utils.configure_dataloader(opt.dset, opt.batch_size, opt.img_size, opt.shuffle)

    # Find img_shape
    # img_shape = (opt.channels, opt.img_size, opt.img_size)
    a, _ = next(iter(dataloader))
    img_shape = (a.shape[1], a.shape[2], a.shape[3])

    cuda = True if torch.cuda.is_available() else False

    # Loss weight for gradient penalty
    lambda_gp = 10

    # Initialize generator and discriminator
    generator = Generator(opt.latent_dim, img_shape)
    discriminator = Discriminator(img_shape)

    if cuda:
        generator.cuda()
        discriminator.cuda()

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    n_batches = len(dataloader)

    # ----------
    #  Training
    # ----------

    try:

        d_losses_in_n_critic = 0
        d_losses = []
        g_losses = []

        for epoch in range(opt.n_epochs):
            for batch, (imgs, _) in enumerate(dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

                # Generate a batch of images
                fake_imgs = generator(z)

                # Real images
                real_validity = discriminator(real_imgs)
                # Fake images
                fake_validity = discriminator(fake_imgs)
                # Gradient penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, fake_imgs.data)
                # Adversarial loss
                d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + lambda_gp * gradient_penalty

                d_losses_in_n_critic += d_loss.item()

                d_loss.backward()
                optimizer_D.step()

                optimizer_G.zero_grad()

                # Train the generator every n_critic steps
                if batch % opt.n_critic == 0:

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Generate a batch of images
                    fake_imgs = generator(z)
                    # Loss measures generator's ability to fool the discriminator
                    # Train on fake images
                    fake_validity = discriminator(fake_imgs)
                    g_loss = -torch.mean(fake_validity)

                    g_loss.backward()
                    optimizer_G.step()

                    print(
                        "[{0:%Y/%m/%d %H:%M:%S}] [Epoch {1:d}/{2:d}] [Batch {3:d}/{4:d}] [D loss: {5:f}] [G loss: {6:f}]".format(datetime.datetime.now(),
                            epoch+1, opt.n_epochs, batch+1, n_batches, d_loss.item(), g_loss.item())
                    )

                    d_losses.append(d_losses_in_n_critic/opt.n_critic)
                    d_losses_in_n_critic = 0
                    g_losses.append(g_loss.item())

                if (epoch*n_batches + batch) % opt.save_images_every_n_batches == 0:
                    utils.plot_images(fake_imgs, fig_dir, model_name, epoch*n_batches + batch)
                    utils.plot_losses(d_losses, g_losses, opt.n_critic, fig_dir, model_name)

                if (epoch*n_batches + batch) % opt.save_weights_every_n_batches == 0:
                    utils.save_models({"gen": generator, "disc": discriminator}, model_dir, epoch*n_batches + batch,
                                      opt.keep_last_n_weights)

    except KeyboardInterrupt:
        pass

    # Save gen and disc
    print("Saving final models...")
    torch.save(generator, os.path.join(model_dir, model_name + "_generator.pth"))
    torch.save(discriminator, os.path.join(model_dir, model_name + "_discriminator.pth"))

    print("DONE.")
