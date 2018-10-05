import cv2
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import subprocess
import torch
import torchvision.transforms as transforms

from torchvision.utils import save_image
from torchvision import datasets


def configure_dataloader(dset, batch_size, img_size, shuffle=True):
    # Configure data loader
    if dset == 'mnist':
        os.makedirs("../../data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )
    else:
        data_transforms = transforms.Compose([
                                              transforms.Resize((img_size, img_size)),
                                              # transforms.RandomResizedCrop(128),
                                              # transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                            ])
        dataset = datasets.ImageFolder(root=dset, transform=data_transforms)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


def setup_dirs(opt):

    model_name = opt.model_name

    # Output path where we store experiment log and weights
    model_dir = os.path.join("../models", model_name)
    fig_dir = os.path.join("../figures", model_name)

    print("Creating", model_dir, "and", fig_dir)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # Copy main.py, train.py and model.py
    py_files = glob.glob("*.py")
    for py_file in py_files:
        subprocess.call(['cp', py_file, model_dir])

    # Write all config params
    print("Writing config params in", os.path.join(model_dir, 'config.txt'))
    with open(os.path.join(model_dir, 'config.txt'), 'w') as f:
        for i in opt.__dict__:
            f.write(str(i) + ' ' + str(opt.__dict__[i]) + '\n')

    print("Writing config params in", os.path.join(fig_dir, 'config.txt'))
    with open(os.path.join(fig_dir, 'config.txt'), 'w') as f:
        for i in opt.__dict__:
            f.write(str(i) + ' ' + str(opt.__dict__[i]) + '\n')

    return model_dir, fig_dir


def plot_images(fake_imgs, fig_dir, model_name, batch, suffix='train', MAX_FRAMES_PER_GIF=100):

    print("Saving images...")
    save_image(fake_imgs.data[:16], os.path.join(fig_dir, "current_batch.png"), nrow=4, normalize=True)

    # Make gif
    gif_frames = []

    # Read old gif frames
    try:
        gif_frames_reader = imageio.get_reader(os.path.join(fig_dir, model_name + "_%s.gif" % suffix))
        for frame in gif_frames_reader:
            gif_frames.append(frame[:, :, :3])
    except:
        pass

    # Append new frame
    fake_im = np.transpose(fake_imgs.data[0], (1, 2, 0))
    fake_im_min = fake_im.min(); fake_im_max = fake_im.max()
    fake_im = (fake_im - fake_im_min)/(fake_im_max - fake_im_min) * 255
    im = cv2.putText(np.concatenate((np.zeros((32, fake_im.shape[1], fake_im.shape[2])), fake_im), axis=0),
                     '%s iter' % str(batch+1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA).astype('uint8')
    gif_frames.append(im)

    # If frames exceeds, save as different file
    if len(gif_frames) > MAX_FRAMES_PER_GIF:
        print("Splitting the GIF...")
        gif_frames_00 = gif_frames[:MAX_FRAMES_PER_GIF]
        num_of_gifs_already_saved = len(glob.glob(os.path.join(fig_dir, model_name + "_%s_*.gif" % suffix)))
        print("Saving", os.path.join(fig_dir, model_name + "_%s_%03d.gif" % (suffix, num_of_gifs_already_saved)))
        imageio.mimsave(os.path.join(fig_dir, model_name + "_%s_%03d.gif" % (suffix, num_of_gifs_already_saved)), gif_frames_00)
        gif_frames = gif_frames[MAX_FRAMES_PER_GIF:]

    # Save gif
    print("Saving", os.path.join(fig_dir, model_name + "_%s.gif" % suffix))
    imageio.mimsave(os.path.join(fig_dir, model_name + "_%s.gif" % suffix), gif_frames)


def plot_losses(d_losses, g_losses, freq,
                fig_dir, model_name, init_epoch=0):
    print("Plotting losses...")
    epochs = np.arange(0, len(d_losses)*freq, freq) + init_epoch
    fig = plt.figure()
    plt.subplot(211)
    plt.plot(epochs, d_losses)
    plt.xlabel("Iterations")
    plt.title('Discriminator Loss')
    plt.plot(epochs, g_losses)
    plt.xlabel("Iterations")
    plt.title('Generator Loss')
    # plt.legend()
    # plt.title("Losses")
    plt.savefig(os.path.join(fig_dir, model_name + "_losses.png"), bbox_inches='tight')
    plt.clf()
    plt.close()


def save_models(models_dict, model_dir, iter_num, keep_last_n_weights):
    print("Saving models...")
    model_names = models_dict.keys()
    purge_weights(keep_last_n_weights, model_dir, model_names)
    for model_name in model_names:
        torch.save(models_dict[model_name].state_dict(), os.path.join(model_dir, model_name + "_iter_{0:06d}".format(iter_num) + ".pth"))


def purge_weights(keep_last_n_weights, model_dir, model_names):
    for name in model_names:
        weight_files = sorted(glob.glob(os.path.join(model_dir, name + "*")))
        for weight_file in weight_files[:-keep_last_n_weights]:
            os.remove(os.path.realpath(weight_file))
