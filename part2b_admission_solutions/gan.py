"""GAN training routine for admission exercise 4 (split out for clarity).

The original ``train_gan`` from ``debugging.ipynb`` is reproduced below with
both bugs fixed. The :class:`Generator` and :class:`Discriminator` modules are
the same as in the source notebook; only the training loop changed.

Bug fixes (see docstring on :func:`train_gan` for the full story):

* the last mini-batch of an epoch may be smaller than ``batch_size``, so the
  labels must be sized from ``real_samples.size(0)`` rather than the constant
  ``batch_size``;
* the logging condition ``n == batch_size - 1`` was tied to the wrong variable
  and is replaced by ``n == len(train_loader) - 1`` which logs once per epoch.
"""

from __future__ import annotations

import time
from typing import Optional

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
from IPython.display import clear_output, display


class Generator(nn.Module):
    """Fully-connected generator that maps a 100-d latent code to a 28x28 image."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).view(x.size(0), 1, 28, 28)


class Discriminator(nn.Module):
    """Fully-connected discriminator that scores a 28x28 image in [0, 1]."""

    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x.view(x.size(0), 784))


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def train_gan(
    batch_size: int = 32,
    num_epochs: int = 100,
    device: Optional[str] = None,
    data_root: str = ".",
    show_progress: bool = True,
) -> tuple[Generator, Discriminator]:
    """Train a small fully-connected GAN on MNIST.

    Bugs fixed in this version:

    1. The original used the constant ``batch_size`` to build the label
       tensors. When the dataset length is not divisible by ``batch_size`` the
       final mini-batch is smaller and the shapes no longer line up, e.g.::

           ValueError: Using a target size (torch.Size([128, 1])) that is
           different to the input size (torch.Size([96, 1])).

       The current implementation uses ``real_samples.size(0)`` for the label
       shape and uses the same value (``current_bs``) consistently for the
       generator input. As a defensive bonus we also pass ``drop_last=True``
       to the DataLoader so the issue cannot reappear if someone forgets the
       fix elsewhere.

    2. The original logged with ``if n == batch_size - 1``, which is a typo:
       it conflates the index of a batch *within an epoch* with the *number
       of samples per batch*. The fix logs once per epoch with
       ``n == len(train_loader) - 1``.
    """
    device = device or _default_device()

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    try:
        train_set = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )
    except Exception:
        # Mirror used in the source notebook - keeps download working when the
        # primary host is unreachable.
        print("Failed to download MNIST, retrying with the S3 mirror")
        torchvision.datasets.MNIST.resources = [
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz",
                "f68b3c2dcbeaaa9fbdd348bbdeb94873",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz",
                "d53e105ee54ea40749a09fcbcd1e9432",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz",
                "9fb629c4189551a2d022fa330f9573f3",
            ),
            (
                "https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz",
                "ec29112dd5afa0611ce80d1b7f02629c",
            ),
        ]
        train_set = torchvision.datasets.MNIST(
            root=data_root, train=True, download=True, transform=transform
        )

    # ``drop_last`` makes the structural fix below redundant in normal cases,
    # but we still derive label sizes from the real tensor so the loop is
    # correct on its own merits.
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
    )

    if show_progress:
        real_samples, _ = next(iter(train_loader))
        fig = plt.figure()
        for i in range(16):
            sub = fig.add_subplot(4, 4, 1 + i)
            sub.imshow(real_samples[i].reshape(28, 28), cmap="gray_r")
            sub.axis("off")
        fig.tight_layout()
        fig.suptitle("Real images")
        display(fig)
        time.sleep(1)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    lr = 1e-4
    loss_function = nn.BCELoss()
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr)

    batches_per_epoch = len(train_loader)
    for epoch in range(num_epochs):
        for n, (real_samples, _) in enumerate(train_loader):
            current_bs = real_samples.size(0)  # bug-fix 1: use actual batch size

            real_samples = real_samples.to(device=device)
            real_labels = torch.ones((current_bs, 1), device=device)
            fake_labels = torch.zeros((current_bs, 1), device=device)

            latent = torch.randn((current_bs, 100), device=device)
            generated = generator(latent)
            all_samples = torch.cat((real_samples, generated))
            all_labels = torch.cat((real_labels, fake_labels))

            discriminator.zero_grad()
            d_out = discriminator(all_samples)
            loss_d = loss_function(d_out, all_labels)
            loss_d.backward()
            optimizer_d.step()

            latent = torch.randn((current_bs, 100), device=device)
            generator.zero_grad()
            generated = generator(latent)
            d_out_fake = discriminator(generated)
            loss_g = loss_function(d_out_fake, real_labels)
            loss_g.backward()
            optimizer_g.step()

            # bug-fix 2: log once per epoch, not "once per batch_size batches".
            if show_progress and n == batches_per_epoch - 1:
                title = (
                    f"Generated images\nEpoch: {epoch} "
                    f"Loss D.: {loss_d.item():.2f} Loss G.: {loss_g.item():.2f}"
                )
                samples = generated.detach().cpu().numpy()
                fig = plt.figure()
                for i in range(16):
                    sub = fig.add_subplot(4, 4, 1 + i)
                    sub.imshow(samples[i].reshape(28, 28), cmap="gray_r")
                    sub.axis("off")
                fig.suptitle(title)
                fig.tight_layout()
                clear_output(wait=False)
                display(fig)

    return generator, discriminator
