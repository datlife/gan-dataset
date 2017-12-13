import numpy as np
import matplotlib.pyplot as plt


def imshow(images, title=None):
    """
    Image show for Tensor.
    """
    images  = images.numpy().transpose((1, 2, 0))
    mean    = np.array([0.485, 0.456, 0.406])
    std     = np.array([0.229, 0.224, 0.225])
    images  = std * images + mean
    images  = np.clip(images, 0, 1)

    plt.imshow(images)
    if title is not None:
        plt.title(title)

    plt.pause(0.001)  # pause a bit so that plots are updated
