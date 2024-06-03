import numpy as np
import matplotlib.pyplot as plt


def normalize(X):
    """Normalize to [0,1]

    Args:
        X (list,number): the input to normalize

    Returns:
        list,array,int: normalized input between 0. and 1.
    """
    if isinstance(X, list):
        return [x / 255.0 for x in X]
    else:
        return X / 255.0


def get_image_mean(images, title, image_shape, display=False):
    # calculate the average
    mean_img = np.mean(images, axis=0)
    # reshape it back to a matrix
    mean_img = mean_img.reshape(image_shape)
    if display:
        plt.imshow(mean_img, vmin=0, vmax=255, cmap="Greys_r")
        plt.title(f"Average {title}")
        plt.axis("off")
        plt.show()
    return mean_img


if __name__ == "__main__":
    pass
