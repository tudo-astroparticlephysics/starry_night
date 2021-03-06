import numpy as np
import astropy.convolution
import skimage.filters
import warnings
from functools import partial


convolve = partial(
    astropy.convolution.convolve,
    nan_treatment='fill',
    normalize_kernel=False,
)


convolve_fft = partial(
    astropy.convolution.convolve,
    nan_treatment='fill',
    normalize_kernel=False,
)


def LoG(x, y, sigma):
    '''
    Return discretized Laplacian of Gaussian kernel.
    Mean = 0 normalized and scale invarian by multiplying with sigma**2
    '''
    r2 = x**2 + y**2
    kernel = (
        1 / (np.pi * sigma**4)
        * (1 - r2 / (2 * sigma**2))
        * np.exp(-r2 / (2 * sigma**2))
    )
    kernel -= np.mean(kernel)
    return kernel * sigma**2


def create_log_kernel(kernel_size):
    lower = np.floor(-3 * kernel_size)
    upper = np.ceil(3 * kernel_size)
    vals = np.arange(lower, upper + 1)
    x, y = np.meshgrid(vals, vals)
    return LoG(x, y, kernel_size)


def apply_log_kernel(image, kernel_size):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return convolve_fft(image, create_log_kernel(kernel_size))


def apply_sobel_kernel(image):
    kernel1 = [[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]]
    kernel2 = [[1, 0, -1],
               [2, 0, -2],
               [1, 0, -1]]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        result = convolve(image, kernel1)**2
        result += convolve(image, kernel2)**2

    return result


def apply_gradient(image):
    resp = (image - np.roll(image, 1, axis=0)).clip(min=0)**2
    resp += (image - np.roll(image, 1, axis=1)).clip(min=0)**2
    return resp


def apply_difference_of_gaussians(image, kernel_size, ratio=1.6):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        resp = skimage.filters.gaussian(image, sigma=kernel_size)
        resp -= skimage.filters.gaussian(image, sigma=ratio * kernel_size)
    return resp
