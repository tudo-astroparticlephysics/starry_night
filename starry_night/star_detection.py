import numpy as np


def calculate_star_visibility(kernel_response, vmag, llim, ulim):
    '''
    calculate visibility percentage
    if response > visibleUpperLimit -> visible=1
    if response < visibleUpperLimit -> visible=0
    if in between: scale linear
    '''

    lower_limit = vmag * llim[0] + llim[1]
    upper_limit = vmag * ulim[0] + ulim[1]
    visibility = (np.log10(kernel_response) - lower_limit) / (upper_limit - lower_limit)

    return np.minimum(1, np.maximum(0, visibility))
