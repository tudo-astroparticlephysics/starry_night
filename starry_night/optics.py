import numpy as np


def theta2r(theta, radius, how='lin'):
    '''
    convert angle to the optical axis into pixel distance to the camera
    center

    assumes linear angle projection function or equisolid angle projection function (Sigma 4.5mm f3.5)
    '''
    if how == 'lin':
        return radius / (np.pi / 2) * theta
    else:
        return 2 / np.sqrt(2) * radius * np.sin(theta / 2)


def r2theta(r, radius, how='lin', mask=False):
    '''
    convert angle to the optical axis into pixel distance to the camera
    center

    assumes linear angle projection function or equisolid angle projection function (Sigma 4.5mm f3.5)

    Returns: -converted coords,
             -mask with valid values
    '''
    if how == 'lin':
        return r / radius * (np.pi / 2)
    else:
        if mask:
            return np.arcsin(r / (2 / np.sqrt(2)) / radius) * 2, r / (2 / np.sqrt(2)) / radius < 1
        else:
            return np.arcsin(r / (2 / np.sqrt(2)) / radius) * 2
