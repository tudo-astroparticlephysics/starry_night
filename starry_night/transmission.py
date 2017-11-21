import numpy as np


def transmission_planar(alt, a, c):
    '''
    Return atmospheric transmission of planar model
    '''
    zenith = np.pi / 2 - alt
    X = 1 / np.cos(zenith)
    return a * np.exp(-c * (X - 1))


def transmission_corrected_planar(alt, a, c):
    '''
    return atmospheric transmission of planar model with correction (Young - 1974)
    '''
    zenith = np.pi / 2 - alt
    X = 1 / np.cos(zenith) * (1 - 0.0012 * (1 / np.cos(zenith)**2 - 1))
    return a * np.exp(-c * (X - 1))


def transmission_spheric(x, a, c):
    '''
    Return atmospheric transmission of spheric model with elevated observer
    x: zenith angle in rad
    a: amplitude. So: transmission(0,a,b) = a
    '''
    yObs = 2.2
    yAtm = 9.5
    rEarth = 6371.0

    x = (np.pi / 2 - x)
    r = rEarth / yAtm
    y = yObs / yAtm

    airMass = np.sqrt((r + y)**2 * np.cos(x)**2 + 2 * r * (1 - y) - y**2 + 1.0) - (r + y) * np.cos(x)
    airM_0 = np.sqrt((r + y)**2 + 2 * r * (1 - y) - y**2 + 1.0) - (r + y)
    # This model does not return 1.0 for zenith angle so we subtract airM_0 instead in the end instead of 1
    return a * np.exp(-c * (airMass - airM_0))
