from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
from astropy.coordinates.angle_utilities import angular_separation
import ephem
import numpy as np

from .optics import theta2r


def degDist(ra1, ra2, dec1, dec2):
    '''
    Returns great circle distance between two points on a sphere in degree.

    Input: ra and dec in rad
    Output: Angle in degree
    '''
    return angular_separation(
        ra1 * u.rad,
        dec1 * u.rad,
        ra2 * u.rad,
        dec2 * u.rad
    ).to(u.deg).value


def obs_setup(latitude, longitude, elevation):
    ''' creates an ephem.Observer for the MAGIC Site at given date '''
    obs = ephem.Observer()
    obs.lat = latitude
    obs.lon = longitude
    obs.elevation = elevation
    obs.epoch = ephem.J2000
    return obs


def eq2ho(ra, dec, prop, time):
    loc = EarthLocation.from_geodetic(
        lat=float(prop['latitude'])*u.deg,
        lon=float(prop['longitude'])*u.deg,
        height=float(prop['elevation'])*u.m
    )
    c = SkyCoord(ra=ra * u.radian, dec=dec * u.radian, frame='icrs', location=loc, obstime=time).transform_to('altaz').altaz
    return c.az.rad, c.alt.rad


def ho2eq(az, alt, prop, time):
    loc = EarthLocation.from_geodetic(
        lat=float(prop['latitude'])*u.deg,
        lon=float(prop['longitude'])*u.deg,
        height=float(prop['elevation'])*u.m
    )
    c = SkyCoord(az=az * u.radian, alt=alt * u.radian, location=loc, frame='altaz', obstime=time).transform_to('icrs')
    return c.ra.rad, c.dec.rad


def equatorial2horizontal(ra, dec, observer):
    '''
    Transforms from right ascension, declination to azimuth, altitude for
    the given observer.
    Formulas are taken from https://goo.gl/1wMU4u

    Parameters
    ----------

    ra : number or array-like
        right ascension in radians of the object of interest

    dec : number or array-like
        declination in radians of the object of interest

    observer : ephem.Observer
        the oberserver for which azimuth and altitude are calculated

    Returns
    -------
    az : number or numpy.ndarray
        azimuth in radians for the given ra, dec
    alt : number or numpy.ndarray
        altitude in radians for the given ra, dec
    '''

    obs_lat = float(observer.lat)

    h = observer.sidereal_time() - ra
    alt = np.arcsin(np.sin(obs_lat) * np.sin(dec) + np.cos(obs_lat) * np.cos(dec) * np.cos(h))
    az = np.arctan2(np.sin(h), np.cos(h) * np.sin(obs_lat) - np.tan(dec) * np.cos(obs_lat))

    # correction for camera orientation
    az = np.mod(az + np.pi, 2 * np.pi)
    return az, alt


def horizontal2image(az, alt, cam):
    '''
    convert azimuth and altitude to pixel_x, pixel_y

    Parameters
    ----------
    az : float or array-like
        the azimuth angle in radians
    alt : float or array-like
        the altitude angle in radians
    cam: dictionary
        contains zenith position, radius

    Returns
    -------
    pixel_x : number or array-like
        x cordinate in pixels for the given az, alt
    pixel_y : number or array-like
        y cordinate in pixels for the given az, alt
    '''

    radius = theta2r(
        np.pi / 2 - alt,
        np.float(cam['radius']),
        how=cam['angleprojection']
    )
    phi = az + np.deg2rad(np.float(cam['azimuthoffset']))

    x = np.float(cam['zenith_x']) + radius * np.cos(phi)
    y = np.float(cam['zenith_y']) - radius * np.sin(phi)

    return x, y


