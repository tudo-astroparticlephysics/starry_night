import numpy as np
from numpy import sin, cos, tan, arctan2, arcsin, pi
import matplotlib.pyplot as plt

import pandas as pd
import ephem

from scipy.io import matlab
from skimage.io import imread
from skimage.color import rgb2gray

from astropy.io import fits
from datetime import datetime
from time import sleep

from configparser import RawConfigParser
from pkg_resources import resource_string
from os.path import join
import requests
import logging

import re
from io import BytesIO


def downloadImg(url, *args, **kwargs):
    return rgb2gray(imread(url, ))


def get_last_modified(url, *args, **kwargs):
    ret = requests.head(url, *args, **kwargs)
    date = datetime.strptime(
        ret.headers['Last-Modified'],
        '%a, %d %b %Y %H:%M:%S GMT'
    )
    return date


def run():
    log = logging.getLogger(__name__)
    wait = 120  #wait 120 seconds between downloads
    old_date = datetime(2000, 1, 1)
    while True:
        # allsky images are only taken during night time
        if True: #datetime.utcnow().hour > 17 or datetime.utcnow().hour < 9:
            log.info('Downloading image')
            try:
                date = get_last_modified(url, timeout=5)
            except (KeyboardInterrupt, SystemExit):
                exit(0)
            except Exception as e:
                log.error(
                    'Fetching Last-Modified failed with error:\n\t{}'.format(e)
                )
                sleep(10)
                continue

            if date > old_date:
                log.info('Found new file, downloading')
                try:
                    log.debug('debug test')
                    old_date = date
                    #downloading an image may take some time. So try next download right after the first one
                    continue 
                except (KeyboardInterrupt, SystemExit):
                    exit(0)
                except Exception as e:
                    print('Download failed with error: \n\t{}'.format(e))
                    log.error('Download failed with error: \n\t{}'.format(e))
                    sleep(10)
                    continue
            else:
                log.info('No new image found')
        else:
                log.info('Daytime - no download')

        sleep(wait)

def theta2r(theta, radius):
    '''
    convert angle to the optical axis to distance to the camera
    center in pixels

    assumes equisolid angle projection function (Sigma 4.5mm f3.5)
    '''

    return 2/np.sqrt(2) * radius * np.sin(theta / 2)


def horizontal2image(az, alt, radius, zenith_x, zenith_y):
    '''
    convert azimuth and altitude to pixel_x, pixel_y

    Parameters
    ----------
    az : float or array-like
        the azimuth angle in radians
    alt : float or array-like
        the altitude angle in radians
    radius : float
        distance from zenith to horizon in pixels
    zenith_x : float
        x coordinate of the zenith in pixels
    zenith_y : float
        y  coordinate of the zenith in pixels

    Returns
    -------
    pixel_x : number or array-like
        x cordinate in pixels for the given az, alt
    pixel_y : number or array-like
        y cordinate in pixels for the given az, alt
    '''

    x = zenith_x + theta2r(np.pi/2 - alt, radius) * np.cos(az)
    y = zenith_y + theta2r(np.pi/2 - alt, radius) * np.sin(az)
    return x, y


def obs_setup(date):
    ''' creates an ephem.Observer for the MAGIC Site at given date '''
    obs = ephem.Observer()
    obs.lon = '-17:53:28'
    obs.lat = '28:45:42'
    obs.elevation = 2200
    obs.date = date
    obs.epoch = ephem.J2000
    return obs


def equatorial2horizontal(ra, dec, observer, rotation=0):
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
    alt = arcsin(sin(obs_lat) * sin(dec) + cos(obs_lat) * cos(dec) * cos(h))
    az = arctan2(sin(h), cos(h) * sin(obs_lat) - tan(dec)*cos(obs_lat))

    # correction for camera orientation
    az = np.mod(az + rotation, 2 * pi)
    return az, alt




def star_dataframe(observer, rotation, altitude=20, vmag=6):
    '''
    Read in the star catalog, add the planets from ephem and calculate
    horizontal coordinates for the stars.
    Remove stars that do not fulfill the requirements.
    '''
    #open(resource_filename('data', 'data.txt'), 'rb')
    print(resource_string('data', 'starCatalogue.csv'))
    stars = pd.read_csv(
        'asu.tsv',
        #'./hipparcos_vmag10.csv',
        sep=';',
        comment='#',
        skipinitialspace=True,
    )

    # transform degrees to radians
    stars.ra = np.deg2rad(stars.ra)
    stars.dec = np.deg2rad(stars.dec)

    # add the planets
    sol_objects = [
        ephem.Mercury(),
        ephem.Venus(),
        ephem.Mars(),
        ephem.Jupiter(),
        ephem.Saturn(),
        ephem.Uranus(),
        ephem.Neptune(),
    ]
    for sol_object in sol_objects:
        sol_object.compute(observer)
        equatorial = ephem.Equatorial(sol_object.g_ra, sol_object.g_dec, epoch=ephem.J2000)
        galactic = ephem.Galactic(equatorial)
        data = {
            'ra': float(sol_object.a_ra),
            'dec': float(sol_object.a_dec),
            'gLon': float(galactic.lon)/np.pi*180,
            'gLat': float(galactic.lat)/np.pi*180,
            'vmag': float(sol_object.mag),
        }
        stars = stars.append(data, ignore_index=True)

    stars['azimuth'], stars['altitude'] = equatorial2horizontal(
        stars.ra, stars.dec, observer, rotation=rotation,
    )

    # remove stars that are not within the limits
    stars = stars.query('altitude > {}'.format(np.deg2rad(altitude)))
    stars = stars.query('vmag < {}'.format(vmag))

    # include moon data
    moon = ephem.Moon()
    moon.compute(observer)
    moonData = {
        'date' : str(observer.date),
        'moonPhase' : moon.moon_phase,
        'moonZenith' : 90 - moon.alt,
        'moonAz' : moon.az,
    }

    # calculate angle to moon
    stars['angleToMoon'] = stars.apply(lambda x : np.arccos(np.sin(x.altitude)*
        np.sin(moon.alt) + np.cos(x.altitude)*np.cos(moon.alt)*
        np.cos((x.azimuth - moon.az)) )/np.pi*180, axis=1)
    return stars, moonData

def findLocalMaxX(img, x, y, distance):
    '''
    ' return x position of brightest pixel within distance
    '''
    maxPos = np.argmax(img[x-distance:x+distance+1, y-distance:y+distance+1])
    xDelta = maxPos%(2*distance+1)-distance
    return int(x+xDelta)

def findLocalMaxY(img, x, y, distance):
    '''
    ' return x position of brightest pixel within distance
    '''
    maxPos = np.argmax(img[x-distance:x+distance+1, y-distance:y+distance+1])
    yDelta = maxPos//(2*distance+1)-distance
    return int(y+yDelta)

def findLocalMaxValue(img, xArr, yArr, distance):
    out = list()
    for x,y in zip(xArr,yArr):
        out.append(np.amax(img[x-distance:x+distance+1, y-distance:y+distance+1]))
    return out


def loadCamImage(filename):
    '''Open an image file and return its content as a numpy array'''
    if filename.endswith('.mat'):
        data = matlab.loadmat(filename)
        return data[dictEntry]
    elif filename.endswith('.fits'):
        hdulist = fits.open(filename)
        return hdulist[0].data
    else:
        return misc.imread(filename, mode='L')
    
def loadImageTime(filename):
    # assuming that the filename only contains numbers of timestamp
    timestamp = re.findall('\d{2,}', filename)
    timestamp = list(map(int, timestamp))

    return datetime(*timestamp)

# display fits image on screen
def dispFits(image):
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.95, 0.95])
    vmin = np.nanpercentile(image, 0.5)
    vmax = np.nanpercentile(image, 99.5)
    image = (image - vmin)*(1000./(vmax-vmin))
    vmin = np.nanpercentile(image, 0.5)
    vmax = np.nanpercentile(image, 99.5)
    ax.imshow(image, vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
    plt.show()

def dispHist(image):
    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.05, 0.95, 0.95])
    '''
    image = (image - vmin)*(1000./(vmax-vmin))
    vmin = np.nanpercentile(image, 0.5)
    vmax = np.nanpercentile(image, 99.5)
    '''
    plt.hist(image[~np.isnan(image)].ravel(), bins=100, range=(-150,2000))
    plt.show()
