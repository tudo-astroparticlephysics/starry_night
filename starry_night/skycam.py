import numpy as np
from numpy import sin, cos, tan, arctan2, arcsin, pi
import matplotlib.pyplot as plt

import pandas as pd
import ephem
import sys

from scipy.io import matlab
from skimage.io import imread
from skimage.color import rgb2gray

from astropy.io import fits
from datetime import datetime
from time import sleep

from configparser import RawConfigParser
from pkg_resources import resource_filename
from os.path import join
import requests
import logging

import re
from io import BytesIO
from IPython import embed


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
                    log.error('Download failed with error: \n\t{}'.format(e))
                    sleep(10)
                    continue
            else:
                log.info('No new image found')
        else:
                log.info('Daytime - no download')

        sleep(wait)

def theta2r(theta, radius, how='lin'):
    '''
    convert angle to the optical axis into pixel distance to the camera
    center

    assumes linear angle projection function or equisolid angle projection function (Sigma 4.5mm f3.5)
    '''
    if how == 'lin':
        return radius / (np.pi/2) * theta
    else:
        return 2/np.sqrt(2) * radius * np.sin(theta/2)


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

    try:
        x = np.float(cam['zenith_x']) + theta2r(np.pi/2 - alt,
                np.float(cam['radius']),
                how=cam['angleprojection']
                ) * np.cos(az+np.float(cam['azimuthoffset']))
        y = np.float(cam['zenith_y']) - theta2r(np.pi/2 - alt,
                np.float(cam['radius']),
                how=cam['angleprojection']
                ) * np.sin(az+np.float(cam['azimuthoffset']))
    except:
        raise
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
    alt = arcsin(sin(obs_lat) * sin(dec) + cos(obs_lat) * cos(dec) * cos(h))
    az = arctan2(sin(h), cos(h) * sin(obs_lat) - tan(dec)*cos(obs_lat))

    # correction for camera orientation
    az = np.mod(az+pi, 2*pi)
    return az, alt




def star_planets_moon_sun_dataframes(observer, cam):
    '''
    Read in the star catalog, add the planets from ephem and calculate
    horizontal coordinates for the stars.
    Remove stars that do not fulfill the requirements.
    '''
    log = logging.getLogger(__name__)
    
    log.debug('Loading stars')
    catalogue = resource_filename('starry_night', '../data/asu.tsv')
    stars = pd.read_csv(
        catalogue,
        sep=';',
        comment='#',
        header=0,
        skipinitialspace=False,
        index_col=4,
    )
    stars = stars.convert_objects(convert_numeric=True)

    # transform degrees to radians
    stars.ra = np.deg2rad(stars.ra)
    stars.dec = np.deg2rad(stars.dec)

    log.debug('Loading planets')
    planets = pd.DataFrame()
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
            'name': sol_object.name,
        }
        planets = planets.append(data, ignore_index=True)
    planets.set_index('name', inplace=True)

    stars['azimuth'], stars['altitude'] = equatorial2horizontal(
        stars.ra, stars.dec, observer,
    )
    planets['azimuth'], planets['altitude'] = equatorial2horizontal(
        planets.ra, planets.dec, observer,
    )

    # remove stars and planets that are not within the limits
    try:
        stars = stars.query('altitude > {}'.format(np.deg2rad(90 - float(cam['openingangle']))))
        planets = planets.query('altitude > {}'.format(np.deg2rad(90 - float(cam['openingangle']))))
        stars = stars.query('vmag < {}'.format(cam['vmaglimit']))
        planets = planets.query('vmag < {}'.format(cam['vmaglimit']))
    except:
        log.error('Using altitude or vmag limit failed!')
        raise

    # include moon data
    log.debug('Loading moon')
    moon = ephem.Moon()
    moon.compute(observer)
    moonData = {
        'moonPhase' : moon.moon_phase,
        'altitude' : np.deg2rad(moon.alt),
        'azimuth' : np.deg2rad(moon.az),
    }

    # calculate angle to moon
    stars['angleToMoon'] = stars.apply(lambda x : np.arccos(np.sin(x.altitude)*
        np.sin(moon.alt) + np.cos(x.altitude)*np.cos(moon.alt)*
        np.cos((x.azimuth - moon.az)) )/np.pi*180, axis=1)
    if not planets.empty:
        planets['angleToMoon'] = planets.apply(lambda x : np.arccos(np.sin(x.altitude)*
            np.sin(moon.alt) + np.cos(x.altitude)*np.cos(moon.alt)*
            np.cos((x.azimuth - moon.az)) )/np.pi*180, axis=1)


    sun = ephem.Sun()
    sun.compute(observer)
    sunData = {
        'altitude' : np.deg2rad(sun.alt),
        'azimuth' : np.deg2rad(sun.az),
    }
    return stars, planets, moonData, sunData

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


def loadImageAndTime(filename, crop=None, fmt=None):
    '''
    Open an image file and return its content as a numpy array.
    
    input:
        filename: full or relativ path to image
        crop: crop image to a circle with center and radius
        fmt: format timestring like 'gtc_allskyimage_%Y%m%d_%H%M%S.jpg'
            used for parsing the date from filename
    '''
    log = logging.getLogger(__name__)
    #TODO: read image time from mat and fits file
    if filename.endswith('.mat'):
        data = matlab.loadmat(filename)
        img = data[dictEntry]
        if fmt is None:
            time = datetime.strptime(filename, fmt)
    elif filename.endswith('.fits'):
        hdulist = fits.open(filename)
        img = hdulist[0].data
        if fmt is None:
            time = datetime.strptime(filename, fmt)
    else:
        try:
            img = imread(filename, mode='L', as_grey=True)
            time = datetime.strptime(filename, fmt)
        except (FileNotFoundError, OSError):
            log.error('File {} not found. Or filetype invalid'.format(filename))
            raise
        except ValueError:
            log.error('Filename {} does not match {}'.format(filename, fmt))
            raise
    if crop is not None:
        x, y, r =  map(int(), crop['crop_x'].split(',')), crop['crop_y'], crop['crop_radius']
        print(x)
        #re.split('\\s*,\\s*', crop['crop_x']))
        nrows, ncols = img.shape
        row, col = np.ogrid[:nrows, :ncols]
        outer_disk_mask = ((row - y)**2 + (col - x)**2 > r**2)
        img[outer_disk_mask] = 0
    return img, time

    
    
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
