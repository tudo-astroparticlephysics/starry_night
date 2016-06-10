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
import skimage.filters
from scipy.ndimage.measurements import label
from dask import delayed
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
    Read the given star catalog, add planets from ephem and calculate
    horizontal coordinates for all celestial object.
    Remove objects that do not fulfill the needed requirements.
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

    # add the planets
    log.debug('Loading planets')
    sol_objects = [
        ephem.Mercury(),
        ephem.Venus(),
        ephem.Mars(),
        ephem.Jupiter(),
        ephem.Saturn(),
        ephem.Uranus(),
        ephem.Neptune(),
    ]
    planets = pd.DataFrame()
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
    log.debug('Calculate Angle to Moon')
    stars['angleToMoon'] = stars.apply(lambda x : np.arccos(np.sin(x.altitude)*
        np.sin(moon.alt) + np.cos(x.altitude)*np.cos(moon.alt)*
        np.cos((x.azimuth - moon.az)) )/np.pi*180, axis=1)
    if not planets.empty:
        planets['angleToMoon'] = planets.apply(lambda x : np.arccos(np.sin(x.altitude)*
            np.sin(moon.alt) + np.cos(x.altitude)*np.cos(moon.alt)*
            np.cos((x.azimuth - moon.az)) )/np.pi*180, axis=1)

    log.debug('Load Sun')
    sun = ephem.Sun()
    sun.compute(observer)
    sunData = {
        'altitude' : np.deg2rad(sun.alt),
        'azimuth' : np.deg2rad(sun.az),
    }
    return stars, planets, moonData, sunData

def findLocalMaxValue(img, x, y, radius):
    '''
    ' Returns value of brightest pixel within radius
    '''
    try:
        x = int(x)
        y = int(y)
    except:
        x = x.astype(int)
        y = y.astype(int)
    
    # get interval border
    x_interval = np.max([x-radius,0]) , np.min([x+radius+1, img.shape[1]])
    y_interval = np.max([y-radius,0]) , np.min([y+radius+1, img.shape[0]])
    radius = x_interval[1]-x_interval[0] , y_interval[1]-y_interval[0]

    # do subselection
    subImg = img[y_interval[0]:y_interval[1] , x_interval[0]:x_interval[1]]
    try:
        return np.nanmax(subImg.flatten())
    except RuntimeWarning:
        print('NAN')
        return 0
    except ValueError:
        print('Star outside image')
        return 0

def findLocalMaxPos(img, x, y, radius):
    '''
    ' Returns x and y position of brightest pixel within radius
    ' If all pixel have equal brightness, current position is returned
    '''
    try:
        x = int(x)
        y = int(y)
    except:
        x = x.astype(int)
        y = y.astype(int)
    # get interval border
    x_interval = np.max([x-radius,0]) , np.min([x+radius+1, img.shape[1]])
    y_interval = np.max([y-radius,0]) , np.min([y+radius+1, img.shape[0]])
    radius = x_interval[1]-x_interval[0] , y_interval[1]-y_interval[0]
    subImg = img[y_interval[0]:y_interval[1] , x_interval[0]:x_interval[1]]
    if np.max(subImg) != np.min(subImg):
        try:
            maxPos = np.nanargmax(subImg)
            x = (maxPos%radius[0])+x_interval[0]
            y = (maxPos//radius[0])+y_interval[0]
        except ValueError:
            return pd.Series({'maxX':0, 'maxY':0})
    return pd.Series({'maxX':int(x), 'maxY':int(y)})


@delayed(pure=True)
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
            sys.exit(1)
        except ValueError:
            log.error('Filename {} does not match {}'.format(filename, fmt))
            sys.exit(1)
    return (img, time)

    
def get_crop_mask(img, crop):
    '''
    crop is dictionary with cropping information
    returns a boolean array in size of img: False got cropped; True not cropped 
    '''
    if crop is not None:
        try:
            x = re.split('\\s*,\\s*', crop['crop_x'])
            y = re.split('\\s*,\\s*', crop['crop_y'])
            r = re.split('\\s*,\\s*', crop['crop_radius'])
            inside = re.split('\\s*,\\s*', crop['crop_deleteinside'])
            nrows, ncols = img.shape
            row, col = np.ogrid[:nrows, :ncols]
            disk_mask = np.full((nrows, ncols), False, dtype=bool)
            for x,y,r,inside in zip(x,y,r,inside):
                if inside == '0':
                    disk_mask = disk_mask | ((row - int(y))**2 + (col - int(x))**2 > int(r)**2)
                else:
                    disk_mask = disk_mask | ((row - int(y))**2 + (col - int(x))**2 < int(r)**2)
        except:
            log.error('Cropping failed, maybe there is a typing error in the config file?')
            raise
        return disk_mask
    
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

def isInRange(position, stars, rng):
    if rng < 0:
        raise ValueError
    '''
    if 'x' in position.keys():
        return ((position.x - star.x)**2 + (position.y - star.y)**2 <= rng**2)
    else:
    '''
    dec1 = position['dec']
    dec2 = stars['dec'].values
    ra1 = position['ra']/12*np.pi
    ra2 = stars['ra'].values/12*np.pi

    deltaDeg = 2*np.arcsin(np.sqrt(np.sin((dec1-dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin((ra1-ra2)/2)**2))

    return deltaDeg <= np.deg2rad(rng)


def calc_star_percentage(position, stars, rng, weight=False):
    '''
    Returns percentage of visible stars that are within range of position
    
    Position is dictionary and can contain Ra,Dec or x,y
    Range is degree or pixel radius depending on whether horizontal or pixel coordinates were used
    '''
    
    if rng < 0:
        starsInRange = stars
    else:
        starsInRange = stars[isInRange(position, stars, rng)]

    try:
        if weight:
            vis = np.sum(np.pow(100**(1/5),starsInRange.query('visible').vmag.values))
            notVis = np.sum(np.pow(100**(1/5),starsInRange.query('~visible').vmag.values))
            percentage = vis/(vis+notVis)
        else:
            percentage = len(starsInRange.query('visible').index)/len(starsInRange.index)
    except ZeroDivisionError:
        log = logging.getLogger(__name__)
        log.warning('No stars in range to calc percentage. Returning -1.')
        percentage = -1

    return percentage

def filter_catalogue(catalogue, rng):
    '''
    Loop through all possible pairs of stars and remove less bright star if distance is < rng

    Input: Pandas DataFrame and a distance in degree

    Returns: List of index that got not removed
    '''
    log = logging.getLogger(__name__)
    try:
        c = catalogue.sort_values('vmag', ascending=True)
        reference_list = list(c[['ra','dec']].values)
        filtered_list = list(c.index)
    except KeyError:
        log.error('Key not found. Please check that your catalogue is labeled correctly')
        raise
    
    i1 = 0
    while i1 < len(reference_list)-1:
        log.debug('Items left: {}/{}'.format(i1,len(reference_list)-1))
        row1 = reference_list[i1]
        pop_index = i1 + 1
        i2 = i1 +1
        while i2 < len(reference_list):
            row2 = reference_list[i2]
            deltaDeg = np.rad2deg(2*np.arcsin(np.sqrt(np.sin((row1[1]-row2[1])/2)**2 + np.cos(row1[1])*np.cos(row2[1])*np.sin((row1[0]-row2[0])/2)**2)))
            if deltaDeg < rng:
                filtered_list.pop(pop_index)
                reference_list.pop(pop_index)
            else:
                pop_index += 1
                i2 += 1
        i1 += 1
    return filtered_list

@delayed(pure=True)
def process_image(images, config):
    log = logging.getLogger(__name__)
    img = images['img']

    # create cropping array to mask unneccessary image regions.
    crop_mask = get_crop_mask(img, config['crop'])

    log.debug('Image time: {}'.format(images['timestamp']))

    log.debug('Creating Observer')
    obs = obs_setup(images['timestamp'])
    log.debug('Parsing Catalogue')
    stars, planets, moon, sun = star_planets_moon_sun_dataframes(
            obs, 
            cam=config['image'],
            )

    #log.info('Filtering catalogue')
    #rem = filter_catalogue(stars, rng = float(config['image']['minAngleBetweenStars']))

    # calculate x and y position
    log.debug('Calculate x and y')
    stars['x'], stars['y'] = horizontal2image(stars.azimuth, stars.altitude, cam=config['image'])
    planets['x'], planets['y'] = horizontal2image(planets.azimuth, planets.altitude, cam=config['image'])
    moon['x'], moon['y'] = horizontal2image(moon['azimuth'], moon['altitude'], cam=config['image'])


    log.debug('Apply image filters')
    grad = (img - np.roll(img, 1, axis=0)).clip(min=0)**2 + (img - np.roll(img, 1, axis=1)).clip(min=0)**2
    sobel = skimage.filters.sobel(img).clip(min=0)
    gauss = skimage.filters.gaussian(img, sigma=1)
    lap = skimage.filters.laplace(gauss, ksize=3).clip(min=0)
    grad[crop_mask] = 0
    sobel[crop_mask] = 0
    lap[crop_mask] = 0

    images['grad'] = grad
    images['sobel'] = sobel
    images['lap'] = lap

    log.debug('Calculate Filter response')
    stars['response'] = stars.apply(lambda s : findLocalMaxValue(grad, s.x, s.y, 2), axis=1)
    stars['response2'] = stars.apply(lambda s : findLocalMaxValue(sobel, s.x, s.y, 2), axis=1)
    stars['response3'] = stars.apply(lambda s : findLocalMaxValue(lap, s.x, s.y, 2), axis=1)

    return stars
