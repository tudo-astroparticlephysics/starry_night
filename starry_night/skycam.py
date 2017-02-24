from starry_night import sql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc, cm, gridspec, ticker
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import ephem
import sys
from time import sleep

from astropy.io import fits
from astropy.time import Time
from astropy.convolution import convolve, convolve_fft
from astropy import units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from scipy.io import matlab
from scipy.ndimage.measurements import label
from scipy.optimize import curve_fit
from io import BytesIO
from skimage.io import imread
from skimage.color import rgb2gray
import skimage.filters
from os import stat
import warnings

from datetime import datetime, timedelta

from pkg_resources import resource_filename
from os.path import join
import requests
import logging

from re import split,findall
from hashlib import sha1
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, InternalError
import requests.exceptions as rex

from IPython import embed


def degDist(ra1, ra2, dec1, dec2):
    '''
    Returns great circle distance between two points on a sphere in degree.
    Using haversine formula.

    Input: ra and dec in rad
    Output: Angle in degree
    '''
    return np.rad2deg(2*np.arcsin(np.sqrt(np.sin((ra1-ra2)/2)**2 +
        np.cos(ra1)*np.cos(ra2)*np.sin((dec1-dec2)/2)**2)))

def LoG(x,y,sigma):
    '''
    Return discretized Laplacian of Gaussian kernel.
    Mean = 0 normalized and scale invarian by multiplying with sigma**2
    '''
    kernel = 1/(np.pi*sigma**4)*(1-(x**2+y**2)/(2*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))
    kernel -= np.mean(kernel)
    return kernel * sigma**2

def lin(x,m,b):
    'Just a linear function'
    return m*x+b

def expo(x,m,b):
    return np.exp(m*x+b)

def transmission(x, a, c):
    '''
    Return atmospheric transmission of planar model
    '''
    x = np.pi/2 -x
    return a*np.exp(-c * (1/np.cos(x) - 1))

def transmission2(x, a, c):
    '''
    return atmospheric transmission of planar model with correction (Young - 1974)
    '''
    x = np.pi/2 -x
    return a * np.exp(-c * (1/np.cos(x)*(1-0.0012*(1/np.cos(x)**2 - 1)) - 1))

def transmission3(x, a, c):
    '''
    Return atmospheric transmission of spheric model with elevated observer
    x: zenith angle in rad
    a: amplitude. So: transmission(0,a,b) = a 
    '''
    yObs=2.2
    yAtm=9.5
    rEarth=6371.0

    x = (np.pi/2 -x)
    r = rEarth / yAtm
    y = yObs / yAtm

    airMass = np.sqrt( ( r + y )**2 * np.cos(x)**2 + 2.*r*(1.-y) - y**2 + 1.0 ) - (r+y)*np.cos(x)
    airM_0 = np.sqrt( ( r + y )**2 + 2.*r*(1.-y) - y**2 + 1.0 ) - (r+y)
    #This model does not return 1.0 for zenith angle so we subtract airM_0 instead in the end instead of 1
    return a* np.exp(-c * (airMass - airM_0 ))


class TooEarlyError(Exception):
    pass

def get_last_modified(url, timeout):
    try:
        ret = requests.head(url, timeout=timeout)
        date = datetime.strptime(
            ret.headers['Last-Modified'],
            '%a, %d %b %Y %H:%M:%S GMT'
        )
    except (rex.ReadTimeout, KeyError):
        log = logging.getLogger(__name__)
        log.error('Failed to retrieve timestamp from {} because website can not be reached.\nRetry later...'.format(url))
        date = None
    return date

def getMagicLidar(passwd):
    '''
    Return dict with data of the Magic lidar on LaPalma.
    passwd is the FACT password to access the data
    '''
    log = logging.getLogger(__name__)
    try:
        response = requests.get('http://www.magic.iac.es/site/weather/protected/lidar_data.txt',
                auth=requests.auth.HTTPBasicAuth('FACT', passwd)
            )
    except rex.ConnectionError as e:
        log.error('Connecting to lidar failed {}'.format(e))
        return
    if response.ok:
        dataString = response.content.decode('utf-8')
    else:
        log.error('Wrong lidar password')
        return
    values = list(map(float, findall("\d+\.\d+|\d+", dataString)))
    timestamp = datetime(*list(map(int,values[-3:])), *list(map(int,values[-6:-3])))

    # abort if last lidar update was more than 15min ago
    if datetime.utcnow() - timestamp > timedelta(minutes=15):
        return
    else:
        return {'timestamp': timestamp, 'altitude': (90-values[0])/180*np.pi, 'azimuth': values[1]/180*np.pi, 'T3':values[3], 'T6':values[5], 'T9':values[7], 'T12':values[9]}

def downloadImg(url, timeout=None):
    '''
    Download image from URL and return a dict with 'img' and 'timestamp'

    Download will only happen, if the website was updated since the last download AND the SHA1
    hashsum differs from the previous image because sometime a website might refresh without
    updating the image.
    Works with fits, mat and all common image filetypes.
    '''
    log = logging.getLogger(__name__)
    if not hasattr(downloadImg, 'lastMod'):
        downloadImg.lastMod = datetime(1,1,1)
        downloadImg.hash = ''
    logging.getLogger('requests').setLevel(logging.WARNING)

    # only download if time since last image is > than wait
    mod = get_last_modified(url, timeout=timeout)
    if not mod:
        return dict()
    elif mod <= downloadImg.lastMod:
        raise TooEarlyError()
    else:
        downloadImg.lastMod = mod

    # download image data and double check if this really is a new image
    log.info('Downloading image from {}'.format(url))
    ret = requests.get(url, timeout=timeout)
    if downloadImg.hash == sha1(ret.content).hexdigest():
        raise TooEarlyError()
    else:
        downloadImg.hash = sha1(ret.content).hexdigest()
    if url.split('.')[-1] == 'mat':
        data = matlab.loadmat(BytesIO(ret.content))
        for d in list(data.values()):
            # loop through all keys and treat the first array with size > 100x100 as image
            # that way the name of the key does not matter
            try:
                if d.shape[0] > 100 and d.shape[1] > 100:
                    img = d
            except AttributeError:
                pass
            try:
                timestamp = datetime.strptime(d[0], '%Y/%m/%d %H:%M:%S')
            except (IndexError, TypeError, ValueError):
                pass
    elif url.split('.')[-1] == 'FIT':
        hdulist = fits.open(BytesIO(ret.content), ignore_missing_end=True)
        img = hdulist[0].data+2**16/2
        timestamp = datetime.strptime(
                        hdulist[0].header['UTC'],
                        '%Y/%m/%d %H:%M:%S')
    else:
        img = rgb2gray(imread(url, ))
        timestamp = get_last_modified(url, timeout=timeout)
        if not timestamp:
            return dict()
        
    return {
        'img' : img,
        'timestamp' : timestamp,
        }


def getBlobsize(img, thresh, limit=0):
    '''
    Returns size of the blob in the center of img.
    If the blob is bigger than limit, limit gets returned immideatly.

    A blob consists of all 8 neighboors that are bigger than 'thresh' and their neighboors respectively.
    '''
    if thresh <= 0:
        raise ValueError('Thresh > 0 required')
    if img.shape[0]%2 == 0 or img.shape[1]%2==0:
        raise IndexError('Only odd sized arrays are supported. Array shape:{}'.format(img.shape))
    if limit == 0:
        limit = img.shape[0]*img.shape[1]

    center = (img.shape[0]//2, img.shape[1]//2)

    # if all pixels are above threshold then return max blob size
    if thresh <= np.min(img):
        return np.minimum(limit, img.shape[0]*img.shape[1])
    
    # work on local copy
    tempImg = img.copy()
    tempImg[~np.isfinite(tempImg)] = 0

    nList = list()
    count = 0

    # fill list with pixels and count them
    nList.append(center)
    while len(nList) > 0:
        x,y = nList.pop(0)

        for i in (-1,0,1):
            for j in (-1,0,1):
                if x+i<0 or x+i>=img.shape[0] or y+j<0 or y+j>=img.shape[1]:
                    pass
                elif tempImg[x+i,y+j] >= thresh:
                    count += 1
                    tempImg[x+i,y+j] = 0
                    nList.append((x+i,y+j))
        if count >= limit:
            return limit
    return count
    

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

def r2theta(r, radius, how='lin', mask=False):
    '''
    convert angle to the optical axis into pixel distance to the camera
    center

    assumes linear angle projection function or equisolid angle projection function (Sigma 4.5mm f3.5)

    Returns: -converted coords,
             -mask with valid values
    '''
    if how == 'lin':
        return r / radius * (np.pi/2)
    else:
        if mask:
            return np.arcsin(r / (2/np.sqrt(2)) / radius) * 2, r/(2/np.sqrt(2))/radius < 1
        else:
            return np.arcsin(r / (2/np.sqrt(2)) / radius) * 2


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
                ) * np.cos(az+np.deg2rad(np.float(cam['azimuthoffset'])))
        y = np.float(cam['zenith_y']) - theta2r(np.pi/2 - alt,
                np.float(cam['radius']),
                how=cam['angleprojection']
                ) * np.sin(az+np.deg2rad(np.float(cam['azimuthoffset'])))
    except:
        raise
    return x, y

def find_matching_pos(img_timestamp, time_pos_list, conf):
    '''
    Return position of (e.g. LIDAR) in the moment the image was taken.

    Returns entry from 'time_pos_list' that has the closest timestamp to 'img_timestamp'.
    Tolerance: +10min into futur ,-1min past. Because LIDAR result is based on 10min measurement.

    Converts coordinates in horizontal coords.
    '''
    subset = time_pos_list.query('-1/24/60 * 10 < MJD - {} < 1/24/60*1'.format(Time(img_timestamp).mjd)).sort_values('MJD')
    closest = subset[subset.MJD==subset.MJD.min()]

    if closest.empty:
        return dict()
        
    # test if equatorial is NaN or undefinded
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            if any(closest['ra'] != closest['ra']) or any(closest['dec'] != closest['dec']):
                raise KeyError
        except (KeyError):
            try:
                if any (closest['azimuth'] != closest['azimuth']) or any(closest['altitude'] != closest['altitude']):
                    raise KeyError
                closest['ra'], closest['dec'] = ho2eq(
                    closest.azimuth, closest.altitude, conf['properties'], img_timestamp,
                )
            except (KeyError):
                log = logging.getLogger(__name__)
                log.error('Failed coord tranformation for find_matching_pos')
                return dict()
        else:
        # equatorial is definded
            closest['azimuth'], closest['altitude'] = eq2ho(
                closest.ra, closest.dec, conf['properties'], img_timestamp
            )
        logging.getLogger(__name__).debug('Found match')
        return closest[['azimuth','altitude','ra','dec']]


def obs_setup(properties):
    ''' creates an ephem.Observer for the MAGIC Site at given date '''
    obs = ephem.Observer()
    obs.lon = '-17:53:28'
    obs.lat = '28:45:42'
    obs.elevation = 2200
    obs.epoch = ephem.J2000
    return obs

def eq2ho(ra, dec, prop, time):
    loc = EarthLocation.from_geodetic(lat=float(prop['latitude'])*u.deg, lon=float(prop['longitude'])*u.deg, height=float(prop['elevation'])*u.m)
    c = SkyCoord(ra=ra*u.radian, dec=dec*u.radian, frame='icrs', location=loc, obstime=time).transform_to('altaz').altaz
    return c.az.rad, c.alt.rad

def ho2eq(az, alt, prop, time):
    loc = EarthLocation.from_geodetic(lat=float(prop['latitude'])*u.deg, lon=float(prop['longitude'])*u.deg, height=float(prop['elevation'])*u.m)
    c = SkyCoord(az=az*u.radian, alt=alt*u.radian, location=loc, frame='altaz', obstime=time).transform_to('icrs')
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
    az = np.arctan2(np.sin(h), np.cos(h) * np.sin(obs_lat) - np.tan(dec)*np.cos(obs_lat))

    # correction for camera orientation
    az = np.mod(az+np.pi, 2*np.pi)
    return az, alt


def celObjects_dict(config):
    '''
    Read the given star catalog, add planets from ephem and fill sun and moon with NaNs
    For horizontal coordinates 'update_star_position()' needs to be called next.

    Returns: dictionary with celestial objects
    '''
    log = logging.getLogger(__name__)
    
    log.debug('Loading stars')
    catalogue = resource_filename('starry_night', 'data/catalogue_10vmag_0.8degFilter.csv')
    try:
        stars = pd.read_csv(
            catalogue,
            sep=',',
            comment='#',
            header=0,
            skipinitialspace=False,
        )
    except OSError as e:
        log.error('Star catalogue not found: {}'.format(e))
        sys.exit(1)
    stars.set_index('HIP', drop=True)

    # transform degrees to radians
    stars.ra = np.deg2rad(stars.ra)
    stars.dec = np.deg2rad(stars.dec)

    stars['altitude'] = np.NaN
    stars['azimuth'] = np.NaN

    # add the planets
    planets = pd.DataFrame()
    for planet in ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']:
        data = {
            'ra': np.NaN,
            'dec': np.NaN,
            'altitude' : np.NaN,
            'azimuth' : np.NaN,
            'gLon': np.NaN,
            'gLat': np.NaN,
            'vmag': np.NaN,
            'name': planet,
        }
        planets = planets.append(data, ignore_index=True)

    # add points_of_interest
    log.debug('Add points of interest')
    try:
        points_of_interest = pd.read_csv(
            config['analysis']['points_of_interest'],
            sep=',',
            comment='#',
            header=0,
            skipinitialspace=False,
            index_col=None,
        )
    except OSError as e:
        log.debug('File with points of interest not found: {}. We will now check internal package files...'.format(e))
        try:
            poi_filename = resource_filename('starry_night', join('data',config['analysis']['points_of_interest']))
            points_of_interest = pd.read_csv(
                poi_filename,
                sep=',',
                comment='#',
                header=0,
                skipinitialspace=False,
                index_col=None,
            )
        except OSError as e:
            log.error('File with points of interest not found: {}'.format(e))
            sys.exit(1)
        else:
            log.debug('Found {}'.format(poi_filename))

    points_of_interest['altitude'] = np.NaN
    points_of_interest['azimuth'] = np.NaN
    points_of_interest['radius'] = float(config['analysis']['poi_radius'])
    points_of_interest['ra'] *= np.pi/180 
    points_of_interest['dec'] *= np.pi/180

    # add moon
    moonData = {
        'moonPhase' : np.NaN,
        'altitude' : np.NaN,
        'azimuth' : np.NaN,
    }
    # add sun
    sunData = {
        'altitude' : np.NaN,
        'azimuth' : np.NaN,
    }

    return dict({'stars': stars,
        'planets': planets,
        'points_of_interest' : points_of_interest,
        'sun': sunData,
        'moon': moonData,
        })


def update_star_position(data, observer, conf, crop, args):
    '''
    Takes the dictionary from 'star_planets_sun_moon_dict(observer)'
    and calculates the current position of each object in the sky
    also sets position of sun and moon (were filled with NaNs so far)
    Objects that are not within the camera limits (vmag, altitude, crop...) get removed.

    Returns: dictionary with updated positions
    '''
    log = logging.getLogger(__name__)

    # include moon data
    log.debug('Loading moon')
    moon = ephem.Moon()
    moon.compute(observer)
    moonData = {
        'moonPhase' : float(moon.moon_phase),
        'altitude' : float(moon.alt),
        'azimuth' : float(moon.az),
    }

    # include sun data
    log.debug('Load Sun')
    sun = ephem.Sun()
    sun.compute(observer)
    sunData = {
        'altitude' : float(sun.alt),
        'azimuth' : float(sun.az),
    }

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
        p = {
            'ra': float(sol_object.a_ra),
            'dec': float(sol_object.a_dec),
            'gLon': float(galactic.lon),
            'gLat': float(galactic.lat),
            'vmag': float(sol_object.mag),
            'azimuth': float(sol_object.az),
            'altitude': float(sol_object.alt),
            'name': sol_object.name,
        }
        planets = planets.append(p, ignore_index=True)
    planets.set_index('name', inplace=True)

    # make a copy here, because we will need ALL stars later again
    # append lidar position from positioning file if any
    # append Total_sky object 
    # update all objects
    # remove objects that are not within the limits
    stars = data['stars'].copy()
    points_of_interest = data['points_of_interest'].copy()

    stars['azimuth'], stars['altitude'] = equatorial2horizontal(
        stars.ra, stars.dec, observer,
    )
    points_of_interest['azimuth'], points_of_interest['altitude'] = equatorial2horizontal(
        points_of_interest.ra, points_of_interest.dec, observer,
    )

    log.debug('Find matching postition')
    if args['-p']:
        lidar_old = find_matching_pos(data['timestamp'], data['positioning_file'], conf)
        lidar_old['name'] = 'Lidar'
        lidar_old['ID'] = -2
        lidar_old['radius'] = float(conf['analysis']['poi_radius'])
        points_of_interest = points_of_interest.append(lidar_old, ignore_index=True)

    # Get the magic Lidar
    magicLidar_now = None
    if data['lidarpwd']:
        magicLidar_now = getMagicLidar(data['lidarpwd'])
        if magicLidar_now:
            magicLidar_now['name'] = 'Magic Lidar now'
            magicLidar_now['ID'] = -3
            magicLidar_now['radius'] = float(conf['analysis']['poi_radius'])
            points_of_interest = points_of_interest.append(
                    {
                        'name': magicLidar_now['name'],
                        'ID' : magicLidar_now['ID'],
                        'radius' : magicLidar_now['radius'],
                        'altitude' : magicLidar_now['altitude'],
                        'azimuth' : magicLidar_now['azimuth'],
                    },ignore_index=True
            )

    try:
        stars.query('altitude > {} & vmag < {}'.format(np.deg2rad(90 - float(conf['image']['openingangle'])), data['vmaglimit']), inplace=True)
        planets.query('altitude > {} & vmag < {}'.format(np.deg2rad(90 - float(conf['image']['openingangle'])), data['vmaglimit']), inplace=True)
        points_of_interest.query('altitude > {}'.format(np.deg2rad(90 - float(conf['image']['openingangle']))), inplace=True)
    except:
        log.error('Using altitude or vmag limit failed!')
        raise

    # calculate angle to moon
    log.debug('Calculate Angle to Moon')
    stars['angleToMoon'] = degDist(stars.altitude.values, moon.alt, stars.azimuth.values, moon.az)
    planets['angleToMoon'] = degDist(planets.altitude.values, moon.alt, planets.azimuth.values, moon.az)
    points_of_interest['angleToMoon'] = degDist(points_of_interest.altitude.values, moon.alt, points_of_interest.azimuth.values, moon.az)

    # remove stars and planets that are too close to moon
    stars.query('angleToMoon > {}'.format(np.deg2rad(float(conf['analysis']['minAngleToMoon']))), inplace=True)
    planets.query('angleToMoon > {}'.format(np.deg2rad(float(conf['analysis']['minAngleToMoon']))), inplace=True)


    # calculate x and y position
    log.debug('Calculate x and y')
    stars['x'], stars['y'] = horizontal2image(stars.azimuth, stars.altitude, cam=conf['image'])
    planets['x'], planets['y'] = horizontal2image(planets.azimuth, planets.altitude, cam=conf['image'])
    points_of_interest['x'], points_of_interest['y'] = horizontal2image(points_of_interest.azimuth, points_of_interest.altitude, cam=conf['image'])
    moonData['x'], moonData['y'] = horizontal2image(moonData['azimuth'], moonData['altitude'], cam=conf['image'])
    sunData['x'], sunData['y'] = horizontal2image(sunData['azimuth'], sunData['altitude'], cam=conf['image'])

    # remove stars and planets that are withing cropping area
    res = list(map(int, split('\\s*,\\s*', conf['image']['resolution'])))
    stars.query('0 < x < {} & 0 < y < {}'.format(res[0] ,res[1]), inplace=True)
    planets.query('0 < x < {} & 0 < y < {}'.format(res[0] ,res[1]), inplace=True)
    points_of_interest.query('0 < x < {} & 0 < y < {}'.format(res[0] ,res[1]), inplace=True)
    stars = stars[stars.apply(lambda s, crop=crop: ~crop[int(s['y']), int(s['x'])], axis=1)]
    planets = planets[planets.apply(lambda p, crop=crop: ~crop[int(p['y']), int(p['x'])], axis=1)]
    points_of_interest = points_of_interest[points_of_interest.apply(lambda s, crop=crop: ~crop[int(s['y']), int(s['x'])], axis=1)]

    # remove stars that are too close to planets because they are brighter and we will mistake them otherwise
    tolerance = int(conf['analysis']['pixelTolerance'])
    for i, pl in planets.iterrows():
        stars.query('~(({} < x < {}) & ({} < y < {}))'.format(pl.x-tolerance, pl.x+tolerance, pl.y-tolerance, pl.y+tolerance), inplace=True) 

    return {'stars':stars, 'planets':planets, 'points_of_interest': points_of_interest, 'moon': moonData, 'sun': sunData, 'lidar': magicLidar_now}


def findLocalStd(img, x, y, radius):
    '''
    ' Returns standard deviation of image within radius
    '''
    try:
        x = int(x)
        y = int(y)
    except TypeError:
        x = x.astype(int)
        y = y.astype(int)
    
    # get interval border
    x_interval = np.max([x-radius,0]) , np.min([x+radius+1, img.shape[1]])
    y_interval = np.max([y-radius,0]) , np.min([y+radius+1, img.shape[0]])
    radius = x_interval[1]-x_interval[0] , y_interval[1]-y_interval[0]

    # do subselection
    subImg = img[y_interval[0]:y_interval[1] , x_interval[0]:x_interval[1]]
    try:
        return np.nanstd(subImg.flatten())
    except RuntimeWarning:
        print('NAN')
        return 0
    except ValueError:
        print('Star outside image')
        return 0


def findLocalMean(img, x, y, radius):
    '''
    ' Returns mean image brightness within radius
    '''
    try:
        x = int(x)
        y = int(y)
    except TypeError:
        x = x.astype(int)
        y = y.astype(int)
    
    # get interval border
    x_interval = np.max([x-radius,0]) , np.min([x+radius+1, img.shape[1]])
    y_interval = np.max([y-radius,0]) , np.min([y+radius+1, img.shape[0]])
    radius = x_interval[1]-x_interval[0] , y_interval[1]-y_interval[0]

    # do subselection
    subImg = img[y_interval[0]:y_interval[1] , x_interval[0]:x_interval[1]]
    try:
        return np.nanmean(subImg.flatten())
    except RuntimeWarning:
        print('NAN')
        return 0
    except ValueError:
        print('Star outside image')
        return 0


def findLocalMaxValue(img, x, y, radius):
    '''
    ' Returns value of brightest pixel within radius
    '''
    try:
        x = int(x)
        y = int(y)
    except TypeError:
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
    except TypeError:
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


def getImageDict(filepath, config, crop=None, fmt=None):
    '''
    Open an image file and return its content as a numpy array.
    
    input:
        filename: full or relativ path to image
        crop: Config section 'crop' defines circles with center and radius
        fmt: timestamp format string. Example: 'gtc_allskyimage_%Y%m%d_%H%M%S.jpg'
            used for parsing the date from filename
    Returns: Dictionary with image array and timestamp datetime object
    '''
    log = logging.getLogger(__name__)

    # get image type from filename
    filename = filepath.split('/')[-1].split('.')[0]
    filetype= filepath.split('.')[-1]

    if stat(filepath).st_size == 0:
        log.error('Image has size 0, aborting!: {}'.format(filepath))
        return
    # is it a matlab file?
    if filetype == 'mat':
        try:
            data = matlab.loadmat(filepath)
            img = data['pic1']
            time = datetime.strptime(
                data['UTC1'][0],
                '%Y/%m/%d %H:%M:%S'
                #config['properties']['timeformat']
            )
        except (KeyError,ValueError,OSError, FileNotFoundError) as e:
            log.error('Failed to open image {}: {}'.format(filepath, e))
            return

    # is it a fits file?
    elif (filetype == 'fits') or (filetype == 'gz'):
        try:
            hdulist = fits.open(filepath, ignore_missing_end=True)
            img = hdulist[0].data
            if hdulist[0].header['BITPIX'] == 16:
                # convert 16bit data from signed to unsigned
                img += 2**15
            if config['properties']['timeKey']:
                # there is a mixture of timekeys in my set of fits files so I need a generic solution
                '''
                time = datetime.strptime(
                    hdulist[0].header[config['properties']['timeKey']],
                    config['properties']['timeformat'],
                    )
                '''
                time=None
                for t in [['UTC', '%Y/%m/%d %H:%M:%S'] , ['TIMEUTC', '%Y-%m-%d %H:%M:%S']]:
                    try:
                        time = datetime.strptime(
                            hdulist[0].header[t[0]],
                            t[1],
                            )
                    except KeyError:
                        pass
                if time == None:
                    raise KeyError('Timestamp not found in file {}'.format(filepath))
            else:
                time = datetime.strptime(
                    filename,
                    config['properties']['timeformat'],
                    )
        except (ValueError, KeyError,OSError,FileNotFoundError) as e:
            log.error('Error parsing timestamp of {}: {}'.format(filepath, e))
            return

    else:
        # read normal image file
        try:
            img = imread(filepath, mode='L', as_grey=True)
        except (FileNotFoundError, OSError, ValueError) as e:
            log.error('Error reading file \'{}\': {}'.format(filename+'.'+filetype, e))
            return
        try:
            if fmt is None:
                time = datetime.strptime(filename, config['properties']['timeformat'])
            else:
                time = datetime.strptime(filename, fmt)
        
        # hardcoded because filename of magic files changed in between
        except ValueError:
            try:
                time = datetime.strptime(filename, 'MAGIC_AllSkyCam_%Y-%m-%d_%H-%M-%S')
            except ValueError:
                try:
                    time = datetime.strptime(filename, 'magic_allskycam_%Y%m%d_%H%M%S')
                except ValueError:
                    fmt = (config['properties']['timeformat'] if fmt is None else fmt)
                    log.error('{},{}'.format(filename,filepath))
                    log.error('Unable to parse image time from filename. Maybe format string is wrong.')
                    return
    time += timedelta(minutes=float(config['properties']['timeoffset']))
    img = img.astype('float32') #needs to be float because we want to set some values NaN while cropping
    return dict({'img': img, 'timestamp': time})


def update_crop_moon(crop_mask, moon, conf):
    '''
    Add crop area for the moon to existing crop mask
    '''
    nrows, ncols = crop_mask.shape
    row, col = np.ogrid[:nrows, :ncols]
    x = moon['x']
    y = moon['y']
    r = theta2r(float(conf['analysis']['minAngleToMoon'])/180*np.pi, float(conf['image']['radius']), how=conf['image']['angleprojection'])
    crop_mask = crop_mask | ((row - y)**2 + (col - x)**2 < r**2)
    return crop_mask


def get_crop_mask(img, crop):
    '''
    Return crop mask specified in 'crop'
    crop is dictionary with cropping information
    returns a boolean array in size of img: False got cropped; True not cropped 
    '''
    nrows, ncols = img.shape
    row, col = np.ogrid[:nrows, :ncols]
    disk_mask = np.full((nrows, ncols), False, dtype=bool)

    try:
        x = list(map(int, split('\\s*,\\s*', crop['crop_x'])))
        y = list(map(int, split('\\s*,\\s*', crop['crop_y'])))
        r = list(map(int, split('\\s*,\\s*', crop['crop_radius'])))
        inside = list(map(int, split('\\s*,\\s*', crop['crop_deleteinside'])))
        for x,y,r,inside in zip(x,y,r,inside):
            if inside == 0:
                disk_mask = disk_mask | ((row - y)**2 + (col - x)**2 > r**2)
            else:
                disk_mask = disk_mask | ((row - y)**2 + (col - x)**2 < r**2)
    except ValueError:
        log = logging.getLogger(__name__)
        log.error('Cropping failed, maybe there is a typing error in the config file?')
        disk_mask = np.full((nrows, ncols), False, dtype=bool)

    return disk_mask


def isInRange(position, stars, rng, unit='deg'):
    '''
    Returns true or false for each star in stars if distance between star and position<rng

    If unit= "pixel" position and star must have attribute .x and .y in pixel and rng is pixel distance
    If unit= "deg" position and star must have attribute .ra and .dec in degree 0<360 and rng is degree
    '''
    if rng < 0:
        raise ValueError
    
    if unit == 'pixel':
        try:
            return ((position.x - stars.x)**2 + (position.y - stars.y)**2 <= rng**2)
        except AttributeError as e:
            log.error('Pixel value needed but object has no x/y attribute. {}'.format(e))
            sys.exit(1)
    elif unit == 'deg':
        try:
            ra1 = position['ra']
            dec1 = position['dec']
            deltaDeg = 2*np.arcsin(np.sqrt(np.sin((dec1-stars.dec)/2)**2 + np.cos(dec1)*np.cos(stars.dec)*np.sin((ra1-stars.ra)/2)**2))
        except (AttributeError, KeyError) as e:
            try:
                alt1 = position['altitude']
                az1 = position['azimuth']
                deltaDeg = 2*np.arcsin(np.sqrt(np.sin((az1-stars.azimuth)/2)**2 + np.cos(az1)*np.cos(stars.azimuth)*np.sin((alt1-stars.altitude)/2)**2))
            except (AttributeError,KeyError) as e:
                log = logging.getLogger(__name__)
                log.error('Degree value needed but object has no ra/dec an no alt/az attribute. {}'.format(e))

                sys.exit(1)

        return deltaDeg <= np.deg2rad(rng)
    else:
        raise ValueError('unit has unknown type')


def calc_star_percentage(position, stars, rng, lim=1, unit='deg', weight=False):
    '''
    Returns: percentage of stars within range of position that are visible 
             and -1 if no stars are in range
    
    Position is dictionary and can contain Ra,Dec and/or x,y
    Range is degree or pixel radius depending on whether unit is 'grad' or 'pixel'
    Lim > 0: is limit visibility that separates visible stars from not visible. [0.0 - 1.0].
    lim < 0 then all stars in range will be used and 'visible' is a weight factor
    Weight = True: each star is multiplied by weight [100**(1/5)]**-magnitude -> bright stars have more impact
    '''

    if rng < 0:
        starsInRange = stars
    else:
        starsInRange = stars[isInRange(position, stars, rng, unit)]

    if starsInRange.empty:
        return np.float64(-1)

    if lim >= 0:
        if weight:
            vis = np.sum(np.power(100**(1/5), -starsInRange.query('visible >= {}'.format(lim)).vmag.values))
            notVis = np.sum(np.power(100**(1/5), -starsInRange.query('visible < {}'.format(lim)).vmag.values))
            percentage = vis/(vis+notVis)
        else:
            percentage = len(starsInRange.query('visible >= {}'.format(lim)).index)/len(starsInRange.index)
    else:
        if weight:
            percentage = np.sum(starsInRange.visible.values * np.power(100**(1/5),-starsInRange.vmag.values)) / \
                np.sum(np.power(100**(1/5),-starsInRange.vmag.values))
        else:
            percentage = np.mean(starsInRange.visible.values)
    return np.float64(percentage)


def calc_cloud_map(stars, rng, img_shape, weight=False):
    '''
    Input:  stars - pandas dataframe
            rng - sigma of gaussian kernel (integer)
            img_shape - size of cloudiness map in pixel (tuple)
            weight - use magnitude as weight or not (boolean)
    Returns: Cloud map of the sky. 1=cloud, 0=clear sky

    Cloudiness is percentage of visible stars in local area. Stars get weighted by
    distance (gaussian) and star magnitude 2.5^-magnitude.
    Instead of a computationally expensive for-loop we use two 2D histograms of the stars (weighted)
    and convolve them with an gaussian kernel resulting in some kind of 'density map'.
    Division of both maps yields the desired cloudines map.
    '''
    if weight:
        scattered_stars_visible,_,_ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=stars.visible.values * 2.5**-stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        scattered_stars,_,_ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=np.ones(len(stars.index)) * 2.5**-stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        density_visible = skimage.filters.gaussian(scattered_stars_visible, rng)
        density_all = skimage.filters.gaussian(scattered_stars, rng)
    else:
        scattered_stars_visible,_,_ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=stars.visible.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        scattered_stars,_,_ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=np.ones(len(stars.index)), bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        density_visible = skimage.filters.gaussian(scattered_stars_visible, rng, mode='mirror')
        density_all = skimage.filters.gaussian(scattered_stars, rng, mode='mirror')
    with np.errstate(divide='ignore',invalid='ignore'):
        # density has some entries close to 0 which will result in artifacts after division.
        # replace them with 1 so that: small number / 1 ~ 0
        density_all[density_all < 10**-10]=1
        cloud_map = np.true_divide(density_visible, density_all)
        cloud_map[~np.isfinite(cloud_map)] = 0
    return 1-cloud_map



def filter_catalogue(catalogue, rng):
    '''
    Check for every star if it has a brighter neighbor. If yes, remove the star, otherwise keep it

    Input:  catalogue - Pandas dataframe (ra and dec in degree)
            rng - Min distance between stars in degree

    Returns: List of indexes that remain in catalogue
    '''
    log = logging.getLogger(__name__)
    c = pd.read_csv(
            catalogue,
            sep=';',
            comment='#',
            header=0,
            skipinitialspace=True,
            na_values=[' ',''],
        )
    # keep stars with varflag < 2 or varflag == NaN
    # these stars have a vmag fluctuation < 0.06mag
    # also remove stars darker than mag=10
    c.query('VarFlag <2 | VarFlag!=VarFlag', inplace=True)
    c.query('vmag < 10', inplace=True)
    c.index = c.HIP

    try:
        c.sort_values('vmag', ascending=False, inplace=True)
        positionList = list(np.deg2rad(c[['ra','dec']].values))
        indexList = list(c.index.values)
    except KeyError:
        log.error('Key not found. Please check that your catalogue is labeled correctly')
        raise
    
    popped = 0 #count popped stars
    i1 = 0 #index of star that will be checked
    while i1 < len(positionList)-1:
        if popped % 50 == 0:
            print('Left to process / size of filtered catalogue: {} / {}'.format(len(positionList)-i1, len(positionList)))
        # calculate distance to all brighter stars
        positions = np.array(positionList[i1+1:])
        deltaDeg = degDist(positionList[i1][1], positions[:,1], positionList[i1][0], positions[:,0])
        # remove = True if any brighter star is closer than rng
        remove = np.sum(deltaDeg < float(rng)) > 0
        if remove:
            positionList.pop(i1)
            indexList.pop(i1)
            popped += 1
        else:
            i1+=1
    print('Catalogue had {} entries, {} remain after filtering'.format(len(c.index), len(indexList)))
    return c.ix[indexList]


def process_image(images, data, configList, args):
    '''
    This function applies all calculations to an image and returns results as dict.
    For details read the comments below.

    Use this in the main loop!
    '''
    font = {'size'   : 12}
    rc('font', **font)

    log = logging.getLogger(__name__)

    output = dict()
    if not images:
        return

    config = None
    for i in range(len(configList)):
        if np.datetime64(configList[i]['properties']['useConfAfter']) < np.datetime64(images['timestamp']):
            config = configList[i]
        else:
            # stop once a config file does not meet the condition. We assume that the config files are ordered
            # such that the start times will be processed in ascending order
            break
    if config == None:
        log.error('No config file with valid start date < {}'.format(images['timestamp']))
        return

    log.info('Processing image taken at: {}'.format(images['timestamp']))
    observer = obs_setup(config['properties'])
    observer.date = images['timestamp']
    data['timestamp'] = images['timestamp']


    # stop processing if sun is too high or config file does not match
    if images['img'].shape[1]  != int(config['image']['resolution'].split(',')[0]) or images['img'].shape[0]  != int(config['image']['resolution'].split(',')[1]):
        log.error('Resolution does not match: {}!={}. Wrong config file?'.format(images['img'].shape,config['image']['resolution']))
        return
    sun = ephem.Sun()
    sun.compute(observer)
    moon = ephem.Moon()
    moon.compute(observer)
    if np.rad2deg(sun.alt) > -15:
        log.info('Sun too high: {}° above horizon. We start below -15°, current time: {}'.format(np.round(np.rad2deg(sun.alt),2), images['timestamp']))
        return 

    # put timestamp and hash sum into output dict
    output['timestamp'] = images['timestamp']
    try:
        output['hash'] = sha1(images['img'].data).hexdigest()
    except BufferError:
        output['hash'] = sha1(np.ascontiguousarray(images['img']).data).hexdigest()
        

    # create cropping mask for unneccessary image regions.
    crop_mask = get_crop_mask(images['img'], config['crop'])

    # update celestial objects
    celObjects = update_star_position(data, observer, config, crop_mask, args)
    # merge objects (ignore planets, because they are bigger than stars and mess up the detection)
    stars = pd.concat([celObjects['stars'],])# celObjects['planets']])
    if stars.empty:
        log.error('No stars in DataFrame. Maybe all got removed by cropping? No analysis possible.')
        return
    # also crop the moon
    crop_mask = update_crop_moon(crop_mask, celObjects['moon'], config)
    images['img'][crop_mask] = np.NaN
    output['brightness_mean'] = np.nanmean(images['img'])
    output['brightness_std'] = np.nanmean(images['img'])
    img = images['img']
    
    # calculate response of stars with image kernel
    if args['--kernel']:
        #kernelSize = float(args['--kernel']),
        kernelSize = np.round(np.arange(1, float(args['--kernel'])+0.1, 0.1),1)
        stars_orig = stars.copy()
    else:
        kernelSize = [float(config['analysis']['kernelsize'])]
    kernelResults = list()

    for k in kernelSize:
        log.debug('Apply image filters. Kernelsize = {}'.format(k))

        # undo all changes, if we are testing multiple kernel sizes
        if len(kernelSize) > 1:
            stars = stars_orig.copy()
        stars['kernel'] = k
    
        # prepare LoG kernel
        x,y = np.meshgrid(range(int(np.floor(-3*k)), int(np.ceil(3*k+1))), range(int(np.floor(-3*k)), int(np.ceil(3*k+1))))
        LoG_kernel = LoG(x, y, k)

        # chose the response function and apply it to the image
        # result will be stored as 'resp'
        if args['--function'] == 'All' or args['--ratescan']:
            grad = (img - np.roll(img, 1, axis=0)).clip(min=0)**2 + (img - np.roll(img, 1, axis=1)).clip(min=0)**2
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                sobel = convolve(img, [[1,2,1],[0,0,0],[-1,-2,-1]])**2 + convolve(img, [[1,0,-1],[2,0,-2],[1,0,-1]])**2
                log = convolve_fft(img, LoG_kernel)

            grad[crop_mask] = np.NaN
            sobel[crop_mask] = np.NaN
            log[crop_mask] = np.NaN

            images['grad'] = grad
            images['sobel'] = sobel
            images['log'] = log
            resp = log
        elif args['--function'] == 'DoG':
            resp = skimage.filters.gaussian(img, sigma=k) - skimage.filters.gaussian(img, sigma=1.6*k)
        elif args['--function'] == 'LoG':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                resp = convolve_fft(img, LoG_kernel)
        elif args['--function'] == 'Grad':
            resp = ((img - np.roll(img, 1, axis=0)).clip(min=0))**2 + ((img - np.roll(img, 1, axis=1)).clip(min=0))**2
        elif args['--function'] == 'Sobel':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                resp = convolve(img, [[1,2,1],[0,0,0],[-1,-2,-1]])**2 + convolve(img, [[1,0,-1],[2,0,-2],[1,0,-1]])**2
        else:
            log.error('Function name: \'{}\' is unknown!'.format(args['--function']))
            sys.exit(1)
        resp[crop_mask] = np.NaN
        images['response'] = resp


        # to correct abberation the max filter response withing tolerance distance around a star will be chosen as 'real' star position 
        # there should be no need for this to be bigger than the diameter of the bigger stars because this means that the transformation we use is quite bad
        # and a bright star next to the real star might be detected by error
        tolerance = int(config['analysis']['pixelTolerance'])
        log.debug('Calculate Filter response')
        
        # calculate x and y position where response has its max value (search within 'tolerance' range)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stars = pd.concat([stars.drop(['maxX','maxY'], errors='ignore', axis=1), stars.apply(
                    lambda s : findLocalMaxPos(resp, s.x, s.y, tolerance),
                    axis=1)], axis=1
            )

        # drop stars that got mistaken for a brighter neighboor
        stars = stars.sort_values('vmag').drop_duplicates(subset=['maxX', 'maxY'], keep='first')

        # calculate response and drop stars that were not found at all, because response=0 interferes with log-plot
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stars['response'] = stars.apply(lambda s : findLocalMaxValue(resp, s.x, s.y, tolerance), axis=1)
        #stars['response_mean'] = stars.apply(lambda s : findLocalMean(img, s.x, s.y, 50), axis=1)
        #stars['response_std'] = stars.apply(lambda s : findLocalStd(img, s.x, s.y, 50), axis=1)
        stars.query('response > 1e-100', inplace=True)

        # correct atmospherice absorbtion
        stars['response_orig'] = stars.response
        stars['response'] = stars.response / transmission3(stars.altitude, 1.0, float(config['calibration']['airmass_absorbtion']))
        
        if args['--function'] == 'All' or args['--ratescan']:
            stars['response_grad'] = stars.apply(lambda s : findLocalMaxValue(grad, s.x, s.y, tolerance), axis=1)
            stars['response_sobel'] = stars.apply(lambda s : findLocalMaxValue(sobel, s.x, s.y, tolerance), axis=1)
        ulim, llim = (list(map(float, split('\\s*,\\s*', config['analysis']['visibleupperlimit']))), 
                list(map(float, split('\\s*,\\s*', config['analysis']['visiblelowerlimit']))))

        # offset of upper limit can be reduced if moonlight reduces exposure time
        if not args['--moon'] and np.rad2deg(moon.alt) > 10.0:
            ulim[1]=ulim[1] - np.log10(float(config['analysis']['moonExposureFactor']))

        # remove stars for all magnitudes where upperLimit < lowerLimit
        if ulim[0] != llim[0]:
            intersection = (llim[1] - ulim[1]) / (ulim[0] - llim[0])
            if ulim[0] < llim[0]:
                data['vmaglimit'] = min(intersection, data['vmaglimit'])
                #stars.loc[stars.vmag.values > data['vmaglimit'], 'visible'] = 0
                stars.query('vmag < {}'.format(data['vmaglimit']), inplace=True)
            else:
                #stars.loc[stars.vmag.values < intersection, 'visible'] = 0
                stars.query('vmag > {}'.format(intersection), inplace=True)

        # calculate visibility percentage
        # if response > visibleUpperLimit -> visible=1
        # if response < visibleUpperLimit -> visible=0
        # if in between: scale linear
        stars['visible'] = np.minimum(
                1,
                np.maximum(
                    0,
                    (np.log10(stars['response']) - (stars['vmag']*llim[0] + llim[1])) / 
                    ((stars['vmag']*ulim[0] + ulim[1]) - (stars['vmag']*float(llim[0]) + llim[1]))
                    )
                )

        # append results
        kernelResults.append(stars)

    del stars
    try:
        del stars_orig
    except UnboundLocalError:
        pass

    # merge all stars (if neccessary)
    try:
        celObjects['stars'] = pd.concat(kernelResults, keys=kernelSize)
    except ValueError:
        celObjects['stars'] = kernelResults[0]

    # use 'stars' as substitution because it is shorter
    celObjects['stars'].reset_index(0, inplace=True)
    stars = celObjects['stars']

    if len(kernelSize) == 1:
        celObjects['points_of_interest']['starPercentage'] = celObjects['points_of_interest'].apply(
                lambda p,stars=stars : calc_star_percentage(p, stars, p.radius, unit='deg', lim=-1, weight=True),
                axis=1)
    else:
        log.warning('Can not process points_of_interest if multiple kernel sizes get used')
    output['global_star_perc'] = calc_star_percentage({'altitude': np.pi/2, 'azimuth':0}, stars, float(config['image']['openingangle']), unit='deg', lim=-1, weight=True)
    output['magic_lidar'] = celObjects['lidar']
    
    ##################################
    # processing done. Now plot everything
    ##################################

    if args['--kernel']:
        res = list()
        gr = stars.query('kernel>=1 & vmag<4').reset_index().groupby('HIP')
        ax = plt.figure().add_subplot(111)
        color = cm.jet(np.linspace(0,1,10 * (gr.vmag.max().max()-gr.vmag.min().min())+2 ))
        for _, s in gr:
            # normalize
            n = s.response_orig.max()
            res.append(s.query('response_orig == {}'.format(n)).kernel.values)
            plt.plot(s.kernel.values, s.response_orig.values/n, marker='o', c=color[round(s.vmag.max()*10)])
        ax.set_xlim(0., 5.1)
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('$\sigma$ of LoG filter')
        ax.set_ylabel('Kernel response normalized')
        lEntry = Line2D([], [], color='black', marker='o', markersize=6, label='Response of all stars')
        ax.grid()
        ax.legend(handles=[lEntry])
        if args['-s']:
            plt.savefig('kernel_curve_{}.png'.format(config['properties']['name']))
        if args['-v']:
            plt.show()
        plt.close('all')


        gr = stars.query('kernel>=1 & vmag<4').reset_index().groupby('kernel')
        res = list()
        fig = plt.figure(figsize=(16,10))
        plt.grid()
        gs = gridspec.GridSpec(2, 1, height_ratios=[4, 1])
        ax = plt.subplot(gs[0])
        for _, s in gr:
            # dont plot faint stars
            popt, pcov = curve_fit(expo, s.vmag.values, s.response_orig.values)
            res.append((s.kernel.max(),*popt, np.sqrt(pcov[0,0]), np.sqrt(pcov[1,1])))
        res = np.array(res)
        ax.scatter(res[:,0],res[:,2], label='b')
        ax.set_xlabel('$\sigma$ of LoG filter')
        ax.set_ylabel('b')
        ax.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(b=True, which='major')
        ax.grid(b=True, which='minor')
        ax2 = plt.subplot(gs[1], sharex=ax)
        ax2.scatter(res[:,0],res[:,4], label='b')
        ax2.set_ylabel('Standard deviation')
        ax2.set_xlim((kernelSize[0]*0.9, kernelSize[-1]+.1))
        ax2.set_ylim((np.min(res[:,4])*0.95, np.max(res[:,4])*1.08))
        ax2.get_xaxis().set_minor_locator(ticker.AutoMinorLocator())
        ax2.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        ax2.grid(b=True, which='major')
        ax2.grid(b=True, which='minor')
        plt.tight_layout()
        if args['-s']:
            plt.savefig('chose_sigma_{}.png'.format(config['properties']['name']))
        if args['-v']:
            plt.show()
        plt.close('all')
        del res, gr


    if args['--cam'] or args['--daemon']:
        output['img'] = img
        fig = plt.figure(figsize=(8,6) )
        ax = fig.add_subplot(111)
        vmin = np.nanpercentile(img, 5)
        vmax = np.nanpercentile(img, 90.)
        ax.imshow(img, vmin=vmin,vmax=vmax, cmap='gray')
        cax = ax.scatter(stars.x.values, stars.y.values, c=stars.visible.values, cmap = plt.cm.RdYlGn, s=10, vmin=0, vmax=1)

        for row in celObjects['points_of_interest'].iterrows():
            ax.plot(row[1].x, row[1].y, marker='^', label=row[1]["name"], linestyle='None')
            
        ax.text(0.04, 0.985, str(output['timestamp']),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            backgroundcolor='white',
            color='black', fontsize=12,
        )
        plt.legend(loc=8, ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.01))
        cbar = fig.colorbar(cax)
        cbar.ax.set_ylabel('Visibility')

        if args['-s']:
            plt.savefig('cam_image_{}.png'.format(images['timestamp'].isoformat()))
        if args['--daemon']:
            plt.savefig('cam_image_{}.png'.format(config['properties']['name']),dpi=300)
        if args['-v']:
            plt.show()
        plt.close('all')

    if args['--single'] or args['--daemon']:
        if args['--response'] or args['--daemon']:
            fig = plt.figure(figsize=(16,9))
            ax = plt.subplot(111)
            ax.semilogy()

            # draw visibility limits
            x = np.linspace(-5+stars.vmag.min(), stars.vmag.max()+5, 20)
            y1 = 10**(x*llim[0] + llim[1])
            y2 = 10**(x*ulim[0] + ulim[1])
            ax.plot(x, y1, c='red', label='lower limit')
            ax.plot(x, y2, c='green', label='upper limit')

            stars.plot.scatter(x='vmag', y='response', ax=ax, logy=True, c=stars.visible.values,
                    cmap = plt.cm.RdYlGn, grid=True, vmin=0, vmax=1, label='Kernel Response', s=40)
            ax.set_xlim((-1, float(data['vmaglimit'])+0.5))
            ax.set_ylim((
                    10**(llim[0]*float(config['analysis']['vmaglimit'])+llim[1]-1),
                    10**(ulim[0]*-1+ulim[1])
                    ))
            ax.set_ylabel('Kernel Response')
            ax.set_xlabel('Star Magnitude')
            if args['-c'] == 'GTC':
                if args['--function'] == 'Grad':
                    ax.axhspan(ymin=11**2/255**2, ymax=13**2/255**2, color='red', alpha=0.5, label='old threshold range')
                ax.axvline(4.5, color='black', label='Magnitude lower limit')

            # show camera image in a subplot
            ax_in= inset_axes(ax,
                    width='30%',
                    height='40%',
                    loc=3)
            vmin = np.nanpercentile(img, 0.5)
            vmax = np.nanpercentile(img, 99.)
            ax_in.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
            color = cm.RdYlGn(stars.visible.values)
            stars.plot.scatter(x='x',y='y', ax=ax_in, c=color, vmin=0, vmax=1, grid=True)
            ax_in.get_xaxis().set_visible(False)
            ax_in.get_yaxis().set_visible(False)
            
            leg = ax.legend(loc='best')
            leg.legendHandles[2].set_color('yellow')
            plt.tight_layout()
            if args['-s']:
                plt.savefig('response_{}_{}.png'.format(args['--function'], images['timestamp'].isoformat()))
            if args['--daemon']:
                plt.savefig('response_{}.png'.format(config['properties']['name']),dpi=200)
            if args['-v']:
                plt.show()
            plt.close('all')

        if args['--ratescan']:
            log.info('Doing ratescan')
            gradList = list()
            sobelList = list()
            logList = list()

            for resp_grad, resp_sobel, resp_log in zip(np.logspace(np.log10(stars.response_grad.min()/5), np.log10(stars.response_grad.max()), 100),
                        np.logspace(np.log10(stars.response_sobel.min()), np.log10(stars.response_sobel.max()), 100),
                        np.logspace(np.log10(stars.response_orig.min()), np.log10(stars.response_orig.max()), 100)):

                _, num_of_clusters = label(grad>resp_grad)
                size_of_clusters = np.sum(grad>resp_grad)/num_of_clusters
                perc_of_vis_stars = np.mean(stars.response_grad > resp_grad)
                gradList.append((resp_grad, num_of_clusters, size_of_clusters, perc_of_vis_stars))

                _, num_of_clusters = label(sobel>resp_sobel)
                size_of_clusters = np.sum(grad>resp_sobel)/num_of_clusters
                perc_of_vis_stars = np.mean(stars.response_sobel > resp_sobel)
                sobelList.append((resp_sobel, num_of_clusters, size_of_clusters, perc_of_vis_stars))

                _, num_of_clusters = label(log>resp_log)
                size_of_clusters = np.sum(log>resp_log)/num_of_clusters
                perc_of_vis_stars = np.mean(stars.response_orig > resp_log)
                logList.append((resp_log, num_of_clusters, size_of_clusters, perc_of_vis_stars))


            gradList = np.array(gradList)
            sobelList = np.array(sobelList)
            logList = np.array(logList)

            #minThresholds = [max(response[l[:,0]==1]) for l in (gradList, sobelList, logList)]

            # find minimal threshold for detecting 100% of all stars and number of clusters at that threshold
            minThresholdPos = -np.array([np.argmax(gradList[::-1,0]), np.argmax(sobelList[::-1,0]), np.argmax(logList[::-1,0])]) + len(response) -1
            thresh = (response[minThresholdPos[0]], response[minThresholdPos[1]],response[minThresholdPos[2]])
            clusters = (gradList[minThresholdPos[0],2], sobelList[minThresholdPos[1],2], logList[minThresholPods[2],2])

            fig = plt.figure(figsize=(19.2,10.8))
            ax1 = fig.add_subplot(111)
            plt.xscale('log')
            plt.grid()
            ax1.plot(response, sobelList[:,0], marker='x', c='blue', label='Sobel Kernel - Percent')
            ax1.plot(response, logList[:,0], marker='x', c='red', label='log Kernel - Percent')
            ax1.plot(response, gradList[:,0], marker='x', c='green', label='Square Gradient - Percent')
            ax1.axvline(response[minThresholdPos[0]], color='green')
            ax1.axvline(response[minThresholdPos[1]], color='blue')
            ax1.axvline(response[minThresholdPos[2]], color='red')
            ax1.axvline(14**2/255**2, color='black', label='old threshold')
            ax1.set_ylabel('')
            ax1.legend(loc='center left')

            ax2 = ax1.twinx()
            #ax2.plot(response, gradList[:,1], marker='o', c='green', label='Square Gradient - Pixcount')
            #ax2.plot(response, sobelList[:,1], marker='o', c='blue', label='Sobel Kernel - Pixcount')
            #ax2.plot(response, logList[:,1], marker='o', c='red', label='log Kernel - Pixcount')
            ax2.plot(response, gradList[:,2], marker='s', c='green', label='Square Gradient - Clustercount')
            ax2.plot(response, sobelList[:,2], marker='s', c='blue', label='Sobel Kernel - Clustercount')
            ax2.plot(response, logList[:,2], marker='s', c='red', label='log Kernel - Clustercount')
            ax2.axhline(gradList[minThresholdPos[0],2], color='green')
            ax2.axhline(sobelList[minThresholdPos[1],2], color='blue')
            ax2.axhline(logList[minThresholdPos[2],2], color='red')
            ax2.legend(loc='upper right')
            ax2.set_xlim((min(response), max(response)))
            ax2.set_ylim((0,16000))
            if args['-v']:
                plt.show()
            if args['-s']:
                plt.savefig('rateScan.pdf')
            plt.close('all')

            output['response'] = response
            output['thresh'] = thresh
            output['minThresh'] = minThresholds
            del grad
            del sobel
            del log

    if args['--cloudmap'] or args['--cloudtrack'] or args['--daemon']:
        log.debug('Calculating cloud map')
        # empirical good value for radius of map: Area r**2 * PI should contain 0.75 stars on average
        cloud_map = calc_cloud_map(stars, np.sqrt(1./ (len(stars.index)/float(config['image']['radius'])**2)), img.shape, weight=True)
        cloud_map[crop_mask] = 1
        if args['--cloudtrack']:
            output['cloudmap'] = cloud_map
            np.save('cMap_{}'.format(output['timestamp']), np.nan_to_num(cloud_map))
        if args['--cloudmap']:
            fig = plt.figure(figsize=(16,9))
            ax1 = fig.add_subplot(121)
            vmin = np.nanpercentile(img, 5.5)
            vmax = np.nanpercentile(img, 99.9)
            ax1.imshow(img, vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
            ax1.set_ylabel('$y$ / px')
            ax1.grid()
            ax1.text(0.98, 0.02, str(output['timestamp']),
                verticalalignment='bottom', horizontalalignment='right',
                transform=ax1.transAxes,
                backgroundcolor='black',
                color='white', fontsize=15,
            )
            ax2 = fig.add_subplot(122)
            ax2.imshow(cloud_map, cmap='gray_r', vmin=0, vmax=1)
            ax2.grid()
            ax2.set_yticks([])
            fig.text(0.53, 0.02, '$x$ / px', ha='center')
            plt.tight_layout(h_pad=-0.1)
            if args['-s']:
                plt.savefig('cloudMap_{}.png'.format(images['timestamp'].isoformat()))
            if args['-v']:
                plt.show()
            plt.close('all')
        if args['--daemon']:
            ax = plt.subplot(111)
            ax.imshow(cloud_map, cmap='gray_r', vmin=0, vmax=1)
            ax.grid()
            plt.savefig('cloudMap_{}.png'.format(config['properties']['name']),dpi=400)
    try:
        output['global_coverage'] = np.nanmean(cloudmap)
    except NameError:
        log.debug('Cloudmap not available. Calculating global_coverage not possible')
        output['global_coverage'] = np.float64(-1)

    del images
    output['stars'] = stars
    output['points_of_interest'] = celObjects['points_of_interest']
    output['sun_alt'] = celObjects['sun']['altitude']
    output['moon_alt'] = celObjects['moon']['altitude']
    output['moon_phase'] = celObjects['moon']['moonPhase']

    if args['--sql']:
        try:
            sql.writeSQL(config, output)
        except (OperationalError) as e:
            log.error('Writing to SQL server failed. Server up? Password correct? {}'.format(e))
            sys.exit(1)
        except InternalError as e:
            log.error('Error while writing to SQL server: {}'.format(e))


    if args['--low-memory']:
        slimOutput = dict()
        for key in ['timestamp', 'hash', 'points_of_interest', 'sun_alt', 'moon_alt', 'moon_phase', 'brightness_mean', 'brightness_std']:
            try:
                slimOutput[key] = [output[key]]
            except KeyError:
                log.warning('Key {} was not found in dataframe so it can not be returned/stored'.format(key))
        del output
        output = slimOutput
        del slimOutput
                
    if args['--daemon']:
        del output
        output = None

    log.info('Done')
    return output
