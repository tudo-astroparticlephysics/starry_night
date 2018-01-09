from starry_night import sql
import pandas as pd
import numpy as np

import ephem
import sys

from astropy.time import Time

from scipy.ndimage.measurements import label
import skimage.filters
import warnings

from pkg_resources import resource_filename
from os.path import join
import logging

import re

from sqlalchemy.exc import OperationalError, InternalError
from hashlib import sha1
import shutil

from .transmission import transmission_spheric
from .skycoords import (
    ho2eq,
    horizontal2image,
    equatorial2horizontal,
    eq2ho,
    obs_setup,
    degDist,
)
from .optics import theta2r
from .plotting import (
    plot_kernel_curve,
    plot_kernel_response,
    plot_camera_image,
    plot_choose_sigma,
    plot_cloudmap_and_image,
    plot_cloudmap,
    plot_ratescan,
)
from .io import getMagicLidar
from .config import get_config_for_timestamp
from .image_kernels import (
    apply_log_kernel,
    apply_sobel_kernel,
    apply_gradient,
    apply_difference_of_gaussians
)
from .star_detection import calculate_star_visibility


def lin(x, m, b):
    'Just a linear function'
    return m * x + b


def getBlobsize(img, thresh, limit=0):
    '''
    Returns size of the blob in the center of img.
    If the blob is bigger than limit, limit gets returned immideatly.

    A blob consists of all 8 neighboors that are bigger than 'thresh' and their neighboors respectively.
    '''
    if thresh <= 0:
        raise ValueError('Thresh > 0 required')
    if img.shape[0] % 2 == 0 or img.shape[1] % 2 == 0:
        raise IndexError('Only odd sized arrays are supported. Array shape:{}'.format(img.shape))
    if limit == 0:
        limit = img.shape[0] * img.shape[1]

    center = (img.shape[0] // 2, img.shape[1] // 2)

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
        x, y = nList.pop(0)

        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                if x + i < 0 or x + i >= img.shape[0] or y + j < 0 or y + j >= img.shape[1]:
                    pass
                elif tempImg[x + i, y + j] >= thresh:
                    count += 1
                    tempImg[x + i, y + j] = 0
                    nList.append((x + i, y + j))
        if count >= limit:
            return limit
    return count


def find_matching_pos(img_timestamp, time_pos_list, conf):
    '''
    Return position of (e.g. LIDAR) in the moment the image was taken.

    Returns entry from 'time_pos_list' that has the closest timestamp to 'img_timestamp'.
    Tolerance: +10min into futur ,-1min past. Because LIDAR result is based on 10min measurement.

    Converts coordinates in horizontal coords.
    '''
    subset = time_pos_list.query('-1/24/60 * 10 < MJD - {} < 1/24/60*1'.format(Time(img_timestamp).mjd)).sort_values('MJD')
    closest = subset[subset.MJD == subset.MJD.min()]

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
        return closest[['azimuth', 'altitude', 'ra', 'dec']]


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

    return {
        'stars': stars,
        'planets': planets,
        'points_of_interest': points_of_interest,
        'sun': sunData,
        'moon': moonData,
    }


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
        'moonPhase': float(moon.moon_phase),
        'altitude': float(moon.alt),
        'azimuth': float(moon.az),
    }

    # include sun data
    log.debug('Load Sun')
    sun = ephem.Sun()
    sun.compute(observer)
    sunData = {
        'altitude': float(sun.alt),
        'azimuth': float(sun.az),
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
            points_of_interest = points_of_interest.append({
                'name': magicLidar_now['name'],
                'ID': magicLidar_now['ID'],
                'radius': magicLidar_now['radius'],
                'altitude': magicLidar_now['altitude'],
                'azimuth': magicLidar_now['azimuth'],
            }, ignore_index=True)

    try:
        stars.query('altitude > {} & vmag < {}'.format(np.deg2rad(90 - float(conf['image']['openingangle'])), data['vmagLimit']), inplace=True)
        planets.query('altitude > {} & vmag < {}'.format(np.deg2rad(90 - float(conf['image']['openingangle'])), data['vmagLimit']), inplace=True)
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
    res = list(map(int, re.split('\\s*,\\s*', conf['image']['resolution'])))
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
    x_interval = np.max([x - radius, 0]), np.min([x + radius + 1, img.shape[1]])
    y_interval = np.max([y - radius, 0]), np.min([y + radius + 1, img.shape[0]])
    radius = x_interval[1] - x_interval[0], y_interval[1] - y_interval[0]

    # do subselection
    subImg = img[y_interval[0]:y_interval[1], x_interval[0]:x_interval[1]]
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
    x_interval = np.max([x - radius, 0]), np.min([x + radius + 1, img.shape[1]])
    y_interval = np.max([y - radius, 0]), np.min([y + radius + 1, img.shape[0]])
    radius = x_interval[1] - x_interval[0], y_interval[1] - y_interval[0]

    # do subselection
    subImg = img[y_interval[0]: y_interval[1], x_interval[0]:x_interval[1]]
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
    x_interval = np.max([x - radius, 0]), np.min([x + radius + 1, img.shape[1]])
    y_interval = np.max([y - radius, 0]), np.min([y + radius + 1, img.shape[0]])
    radius = x_interval[1] - x_interval[0], y_interval[1] - y_interval[0]

    # do subselection
    subImg = img[y_interval[0]:y_interval[1], x_interval[0]:x_interval[1]]
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
    x_interval = np.max([x - radius, 0]), np.min([x + radius + 1, img.shape[1]])
    y_interval = np.max([y - radius, 0]), np.min([y + radius + 1, img.shape[0]])
    radius = x_interval[1] - x_interval[0], y_interval[1] - y_interval[0]
    subImg = img[y_interval[0]:y_interval[1], x_interval[0]:x_interval[1]]
    if np.max(subImg) != np.min(subImg):
        try:
            maxPos = np.nanargmax(subImg)
            x = (maxPos % radius[0]) + x_interval[0]
            y = (maxPos // radius[0]) + y_interval[0]
        except ValueError:
            return pd.Series({'maxX': 0, 'maxY': 0})
    return pd.Series({'maxX': int(x), 'maxY': int(y)})


def update_crop_moon(crop_mask, moon, conf):
    '''
    Add crop area for the moon to existing crop mask
    '''
    nrows, ncols = crop_mask.shape
    row, col = np.ogrid[:nrows, :ncols]
    x = moon['x']
    y = moon['y']
    r = theta2r(float(conf['analysis']['minAngleToMoon']) / 180 * np.pi, float(conf['image']['radius']), how=conf['image']['angleprojection'])
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
        x = list(map(int, re.split('\\s*,\\s*', crop['crop_x'])))
        y = list(map(int, re.split('\\s*,\\s*', crop['crop_y'])))
        r = list(map(int, re.split('\\s*,\\s*', crop['crop_radius'])))
        inside = list(map(int, re.split('\\s*,\\s*', crop['crop_deleteinside'])))
        for x, y, r, inside in zip(x, y, r, inside):
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
            log = logging.getLogger(__name__)
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
            vis = np.sum(np.power(100**(1 / 5), -starsInRange.query('visible >= {}'.format(lim)).vmag.values))
            notVis = np.sum(np.power(100**(1 / 5), -starsInRange.query('visible < {}'.format(lim)).vmag.values))
            percentage = vis / (vis + notVis)
        else:
            percentage = len(starsInRange.query('visible >= {}'.format(lim)).index)/len(starsInRange.index)
    else:
        if weight:
            percentage = np.sum(
                starsInRange.visible.values * np.power(100**(1/5), -starsInRange.vmag.values)
            ) / np.sum(np.power(100**(1/5),-starsInRange.vmag.values))
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
        scattered_stars_visible, _, _ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=stars.visible.values * 2.5**-stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        scattered_stars, _, _ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=np.ones(len(stars.index)) * 2.5**-stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
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
    return 1 - cloud_map


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

    popped = 0  # count popped stars
    i1 = 0  # index of star that will be checked
    while i1 < len(positionList) - 1:
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


def hash_image(image):
    try:
        return sha1(image.data).hexdigest()
    except BufferError:
        return sha1(np.ascontiguousarray(image).data).hexdigest()


def process_image(image, timestamp, data, configs, args, kernel_function):
    '''
    This function applies all calculations to an image and returns results as dict.
    For details read the comments below.

    Use this in the main loop!
    '''
    log = logging.getLogger(__name__)
    image = image.copy()

    allowed_kernels = {'All', 'Grad', 'DoG', 'LoG', 'Sobel'}
    if kernel_function not in allowed_kernels:
        raise ValueError('kernel_function must be any of {}'.format(allowed_kernels))

    try:
        config = get_config_for_timestamp(configs, timestamp)
    except ValueError:
        log.error('No valid config found for {}'.format(timestamp))
        return

    log.info('Processing image taken at: {}'.format(timestamp))
    observer = obs_setup(
        config['properties']['latitude'],
        config['properties']['longitude'],
        float(config['properties']['elevation']),
    )
    observer.date = timestamp

    # stop processing if sun is too high or config file does not match
    reference_shape = tuple(map(int, config['image']['resolution'].split(',')))
    if image.shape == reference_shape:
        log.error(
            'Resolution does not match: {} != {}. Wrong config file?'.format(
                image.shape, reference_shape
            )
        )
        return

    sun = ephem.Sun()
    sun.compute(observer)
    moon = ephem.Moon()
    moon.compute(observer)
    if np.rad2deg(sun.alt) > -15:
        log.info((
            'Sun too high: {:1.2f}° above horizon.'
            ' We start below -15°, current time: {}'
        ).format(np.rad2deg(sun.alt), timestamp))
        return

    # put timestamp and hash sum into output dict
    output = dict()
    output['timestamp'] = timestamp
    output['hash'] = hash_image(image)

    # create cropping mask for unneccessary image regions.
    crop_mask = get_crop_mask(image, config['crop'])

    # update celestial objects
    celObjects = update_star_position(data, observer, config, crop_mask, args)
    # merge objects (ignore planets, because they are bigger than stars and mess up the detection)
    stars = pd.concat([celObjects['stars'], ]) # celObjects['planets']])
    if stars.empty:
        log.error('No stars in DataFrame. Maybe all got removed by cropping? No analysis possible.')
        return

    # also crop the moon
    crop_mask = update_crop_moon(crop_mask, celObjects['moon'], config)
    image[crop_mask] = np.NaN
    output['brightness_mean'] = np.nanmean(image)
    output['brightness_std'] = np.nanmean(image)

    # calculate response of stars with image kernel
    if args['--kernel']:
        # kernelSize = float(args['--kernel']),
        kernelSize = np.round(np.arange(1, float(args['--kernel']) + 0.1, 0.1), 1)
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

        # chose the response function and apply it to the image
        # result will be stored as 'resp'
        if kernel_function == 'All' or args['--ratescan']:
            grad = apply_gradient(image)
            sobel = apply_sobel_kernel(image)
            log = apply_log_kernel(image, k)

            grad[crop_mask] = np.NaN
            sobel[crop_mask] = np.NaN
            log[crop_mask] = np.NaN
            resp = log

        elif kernel_function == 'DoG':
            resp = apply_difference_of_gaussians(image, k)
        elif kernel_function == 'LoG':
            resp = apply_log_kernel(image, k)
        elif kernel_function == 'Grad':
            resp = apply_gradient(image)
        elif kernel_function == 'Sobel':
            resp = apply_sobel_kernel(image)

        resp[crop_mask] = np.NaN
        response = resp

        # to correct abberation the max filter response within tolerance distance
        # around a star will be chosen as 'real' star position
        # there should be no need for this to be bigger than the diameter
        # of the bigger stars because this means that the
        # transformation we use is quite bad
        # and a bright star next to the real star might be detected by error
        tolerance = int(config['analysis']['pixelTolerance'])
        log.debug('Calculate Filter response')

        # calculate x and y position where response
        # has its max value (search within 'tolerance' range)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stars = pd.concat([
                stars.drop(['maxX', 'maxY'], errors='ignore', axis=1),
                stars.apply(lambda s: findLocalMaxPos(resp, s.x, s.y, tolerance), axis=1)
            ], axis=1)

        # drop stars that got mistaken for a brighter neighboor
        stars = stars.sort_values('vmag').drop_duplicates(subset=['maxX', 'maxY'], keep='first')

        # calculate response and drop stars that were not found at all, because response=0 interferes with log-plot
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            stars['response'] = stars.apply(lambda s : findLocalMaxValue(resp, s.x, s.y, tolerance), axis=1)

        # stars['response_mean'] = stars.apply(lambda s : findLocalMean(image, s.x, s.y, 50), axis=1)
        # stars['response_std'] = stars.apply(lambda s : findLocalStd(img, s.x, s.y, 50), axis=1)
        stars.query('response > 1e-100', inplace=True)

        # correct atmospherice absorbtion
        stars['response_orig'] = stars.response
        stars['response'] /= transmission_spheric(
            stars.altitude, 1.0, float(config['calibration']['airmass_absorbtion'])
        )

        if args['--function'] == 'All' or args['--ratescan']:
            stars['response_grad'] = stars.apply(lambda s : findLocalMaxValue(grad, s.x, s.y, tolerance), axis=1)
            stars['response_sobel'] = stars.apply(lambda s : findLocalMaxValue(sobel, s.x, s.y, tolerance), axis=1)

        ulim, llim = (
            list(map(float, re.split('\\s*,\\s*', config['analysis']['visibleupperlimit']))),
            list(map(float, re.split('\\s*,\\s*', config['analysis']['visiblelowerlimit']))),
        )

        # offset of upper limit can be reduced if moonlight reduces exposure time
        if not args['--moon'] and np.rad2deg(moon.alt) > 10.0:
            ulim[1] = ulim[1] - np.log10(float(config['analysis']['moonExposureFactor']))

        # remove stars for all magnitudes where upperLimit < lowerLimit
        if ulim[0] != llim[0]:
            intersection = (llim[1] - ulim[1]) / (ulim[0] - llim[0])
            if ulim[0] < llim[0]:
                data['vmagLimit'] = min(intersection, data['vmagLimit'])
                # stars.loc[stars.vmag.values > data['vmagLimit'], 'visible'] = 0
                stars.query('vmag < {}'.format(data['vmagLimit']), inplace=True)
            else:
                # stars.loc[stars.vmag.values < intersection, 'visible'] = 0
                stars.query('vmag > {}'.format(intersection), inplace=True)

        stars['visible'] = calculate_star_visibility(
            stars['response'],
            stars['vmag'],
            llim,
            ulim,
        )
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

    if args['--kernel'] and args['-s']:
        plot_kernel_curve(
            stars,
            'kernel_curve_{}.png'.format(config['properties']['name']),
        )
        plot_choose_sigma(
            stars,
            kernelSize,
            'chose_sigma_{}.png'.format(config['properties']['name']),
        )

    output['img'] = image
    if (args['--cam'] or args['--daemon']) and args['-s']:
        outputfile = 'cam_image_{}_{}.png'.format(
            config['properties']['name'],
            timestamp.isoformat()
        )
        plot_camera_image(
            image,
            timestamp,
            stars,
            celObjects,
            outputfile=outputfile,
        )
        # if running as daemon, also save the current file without timestamp
        if args['--daemon']:
            shutil.copy2(
                outputfile,
                'cam_image_{}.png'.format(config['properties']['name'])
            )

    if args['--single'] or args['--daemon']:
        if args['--response'] or args['--daemon']:
            plot_kernel_response(
                llim, ulim, float(config['analysis']['vmagLimit']),
                image, data, stars,
                outputfile='response_{}_{}.png'.format(
                    args['--function'],
                    timestamp.isoformat()
                )
            )

        if args['--ratescan']:
            log.info('Doing ratescan')

            ratescan_results = {}
            kernels = ['grad', 'sobel', 'log']

            for kernel in kernels:
                results = []
                response = stars['response_' + kernel]
                thresholds = np.logspace(response.min(), response.max(), 100)
                for threshold in thresholds:
                    _, num_of_clusters = label(grad > threshold)
                    size_of_clusters = np.sum(grad > threshold) / num_of_clusters
                    perc_of_vis_stars = np.mean(response > threshold)
                    results.append((response, num_of_clusters, size_of_clusters, perc_of_vis_stars))

                ratescan_results[kernel] = np.array(results)

            gradList = ratescan_results['grad']
            sobelList = ratescan_results['sobel']
            logList = ratescan_results['log']

            minThresholds = [max(response[l[:, 0] == 1]) for l in (gradList, sobelList, logList)]

            # find minimal threshold for detecting 100% of all stars and number of clusters at that threshold
            minThresholdPos = -np.array([np.argmax(gradList[::-1,0]), np.argmax(sobelList[::-1,0]), np.argmax(logList[::-1,0])]) + len(response) -1
            thresh = (response[minThresholdPos[0]], response[minThresholdPos[1]],response[minThresholdPos[2]])
            clusters = (gradList[minThresholdPos[0], 2], sobelList[minThresholdPos[1], 2], logList[minThresholdPos[2], 2])

            plot_ratescan(response, sobelList, logList, gradList, minThresholdPos, 'ratescan.pdf')

            output['response'] = response
            output['thresh'] = thresh
            output['minThresh'] = minThresholds
            del grad
            del sobel
            del log

    if args['--cloudmap'] or args['--cloudtrack'] or args['--daemon']:
        log.debug('Calculating cloud map')
        # empirical good value for radius of map: Area r**2 * PI should contain 0.75 stars on average
        cloud_map = calc_cloud_map(stars, np.sqrt(1./ (len(stars.index)/float(config['image']['radius'])**2)), image.shape, weight=True)
        cloud_map[crop_mask] = 1
        if args['--cloudtrack']:
            output['cloudmap'] = cloud_map
            np.save('cMap_{}'.format(output['timestamp']), np.nan_to_num(cloud_map))
        if args['--cloudmap']:
            plot_cloudmap_and_image(
                image,
                cloud_map,
                timestamp,
                'cloudMap_{}.png'.format(timestamp.isoformat())
            )
        if args['--daemon']:
            plot_cloudmap(
                cloud_map,
                'cloudMap_{}.png'.format(config['properties']['name']),
            )
    try:
        output['global_coverage'] = np.nanmean(cloudmap)
    except NameError:
        log.debug('Cloudmap not available. Calculating global_coverage not possible')
        output['global_coverage'] = np.float64(-1)

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
