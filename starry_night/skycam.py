import numpy as np
from numpy import sin, cos, tan, arctan2, arcsin, pi
import matplotlib.pyplot as plt

import pandas as pd
import ephem
import sys

from scipy.io import matlab
from skimage.io import imread
from skimage.color import rgb2gray
import skimage.filters.rank as rank

from astropy.io import fits
from datetime import datetime, timedelta
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
                ) * np.cos(az+np.deg2rad(np.float(cam['azimuthoffset'])))
        y = np.float(cam['zenith_y']) - theta2r(np.pi/2 - alt,
                np.float(cam['radius']),
                how=cam['angleprojection']
                ) * np.sin(az+np.deg2rad(np.float(cam['azimuthoffset'])))
    except:
        raise
    return x, y


def obs_setup(properties):
    ''' creates an ephem.Observer for the MAGIC Site at given date '''
    obs = ephem.Observer()
    obs.lon = '-17:53:28'
    obs.lat = '28:45:42'
    obs.elevation = 2200
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


def star_planets_sun_moon_dict():
    '''
    Read the given star catalog, add planets from ephem and fill sun and moon with NaNs
    For horizontal coordinates 'update_star_position()' needs to be called next.

    Returns: dictionary with celestial objects
    '''
    log = logging.getLogger(__name__)
    
    log.debug('Loading stars')
    catalogue = resource_filename('starry_night', '../data/asu.tsv')
    #catalogue = 
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

    stars['altitude'] = np.NaN
    stars['azimuth'] = np.NaN

    # add the planets
    planets = pd.DataFrame()
    for planet in ['Mercury', 'Venus', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']:
        data = {
            'ra': np.NaN,
            'dec': np.NaN,
            'gLon': np.NaN,
            'gLat': np.NaN,
            'vmag': np.NaN,
            'name': planet,
        }
        planets = planets.append(data, ignore_index=True)

    moonData = {
        'moonPhase' : np.NaN,
        'altitude' : np.NaN,
        'azimuth' : np.NaN,
    }

    sunData = {
        'altitude' : np.NaN,
        'azimuth' : np.NaN,
    }

    return dict({'stars': stars,
        'planets': planets,
        'sun': sunData,
        'moon': moonData,
        })


def update_star_position(celestialObjects, observer, cam):
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
        'moonPhase' : moon.moon_phase,
        'altitude' : np.deg2rad(moon.alt),
        'azimuth' : np.deg2rad(moon.az),
    }

    # include sun data
    log.debug('Load Sun')
    sun = ephem.Sun()
    sun.compute(observer)
    sunData = {
        'altitude' : np.deg2rad(sun.alt),
        'azimuth' : np.deg2rad(sun.az),
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
        data = {
            'ra': float(sol_object.a_ra),
            'dec': float(sol_object.a_dec),
            'gLon': float(galactic.lon)/np.pi*180,
            'gLat': float(galactic.lat)/np.pi*180,
            'vmag': float(sol_object.mag),
            'azimuth': float(sol_object.az),
            'altitude': float(sol_object.alt),
            'name': sol_object.name,
        }
        planets = planets.append(data, ignore_index=True)
    planets.set_index('name', inplace=True)

    # remove stars and planets that are not within the limits
    try:
        stars = celestialObjects['stars'].copy()
        stars['azimuth'], stars['altitude'] = equatorial2horizontal(
            stars.ra, stars.dec, observer,
        )
        stars.query('altitude > {} & vmag < {}'.format(np.deg2rad(90 - float(cam['openingangle'])), cam['vmaglimit']), inplace=True)
        planets.query('altitude > {} & vmag < {}'.format(np.deg2rad(90 - float(cam['openingangle'])), cam['vmaglimit']), inplace=True)
    except:
        log.error('Using altitude or vmag limit failed!')
        raise

    # calculate angle to moon
    log.debug('Calculate Angle to Moon')
    stars['angleToMoon'] = stars.apply(lambda x : np.arccos(np.sin(x.altitude)*
        np.sin(moon.alt) + np.cos(x.altitude)*np.cos(moon.alt)*
        np.cos((x.azimuth - moon.az)) )/np.pi*180, axis=1)
    if not planets.empty:
        planets['angleToMoon'] = planets.apply(lambda x : np.arccos(np.sin(x.altitude)*
            np.sin(moon.alt) + np.cos(x.altitude)*np.cos(moon.alt)*
            np.cos((x.azimuth - moon.az)) )/np.pi*180, axis=1)

    # remove stars and planets that are too close to moon
    stars.query('angleToMoon > {}'.format(np.deg2rad(float(cam['minAngleToMoon']))), inplace=True)


    # calculate x and y position
    log.debug('Calculate x and y')
    stars['x'], stars['y'] = horizontal2image(stars.azimuth, stars.altitude, cam=cam)
    planets['x'], planets['y'] = horizontal2image(planets.azimuth, planets.altitude, cam=cam)
    moonData['x'], moonData['y'] = horizontal2image(moonData['azimuth'], moonData['altitude'], cam=cam)
    sunData['x'], sunData['y'] = horizontal2image(sunData['azimuth'], sunData['altitude'], cam=cam)

    return {'stars':stars, 'planets':planets, 'moon': moonData, 'sun': sunData}


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


def getImageDict(filepath, config, crop=None, fmt=None):
    '''
    Open an image file and return its content as a numpy array.
    
    input:
        filename: full or relativ path to image
        crop: crop image to a circle with center and radius
        fmt: format timestring like 'gtc_allskyimage_%Y%m%d_%H%M%S.jpg'
            used for parsing the date from filename
    Returns: Dictionary with image array and timestamp datetime object
    '''
    log = logging.getLogger(__name__)

    #TODO: read image time from mat and fits file
    # get image time from filename
    filename = filepath.split('/')[-1].split('.')[0]
    filetype= filepath.split('.')[-1]
    try:
        if fmt is None:
            time = datetime.strptime(filename, config['image']['timeformat'])
        else:
            time = datetime.strptime(filename, fmt)
        time += timedelta(minutes=float(config['properties']['timeoffset']))
    
    except ValueError:
        fmt = (config['image']['timeformat'] if fmt is None else fmt)
        log.error('{},{}'.format(filename,filepath))
        log.error('Unable to parse image time from filename. Maybe format is wrong: {}'.format(fmt))
        raise
        sys.exit(1)

    # read mat file
    if filetype == 'mat':
        data = matlab.loadmat(filepath)
        img = data[dictEntry]

    # read fits file
    elif (filetype == 'fits') or (filetype == 'gz'):
        hdulist = fits.open(filepath)
        img = hdulist[0].data
    else:
        # read normal image file
        try:
            img = imread(filepath, mode='L', as_grey=True)
        except (FileNotFoundError, OSError):
            log.error('File {} not found. Or filetype invalid'.format(filename))
            sys.exit(1)
    return dict({'img': img, 'timestamp': time})

    
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
            log = logging.getLogger(__name__)
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


def isInRange(position, stars, rng, unit='deg'):
    if rng < 0:
        raise ValueError
    
    if unit == 'pixel':
        if type(position) == dict:
            return ((position.x - stars.x)**2 + (position.y - stars.y)**2 <= rng**2)
        else:
            return ((position[0] - stars.x)**2 + (position[1] - stars.y)**2 <= rng**2)
    elif unit == 'deg':
        ra2 = stars['ra'].values/12*np.pi
        dec2 = stars['dec'].values
        if type(position) == dict:
            ra1 = position['ra']/12*np.pi
            dec1 = position['dec']
        else:
            ra1 = position[0]/12*np.pi
            dec1 = position[1]/180*np.pi

        deltaDeg = 2*np.arcsin(np.sqrt(np.sin((dec1-dec2)/2)**2 + np.cos(dec1)*np.cos(dec2)*np.sin((ra1-ra2)/2)**2))
        return deltaDeg <= np.deg2rad(rng)
    else:
        raise ValueError('unit has unknown type')


def calc_star_percentage(position, stars, rng, unit='deg', weight=False):
    '''
    Returns percentage of visible stars that are within range of position
    
    Position is dictionary and can contain Ra,Dec and/or x,y
    Range is degree or pixel radius depending on whether unit is 'grad' or 'pixel'
    '''
    if rng < 0:
        starsInRange = stars
    else:
        starsInRange = stars[isInRange(position, stars, rng, unit)]

    try:
        if weight:
            vis = np.sum(np.pow(100**(1/5),starsInRange.query('visible').vmag.values))
            notVis = np.sum(np.pow(100**(1/5),starsInRange.query('~visible').vmag.values))
            percentage = vis/(vis+notVis)
        else:
            percentage = len(starsInRange.query('visible').index)/len(starsInRange.index)
    except ZeroDivisionError:
        #log = logging.getLogger(__name__)
        #log.warning('No stars in range to calc percentage. Returning -1.')
        percentage = -1

    return percentage


def calc_cloud_map(stars, rng, img_shape, weight=False):
    visible = stars.query('visible == True')
    nrows = ncols = rng*2 +1
    row, col = np.ogrid[:nrows, :ncols]
    disk_mask = np.zeros((nrows, ncols))
    disk_mask[((row - rng)**2 + (col - rng)**2 < int(rng)**2)] = 1
    if weight:
        scattered_stars,_,_ = np.histogram2d(stars.y.values, stars.x.values, weight=2.5**stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        scattered_stars_visible, _, _ = np.histogram2d(x=visible.y.values, y=visible.x.values, weight=2.5**stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        density_visible = skimage.filters.gaussian(scattered_stars_visible.astype(np.int16), rng)
        density_all = skimage.filters.gaussian(scattered_stars.astype(np.int16), rng)
    else:
        scattered_stars,_,_ = np.histogram2d(stars.y.values, stars.x.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        scattered_stars_visible, _, _ = np.histogram2d(visible.y.values, visible.x.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        #a=rank.sum(star_count_map.astype(np.int16), disk_mask)
        #b=rank.sum(cloud_map.astype(np.int16), disk_mask)
        density_visible = skimage.filters.gaussian(scattered_stars_visible.astype(np.int16), rng)
        density_all = skimage.filters.gaussian(scattered_stars.astype(np.int16), rng)
    #cut = a<3
    #a[cut] = 0
    #b[cut] = 0
    
    return (1-density_visible/density_all)



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


def process_image(images, celestialObjects, config, args):
    log = logging.getLogger(__name__)

    log.debug('Creating observer')
    observer = obs_setup(config['properties'])
    observer.date = images['timestamp']
    log.debug('Image time: {}'.format(images['timestamp']))


    # create cropping array to mask unneccessary image regions.
    img = images['img']
    crop_mask = get_crop_mask(img, config['crop'])

    # update celestial objects
    celObjects = update_star_position(celestialObjects, observer, config['image'])

    stars = pd.concat([celObjects['stars'], celObjects['planets']])


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
    stars['visible'] = (stars['response3'] > 10**-0.9 - 10**-(stars['vmag']*1.6/5))#0.15)#- vmag*0.03)

    thresh = (np.NaN,np.NaN,np.NaN)
    ##################################


    if args['--response']:        
        stars = pd.concat([stars, stars.apply(
                lambda s : findLocalMaxPos(lap, s.x, s.y, 2),
                axis=1)],axis=1)
        stars2 = stars.sort_values('vmag').drop_duplicates(subset=['maxX', 'maxY'], keep='first')

        fig = plt.figure(figsize=(4,3))
        ax = plt.subplot(111)
        stars2.plot.scatter(x='vmag',y='response3',ax=ax, logy=True, grid=True)
        ax.set_xlim((0,6))
        #ax.set_ylim((10,2e4))
        ax.set_ylim((1e-3,1))
        ax.set_ylabel('Kernel Response')
        ax.set_xlabel('Star Magnitude')
        
        font = {'size'   : 16, 'weight':'bold'}
        plt.show()
        #embed()
        #rc('font', **font)


    if args['--ratescan']:
        log.info('Doing ratescan')
        gradList = list()
        sobelList = list()
        lapList = list()

        response = np.logspace(-4,0,500)
        for resp in response:
            labeled, labelCnt = label(grad>resp)
            stars['visible'] = stars.response > resp
            gradList.append((calc_star_percentage(0, stars, -1), np.sum(grad > resp), labelCnt, sum(stars.visible)))
            labeled, labelCnt = label(sobel>resp)
            stars['visible'] = stars.response2 > resp
            sobelList.append((calc_star_percentage(0, stars, -1), np.sum(sobel > resp), labelCnt, sum(stars.visible)))
            labeled, labelCnt = label(lap>resp)
            stars['visible'] = stars.response3 > resp
            lapList.append((calc_star_percentage(0, stars, -1), np.sum(lap > resp), labelCnt, sum(stars.visible)))

        gradList = np.array(gradList)
        sobelList = np.array(sobelList)
        lapList = np.array(lapList)

        #minThresholds = [max(response[l[:,0]==1]) for l in (gradList, sobelList, lapList)]

        minThresholds = -np.array([np.argmax(gradList[::-1,0]), np.argmax(sobelList[::-1,0]), np.argmax(lapList[::-1,0])]) + len(response) -1
        clusters = (gradList[minThresholds[0],2], sobelList[minThresholds[1],2], lapList[minThresholds[2],2])
        thresh = (response[minThresholds[0]], response[minThresholds[1]],response[minThresholds[2]])
        fig = plt.figure(figsize=(19.2,10.8))
        ax1 = fig.add_subplot(111)
        plt.xscale('log')
        plt.grid()
        ax1.plot(response, sobelList[:,0], marker='x', c='blue', label='Sobel Kernel - Percent')
        ax1.plot(response, lapList[:,0], marker='x', c='red', label='LoG Kernel - Percent')
        ax1.plot(response, gradList[:,0], marker='x', c='green', label='Square Gradient - Percent')
        ax1.axvline(response[minThresholds[0]], color='green')
        ax1.axvline(response[minThresholds[1]], color='blue')
        ax1.axvline(response[minThresholds[2]], color='red')
        ax1.set_ylabel('')
        ax1.legend(loc='center left')

        ax2 = ax1.twinx()
        #ax2.plot(response, gradList[:,1], marker='o', c='green', label='Square Gradient - Pixcount')
        #ax2.plot(response, sobelList[:,1], marker='o', c='blue', label='Sobel Kernel - Pixcount')
        #ax2.plot(response, lapList[:,1], marker='o', c='red', label='LoG Kernel - Pixcount')
        ax2.plot(response, gradList[:,2], marker='s', c='green', label='Square Gradient - Clustercount')
        ax2.plot(response, sobelList[:,2], marker='s', c='blue', label='Sobel Kernel - Clustercount')
        ax2.plot(response, lapList[:,2], marker='s', c='red', label='LoG Kernel - Clustercount')
        ax2.axhline(gradList[minThresholds[0],2], color='green')
        ax2.axhline(sobelList[minThresholds[1],2], color='blue')
        ax2.axhline(lapList[minThresholds[2],2], color='red')
        ax2.legend(loc='upper right')
        ax2.set_ylim((0,16000))


        if args['-s']:
            plt.savefig('rateScan.pdf')
        if args['-v']:
            plt.show()
        #plt.close()
    if args['--cloudmap']:
        ax1 = plt.subplot(121)
        ax1.imshow(img, cmap='gray')
        stars.query('visible').plot.scatter(x='x',y='y', ax=ax1, color='green', grid=True)
        stars.query('not visible').plot.scatter(x='x',y='y', ax=ax1, color='red', grid=True)
        ax1.grid()

        ax2 = plt.subplot(122)
        cloud_map = calc_cloud_map(stars, 30, img.shape, weight=False)
        cloud_map = calc_cloud_map(stars, 30, img.shape, weight=True)
        ax2.imshow(cloud_map, cmap='gray_r')
        ax2.grid()
        #plt.imshow(cloud_map, cmap='gray')
        plt.show()
        embed()

    del images
    del grad
    del sobel
    del lap
    return stars, thresh
