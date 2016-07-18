import numpy as np
from numpy import sin, cos, tan, arctan2, arcsin, pi
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.inset_locator import inset_axes

import pandas as pd
import ephem
import sys

from scipy.io import matlab
from io import BytesIO
from skimage.io import imread
from skimage.color import rgb2gray

from astropy.io import fits
from datetime import datetime, timedelta
from time import sleep

from pkg_resources import resource_filename
from os.path import join
import requests
import logging

from re import split
import skimage.filters
from scipy.ndimage.measurements import label
from IPython import embed


def downloadImg(url, *args, **kwargs):
    if url.split('.')[-1]== 'mat':
        ret = requests.get(url)
        data = matlab.loadmat(BytesIO(ret.content))
        imgList = list()
        timeList = list()
        outList = list()
        for d in list(data.values()):
            try:
                if d.shape[0] > 100 and d.shape[1] > 100:
                    imgList.append(d)
            except AttributeError:
                pass
            try:
                timeList.append(datetime.strptime(d[0], '%Y/%m/%d %H:%M:%S'))
            except (IndexError, TypeError, ValueError):
                pass
        
        for img, t in zip(imgList,timeList):
            outList.append(dict({
                'img' : img,
                'timestamp' : t,
                })
            )

        return outList
    return [rgb2gray(imread(url, ))]


def get_last_modified(url, *args, **kwargs):
    ret = requests.head(url, *args, **kwargs)
    date = datetime.strptime(
        ret.headers['Last-Modified'],
        '%a, %d %b %Y %H:%M:%S GMT'
    )
    return date


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
    catalogue = resource_filename('starry_night', 'data/catalogue_10vmag_1degFilter.csv')
    stars = pd.read_csv(
        catalogue,
        sep=',',
        comment='#',
        header=0,
        skipinitialspace=False,
        index_col=0,
    )
    #stars = stars.to_numeric()

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


def update_star_position(celestialObjects, observer, cam, crop):
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
        data = {
            'ra': float(sol_object.a_ra),
            'dec': float(sol_object.a_dec),
            'gLon': float(galactic.lon),
            'gLat': float(galactic.lat),
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
    stars['angleToMoon'] = np.arccos(np.sin(stars.altitude.values)*
        np.sin(moon.alt) + np.cos(stars.altitude.values)*np.cos(moon.alt)*
        np.cos((stars.azimuth.values - moon.az)))
    planets['angleToMoon'] = np.arccos(np.sin(planets.altitude.values)*
        np.sin(moon.alt) + np.cos(planets.altitude.values)*np.cos(moon.alt)*
        np.cos((planets.azimuth.values - moon.az)))

    # remove stars and planets that are too close to moon
    stars.query('angleToMoon > {}'.format(np.deg2rad(float(cam['minAngleToMoon']))), inplace=True)
    planets.query('angleToMoon > {}'.format(np.deg2rad(float(cam['minAngleToMoon']))), inplace=True)


    # calculate x and y position
    log.debug('Calculate x and y')
    stars['x'], stars['y'] = horizontal2image(stars.azimuth, stars.altitude, cam=cam)
    planets['x'], planets['y'] = horizontal2image(planets.azimuth, planets.altitude, cam=cam)
    moonData['x'], moonData['y'] = horizontal2image(moonData['azimuth'], moonData['altitude'], cam=cam)
    sunData['x'], sunData['y'] = horizontal2image(sunData['azimuth'], sunData['altitude'], cam=cam)
    stars = stars[stars.apply(lambda x, crop=crop: ~crop[int(x['y']), int(x['x'])], axis=1)]
    planets = planets[planets.apply(lambda x, crop=crop: ~crop[int(x['y']), int(x['x'])], axis=1)]

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

    # get image type from filename
    filename = filepath.split('/')[-1].split('.')[0]
    filetype= filepath.split('.')[-1]

    # read mat file
    if filetype == 'mat':
        data = matlab.loadmat(filepath)
        img = data['pic1']
        time = datetime.strptime(
            data['UTC1'][0], '%Y/%m/%d %H:%M:%S'
        )

    # read fits file
    elif (filetype == 'fits') or (filetype == 'gz'):
        hdulist = fits.open(filepath, ignore_missing_end=True)
        img = hdulist[0].data
        time = datetime.strptime(
            hdulist[0].header['TIMEUTC'],
            '%Y-%m-%d %H:%M:%S'
        )
    else:
        # read normal image file
        try:
            img = imread(filepath, mode='L', as_grey=True)
        except (FileNotFoundError, OSError):
            log.error('File \'{}\' not found. Or filetype invalid'.format(filename))
            sys.exit(1)
        try:
            if fmt is None:
                time = datetime.strptime(filename, config['image']['timeformat'])
            else:
                time = datetime.strptime(filename, fmt)
        
        except ValueError:
            fmt = (config['image']['timeformat'] if fmt is None else fmt)
            log.error('{},{}'.format(filename,filepath))
            log.error('Unable to parse image time from filename. Maybe format is wrong: {}'.format(fmt))
            raise
            sys.exit(1)
    time += timedelta(minutes=float(config['properties']['timeoffset']))
    return dict({'img': img, 'timestamp': time})

    
def get_crop_mask(img, crop):
    '''
    crop is dictionary with cropping information
    returns a boolean array in size of img: False got cropped; True not cropped 
    '''
    if crop is not None:
        try:
            x = split('\\s*,\\s*', crop['crop_x'])
            y = split('\\s*,\\s*', crop['crop_y'])
            r = split('\\s*,\\s*', crop['crop_radius'])
            inside = split('\\s*,\\s*', crop['crop_deleteinside'])
            nrows,ncols = img.shape
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
    '''
    Returns true or false for each star in stars if distance between star and position<rng

    If unit= "pixel" position and star must have attribute .x and .y in pixel and rng is pixel distance
    If unit= "deg" position and star must have attribute .ra and .dec in degree 0<360 and rng is degree
    '''
    if rng < 0:
        raise ValueError
    
    if unit == 'pixel':
        if type(position) == dict:
            return ((position.x - stars.x)**2 + (position.y - stars.y)**2 <= rng**2)
        else:
            return ((position[0] - stars.x)**2 + (position[1] - stars.y)**2 <= rng**2)
    elif unit == 'deg':
        if type(position) == dict:
            ra1 = position['ra']
            dec1 = position['dec']
        else:
            ra1 = position[0]
            dec1 = position[1]

        deltaDeg = 2*np.arcsin(np.sqrt(np.sin((dec1-stars.dec)/2)**2 + np.cos(dec1)*np.cos(stars.dec)*np.sin((ra1-stars.ra)/2)**2))
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
            vis = np.sum(np.pow(100**(1/5),starsInRange.query('visible == 1').vmag.values))
            notVis = np.sum(np.pow(100**(1/5),starsInRange.query('visible == 0').vmag.values))
            percentage = vis/(vis+notVis)
        else:
            percentage = len(starsInRange.query('visible == 1').index)/len(starsInRange.index)
    except ZeroDivisionError:
        #log = logging.getLogger(__name__)
        #log.warning('No stars in range to calc percentage. Returning -1.')
        percentage = -1

    return percentage


def calc_cloud_map(stars, rng, img_shape, weight=False):
    '''
    Input:  stars - pandas dataframe
            rng - sigma of gaussian kernel (integer)
            img_shape - size of cloudiness map in pixel (tuple)
            weight - use magnitude as weight or not (boolean)
    Returns: Cloudines map of the sky. 1=cloud, 0=clear sky

    Cloudiness is percentage of visible stars in local area. Stars get weighted by
    distance (gaussian) and star magnitude 2.5^magnitude.
    Instead of a computationally expensive for-loop we use two 2D histograms of the stars (weighted)
    and convolve them with an gaussian kernel resulting in some kind of 'density map'.
    Division of both maps yields the desired cloudines map.
    '''
    nrows = ncols = rng*2 +1
    row, col = np.ogrid[:nrows, :ncols]
    disk_mask = np.zeros((nrows, ncols))
    disk_mask[((row - rng)**2 + (col - rng)**2 < int(rng)**2)] = 1
    if weight:
        scattered_stars_visible,_,_ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=stars.visible.values * 2.5**-stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        scattered_stars,_,_ = np.histogram2d(stars.y.values, stars.x.values, weights=np.ones(len(stars.index)) * 2.5**-stars.vmag.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        density_visible = skimage.filters.gaussian(scattered_stars_visible, rng)
        density_all = skimage.filters.gaussian(scattered_stars, rng)
    else:
        scattered_stars_visible,_,_ = np.histogram2d(x=stars.y.values, y=stars.x.values, weights=stars.visible.values, bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        scattered_stars,_,_ = np.histogram2d(stars.y.values, stars.x.values, weights=np.ones(len(stars.index)), bins=img_shape, range=[[0,img_shape[0]],[0,img_shape[1]]])
        density_visible = skimage.filters.gaussian(scattered_stars_visible, rng, mode='mirror')
        density_all = skimage.filters.gaussian(scattered_stars, rng, mode='mirror')
    with np.errstate(divide='ignore',invalid='ignore'):
        cloud_map = np.true_divide(density_visible, density_all)
        cloud_map[~np.isfinite(cloud_map)] = 0
    return 1-cloud_map



def filter_catalogue(catalogue, rng):
    '''
    Loop through all possible pairs of stars and remove less bright star if distance is < rng

    Input:  catalogue - Pandas dataframe (ra and dec in degree)
            rng - Min distance between stars in degree

    Returns: List of indexes that remain in catalogue
    '''
    log = logging.getLogger(__name__)
    try:
        c = catalogue.sort_values('vmag', ascending=True)
        reference = np.deg2rad(c[['ra','dec']].values)
        index = c.index
    except KeyError:
        log.error('Key not found. Please check that your catalogue is labeled correctly')
        raise
    
    i1 = 0 #star index that is used as filter base
    while i1 < len(reference)-1:
        print('Items left: {}/{}'.format(i1,len(reference)-1))
        deltaDeg = np.rad2deg(2*np.arcsin(np.sqrt(np.sin((reference[i1,1]-reference[:,1])/2)**2 + np.cos(reference[i1,1])*np.cos(reference[:,1])*np.sin((reference[i1,0]-reference[:,0])/2)**2)))
        keep = deltaDeg > rng
        keep[:i1+1] = True #don't remove stars that already passed the filter
        reference = reference[keep]
        index = index[keep]
        i1+=1
    return index


def process_image(images, celestialObjects, config, args):
    '''
    This function applies all neccessary calculations to an image and returns the results.
    Use it in the main loop!
    '''
    log = logging.getLogger(__name__)


    log.info('Processing image take at: {}'.format(images['timestamp']))
    observer = obs_setup(config['properties'])
    observer.date = images['timestamp']

    # stop processing if sun is too high or config file does not match
    if images['img'].shape[1]  != int(config['image']['resolution'].split(',')[0]) or images['img'].shape[0]  != int(config['image']['resolution'].split(',')[1]):
        log.error('Resolution does not match: {}!={}. Wrong config file?'.format(c_res, i_res))
        return
    sun = ephem.Sun()
    sun.compute(observer)
    if np.rad2deg(sun.alt) > -10:
        log.info('Sun too high: {}° above horizon. We start at -10°, current time: {}'.format(np.round(np.rad2deg(sun.alt),2), images['timestamp']))
        return

    # create cropping array to mask unneccessary image regions.
    img = images['img']
    crop_mask = get_crop_mask(img, config['crop'])

    # update celestial objects
    celObjects = update_star_position(celestialObjects, observer, config['image'], crop_mask)
    all_stars = pd.concat([celObjects['stars'], celObjects['planets']])


    # calculate response of stars
    if args['--kernel']:
        kernelSize = np.arange(1, int(args['--kernel'])+1, 5)
    else:
        kernelSize = [float(config['image']['kernelsize'])]
    kernelResults = list()

    for k in kernelSize:
        log.debug('Apply image filters. Kernelsize = {}'.format(k))

        # work on a copy if this is a loop
        if args['--kernel']:
            stars = all_stars.copy()
        else:
            stars = all_stars
         
        gauss = skimage.filters.gaussian(img, sigma=k)

        # chose the response function
        if args['--function'] == 'All' or args['--ratescan']:
            grad = (img - np.roll(img, 1, axis=0)).clip(min=0)**2 + (img - np.roll(img, 1, axis=1)).clip(min=0)**2
            sobel = skimage.filters.sobel(img).clip(min=0)
            lap = skimage.filters.laplace(gauss, ksize=3).clip(min=0)
            grad[crop_mask] = 0
            sobel[crop_mask] = 0
            lap[crop_mask] = 0
            images['grad'] = grad
            images['sobel'] = sobel
            images['lap'] = lap
            resp = lap
        elif args['--function'] == 'DoG':
            resp = skimage.filters.gaussian(img, sigma=k) - skimage.filters.gaussian(img, sigma=1.6*k)
        elif args['--function'] == 'LoG':
            resp = skimage.filters.laplace(gauss, ksize=3).clip(min=0)
        elif args['--function'] == 'Grad':
            resp = ((img - np.roll(img, 1, axis=0)).clip(min=0))**2 + ((img - np.roll(img, 1, axis=1)).clip(min=0))**2
        elif args['--function'] == 'Sobel':
            resp = skimage.filters.sobel(img).clip(min=0)
        else:
            log.error('Function name: \'{}\' is unknown!'.format(args['--function']))
            sys.exit(1)
        resp[crop_mask] = 0
        images['response'] = resp


        # tolerance is max distance between actual star position and expected star position
        # this should be a little smaller than 1° because this is the minimum distance
        # between 2 catalogue stars (catalogue was filtered for this)
        tolerance = int((float(config['image']['radius'])/90-1)/2)-3 
        log.debug('Calculate Filter response')
        
        # calculate x and y position where response has its max value (search within 'tolerance' range)
        stars = pd.concat([stars.drop(['maxX','maxY'], errors='ignore', axis=1), stars.apply(
                lambda s : findLocalMaxPos(resp, s.x, s.y, tolerance),
                axis=1)], axis=1
        )

        # drop stars that got mistaken for a brighter neighboor
        stars = stars.sort_values('vmag').drop_duplicates(subset=['maxX', 'maxY'], keep='first')

        #calculate response
        stars['response'] = stars.apply(lambda s : findLocalMaxValue(resp, s.x, s.y, tolerance), axis=1)

        # drop stars that were not found at all, because response=0 interferes with log-plot
        stars.query('response > 1e-100', inplace=True)
        
        if args['--function'] == 'All' or args['--ratescan']:
            stars['response_grad'] = stars.apply(lambda s : findLocalMaxValue(grad, s.x, s.y, tolerance), axis=1)
            stars['response_sobel'] = stars.apply(lambda s : findLocalMaxValue(sobel, s.x, s.y, tolerance), axis=1)
        lim = (split('\\s*,\\s*', config['image']['visibleupperlimit']), split('\\s*,\\s*', config['image']['visiblelowerlimit']))

        # calculate visibility percentage
        # if response > visibleUpperLimit -> visible=1
        # if response < visibleUpperLimit -> visible=0
        # if in between: scale linear
        stars['visible'] = np.minimum(
                1,
                np.maximum(
                    0,
                    (np.log10(stars['response']) - (stars['vmag']*float(lim[1][0]) + float(lim[1][1]))) / 
                    ((stars['vmag']*float(lim[0][0]) + float(lim[0][1])) - (stars['vmag']*float(lim[1][0]) + float(lim[1][1])))
                    )
                )
        # set visible = 0 for all magnitudes where upperLimit < lowerLimit
        stars.loc[stars.vmag.values > (float(lim[1][1]) - float(lim[0][1])) / (float(lim[0][0]) - float(lim[1][0])), 'visible'] = 0

        # append results
        kernelResults.append(stars)



    ##################################
    try:
        df = pd.concat(kernelResults, keys=kernelSize)
    except ValueError:
        df = kernelResults[0]

    if args['--cam']:
        fig = plt.figure()
        #k = 1
        #resp = skimage.filters.gaussian(img, sigma=k) - skimage.filters.gaussian(img, sigma=6*k)
        img = resp
        vmin = np.nanpercentile(img, 0.5)
        vmax = np.nanpercentile(img, 99.)
        plt.imshow(img,vmin=vmin,vmax=vmax)
        stars.plot.scatter(x='x',y='y', ax=plt.gca(), c=stars.visible.values, cmap = plt.cm.RdYlGn, vmin=0, vmax=1, grid=True)
        plt.colorbar()
        plt.show()

        embed()

        if args['-s']:
            plt.savefig('cam_image_{}.pdf'.format(images['timestamp'].isoformat()))
        if args['-v']:
            plt.show()
        plt.close('all')

    if args['--single']:
        if args['--response']:
            fig = plt.figure(figsize=(16,9))
            ax = plt.subplot(111)
            ax.semilogy()

            # draw visibility limits
            x = np.linspace(-5+stars.vmag.min(), stars.vmag.max()+5, 20)
            lim = (split('\\s*,\\s*', config['image']['visibleupperlimit']), split('\\s*,\\s*', config['image']['visiblelowerlimit']))
            y1 = 10**(x*float(lim[1][0]) + float(lim[1][1]))
            y2 = 10**(x*float(lim[0][0]) + float(lim[0][1]))
            ax.plot(x, y1, c='red', label='lower limit')
            ax.plot(x, y2, c='green', label='upper limit')

            stars.query('vmag<4').plot.scatter(x='vmag', y='response', ax=ax, logy=True, c=stars.query('vmag<4').visible.values, cmap = plt.cm.RdYlGn, grid=True, vmin=0, vmax=1, label='Kernel Response')
            ax.set_xlim((-1, max(stars['vmag'])+0.5))
            #ax.set_ylim(bottom=10**(np.log10(np.nanpercentile(stars.response.values,10.0))//1-1),
            #    top=10**(np.log10(np.nanpercentile(stars.response.values,99.9))//1+1))
            ax.set_ylim((1e-1,1e3))
            ax.set_ylabel('Kernel Response')
            ax.set_xlabel('Star Magnitude')
            if args['-c'] == 'GTC':
                if args['--function'] == 'Grad':
                    ax.axhspan(ymin=11**2/255**2, ymax=13**2/255**2, color='red', alpha=0.5, label='old threshold range')
                if args['--function'] == 'LoG':
                    ax.axhline(0.015, color='red', label='Estimated threshold')
                ax.axvline(4.5, color='green', label='Magnitude lower limit')

            # show camera image in a subplot
            ax_in= inset_axes(ax,
                    width='30%',
                    height='40%',
                    loc=3)
            vmin = np.nanpercentile(img, 0.5)
            vmax = np.nanpercentile(img, 99.)
            ax_in.imshow(img,cmap='gray',vmin=vmin,vmax=vmax)
            ax_in.get_xaxis().set_visible(False)
            ax_in.get_yaxis().set_visible(False)
            
            ax.legend(loc='best')
            if args['-s']:
                plt.savefig('response_{}_{}.pdf'.format(args['--function'], images['timestamp'].isoformat()))
            if args['-v']:
                plt.show()
            plt.close('all')

        if args['--ratescan']:
            log.info('Doing ratescan')
            gradList = list()
            sobelList = list()
            lapList = list()

            response = np.logspace(-4.5,-0.5,200)
            for resp in response:
                labeled, labelCnt = label(grad>resp)
                stars['visible'] = stars.response_grad > resp
                gradList.append((calc_star_percentage(0, stars, -1), np.sum(grad > resp), labelCnt, sum(stars.visible)))
                labeled, labelCnt = label(sobel>resp)
                stars['visible'] = stars.response_sobel > resp
                sobelList.append((calc_star_percentage(0, stars, -1), np.sum(sobel > resp), labelCnt, sum(stars.visible)))
                labeled, labelCnt = label(lap>resp)
                stars['visible'] = stars.response > resp
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
            ax1.axvline(14**2/255**2, color='black', label='old threshold')
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
            ax2.set_xlim((min(response), max(response)))
            ax2.set_ylim((0,16000))
            if args['-v']:
                plt.show()
            if args['-s']:
                plt.savefig('rateScan.pdf')
            plt.close('all')
            del grad
            del sobel
            del lap

    if args['--cloudmap']:
        log.debug('Calculating cloud map')
        ax1 = plt.subplot(121)
        vmin = np.nanpercentile(img, 5.5)
        vmax = np.nanpercentile(img, 99.9)
        ax1.imshow(img, vmin=vmin, vmax=vmax, cmap='gray', interpolation='none')
        ax1.grid()

        ax2 = plt.subplot(122)
        #cloud_map1 = calc_cloud_map(stars, img.shape[1]//30, img.shape, weight=False)
        #cloud_map1[crop_mask] = 0
        cloud_map2 = calc_cloud_map(stars, img.shape[1]//30, img.shape, weight=True)
        cloud_map2[crop_mask] = 0
        ax2.imshow(cloud_map2, cmap='gray_r',vmin=0,vmax=1)
        ax2.grid()
        if args['-s']:
            plt.savefig('cloudMap_{}.png'.format(images['timestamp'].isoformat()))
        plt.close('all')

        plt.show()

    timestamp = images['timestamp']
    del images
    try:
        return stars, timestamp, response, thresh
    except UnboundLocalError:
        return stars, timestamp, (np.NaN,np.NaN,np.NaN)
