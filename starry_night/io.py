import logging
from datetime import datetime, timedelta
import requests
from requests.exceptions import ReadTimeout
from skimage.color import rgb2gray
from io import BytesIO
from scipy.io import matlab
from astropy.io import fits
from skimage.io import imread
from os import stat
from hashlib import sha1
import re
import numpy as np


class TooEarlyError(Exception):
    pass


def get_last_modified(url, timeout):
    try:
        ret = requests.head(url, timeout=timeout)
        date = datetime.strptime(
            ret.headers['Last-Modified'],
            '%a, %d %b %Y %H:%M:%S GMT'
        )
    except (ReadTimeout, KeyError):
        log = logging.getLogger(__name__)
        log.error('Failed to retrieve timestamp from {} because website can not be reached.\nRetry later...'.format(url))
        date = None

    return date


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
        downloadImg.lastMod = datetime(1, 1, 1)
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
        img = hdulist[0].data + 2**15
        timestamp = datetime.strptime(hdulist[0].header['UTC'], '%Y/%m/%d %H:%M:%S')
    else:
        img = rgb2gray(imread(url, ))
        timestamp = get_last_modified(url, timeout=timeout)
        if not timestamp:
            return dict()

    return {
        'img': img,
        'timestamp': timestamp,
    }



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
    filetype = filepath.split('.')[-1]

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
                # config['properties']['timeFormat']
            )
        except (KeyError, ValueError, OSError, FileNotFoundError) as e:
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
                    config['properties']['timeFormat'],
                    )
                '''
                time = None
                for t in [['UTC', '%Y/%m/%d %H:%M:%S'] , ['TIMEUTC', '%Y-%m-%d %H:%M:%S']]:
                    try:
                        time = datetime.strptime(
                            hdulist[0].header[t[0]],
                            t[1],
                            )
                    except KeyError:
                        pass
                if time is None:
                    raise KeyError('Timestamp not found in file {}'.format(filepath))
            else:
                time = datetime.strptime(
                    filename,
                    config['properties']['timeFormat'],
                )
        except (ValueError, KeyError, OSError, FileNotFoundError) as e:
            log.error('Error parsing timestamp of {}: {}'.format(filepath, e))
            return

    else:
        # read normal image file
        try:
            img = imread(filepath, mode='L', as_grey=True)
        except (FileNotFoundError, OSError, ValueError) as e:
            log.error('Error reading file \'{}\': {}'.format(filename + '.' + filetype, e))
            return
        try:
            if fmt is None:
                time = datetime.strptime(filename, config['properties']['timeFormat'])
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
                    fmt = (config['properties']['timeFormat'] if fmt is None else fmt)
                    log.error('{},{}'.format(filename,filepath))
                    log.error('Unable to parse image time from filename. Maybe format string is wrong.')
                    return
    time += timedelta(minutes=float(config['properties']['timeOffset']))
    img = img.astype('float32') # needs to be float because we want to set some values NaN while cropping
    return dict({'img': img, 'timestamp': time})


def getMagicLidar(passwd):
    '''
    Return dict with data of the Magic lidar on LaPalma.
    passwd is the FACT password to access the data
    '''
    log = logging.getLogger(__name__)
    try:
        response = requests.get(
            'http://www.magic.iac.es/site/weather/protected/lidar_data.txt',
            auth=requests.auth.HTTPBasicAuth('FACT', passwd)
        )
    except ConnectionError as e:
        log.error('Connecting to lidar failed {}'.format(e))
        return
    if response.ok:
        dataString = response.content.decode('utf-8')
    else:
        log.error('Wrong lidar password')
        return
    values = list(map(float, re.findall("\d+\.\d+|\d+", dataString)))
    timestamp = datetime(*list(map(int, values[-3:])), *list(map(int, values[-6:-3])))

    # abort if last lidar update was more than 15min ago
    if datetime.utcnow() - timestamp > timedelta(minutes=15):
        return
    else:
        return {
            'timestamp': timestamp,
            'altitude': (90 - values[0]) / 180 * np.pi,
            'azimuth': values[1] / 180 * np.pi,
            'T3': values[3],
            'T6': values[5],
            'T9': values[7],
            'T12': values[9],
        }
