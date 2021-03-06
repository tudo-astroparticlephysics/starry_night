#!/usr/bin/env python
'''
Usage:
    starry_night -c <confFile> [<image>...] [options]
    starry_night -c <confFile> --daemon [options]
    starry_night -c <confFile> --daemon

Options:
                    If none
    <image>         Image file(s) or folder(s)
    -p <posFile>    File that contains Positions and timestamps to analyse
    -c Camera       Provide a camera config file or use one of these names: 'GTC', 'Magic' or 'CTA'
    -v              Visual output
    -s              Save output to files
    --kernel=<k>    Try different kernel sizes from 1 to <k> in steps of 5
    --function=<f>  Function used for calculation of response ('Grad','Sobel','LoG','All')
                    Using option '--ratescan' implies 'LoG'. [default: LoG]
    --vmag=<v>      Set custom magnitude limit.
    --threads=<t>   Set number of threads for multiprocessing. Can not spawn more threads than cores. [Default: 4]
    --moon          Do not correct changing exposure even if the moon shines.

    --cam           Display camera image
    --ratescan      Create ratescan-like plot of visibility vs kernel response threshold
    --response      Plot response vs Magnitude in log scale
    --cloudmap      Create cloud map of the sky
    --cloudtrack    Track and predict clouds as they move
    --single        Display information for every single image
    --airmass       Calculate airmass absorbtion
    --sql           Store results in SQL database
    --low-memory    Don't store results of each image in memory for final processing.
                    Use this option if you are not planning to merge the results because the
                    amount of files is too big or because you run this as a daemon at night.
    --daemon        Run as daemon during the night, no input possible.
    --version       Show version.
    --debug         debug it [default: False]
'''
from docopt import docopt
import pkg_resources
import logging
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
from multiprocessing import Pool, cpu_count
from scipy.optimize import curve_fit
from re import split
from getpass import getpass
from sqlalchemy import create_engine
import os

from sqlalchemy.exc import OperationalError, InternalError
from tables import HDF5ExtError
from functools import partial

from .. import skycam, cloud_tracker
from ..io import getImageDict, downloadImg, TooEarlyError
from ..config import load_config

#######################################################


def process_image(img, configList, data, args):
    image_dict = getImageDict(img, configList[0])
    return skycam.process_image(
        image_dict['img'],
        image_dict['timestamp'],
        data,
        configList,
        args, args['--function'],
    )


__version__ = pkg_resources.require('starry_night')[0].version
directory = os.path.join(os.environ['HOME'], '.starry_night')
if not os.path.exists(directory):
    os.makedirs(directory)

# create handler for file and console output
logfile_path = os.path.join(
    directory, 'starry_night-{}.log'.format(datetime.utcnow().isoformat())
)
logfile_handler = logging.FileHandler(filename=logfile_path, mode='w')
logfile_handler.setLevel(logging.ERROR)
logstream_handler = logging.StreamHandler()
formatter = logging.Formatter(
    fmt='%(asctime)s|%(levelname)s|%(name)s|%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
formatter.converter = time.gmtime  # use utc in log
logfile_handler.setFormatter(formatter)
logstream_handler.setFormatter(formatter)

# setup logging
logging.basicConfig(
    handlers=[
        logfile_handler,
        logstream_handler,
    ],
    level=logging.INFO,
)
logging.captureWarnings(True)


def main():
    args = docopt(doc=__doc__, version=__version__)

    if 'FACT_PASSWORD' in os.environ:
        passwd = os.environ['FACT_PASSWORD']
    else:
        passwd = False

    log = logging.getLogger('starry_night')
    log.info('starry_night started')
    log.info('version: {}'.format(__version__))

    if args['--debug']:
        log.info('DEBUG MODE - NOT FOR REGULAR USE')
        log.setLevel(logging.DEBUG)

    configList = load_config(args['-c'])
    # use first config file, the other ones are only needed if the camera was moved
    config = configList[0]

    # data is a container for global data
    # and variables which do belong neither to config nor to args.
    # data will be passed to all threads and is available for the analysis
    log.debug('Parsing Catalogue')
    data = skycam.celObjects_dict(config)

    if args['--vmag']:
        data['vmagLimit'] = args['--vmag']
    else:
        data['vmagLimit'] = float(config['analysis']['vmagLimit'])

    # prepare everything for sql connection
    if args['--sql']:
        log.info('Storing results in SQL Database.\nConnection: {}'.format(config['SQL']['connection']))

        if passwd:
            config['SQL']['connection'] = config['SQL']['connection'].format(passwd)
        else:
            log.info('Please enter password'.format(config['SQL']['connection']))
            config['SQL']['connection'] = config['SQL']['connection'].format(getpass())

        try:
            engine = create_engine(config['SQL']['connection'])
            if not engine.execute('SELECT VERSION();'):
                log.error('SQL connection failed! Aborting')
                sys.exit(1)
            del engine
        except (OperationalError, InternalError) as e:
            log.error(e)
            sys.exit(1)
    if not args['<image>']:
        if passwd:
            data['lidarpwd'] = passwd
        else:
            log.info('Please enter password to get acces to Magic lidar. Leave blank to skip.')
            data['lidarpwd'] = getpass()
    else:
        data['lidarpwd'] = None

    # read positioning file if any
    if args['-p']:
        log.info('Parsing positioning file.')
        try:
            data['positioning_file'] = pd.read_hdf(args['-p'], key='table')
        except (IOError, HDF5ExtError) as e:
            log.error(e.args[0])
            sys.exit(1)

    log.debug('Aquire Image(s)')
    results = list()

    if not args['<image>']:
        # download image(s) from URL
        while True:
            try:
                image_dict = downloadImg(
                    config['properties']['url'],
                    timeout=10,
                )
                log.debug('Download finished')
            except TooEarlyError as e:
                log.info('No new image available. Try again in 30 s.')
                time.sleep(30)
                continue
            except ConnectionError as e:
                log.error('Download of image failed. Try again in 30 s. {}'.format(e))
                time.sleep(30)
                continue

            results.append(skycam.process_image(
                image_dict['img'], image_dict['timestamp'], data, configList, args, args['--function']
            ))

            if not args['--daemon']:
                log.info('Do not download any more images because option --daemon is not set')
                break

    else:
        # use image(s) provided by the user and search for directories
        # remove entries that are no valid files
        log.info('Parsing file list. This might take a minute...')

        i = 0
        while len(args['<image>']) > i:
            if (i + 1) % 2000 == 0:
                print('Prepared {} images of {}'.format(i + 1, len(args['<image>'])))

            # expand content of directories
            if os.path.isdir(args['<image>'][i]):
                _dir = args['<image>'].pop(i)
                for root, dirs, files in os.walk(_dir):
                    for f in files:
                        args['<image>'].append(os.path.join(root, f))
            else:
                # remove invalid files
                if os.path.isfile(args['<image>'][i]):
                    i += 1
                else:
                    args['<image>'].pop(i)
        imgCount = len(args['<image>'])

        log.info('Processing {} images.'.format(imgCount))

        process_current_image = partial(
            process_image, data=data, configList=configList, args=args
        )

        # don't use multiprocessing in debug mode
        # process all images and store results
        if args['--debug'] or len(args['<image>']) == 1:
            for img in args['<image>']:
                results.append(process_current_image(img))
        else:
            threads = np.min([int(args['--threads']), cpu_count()])
            pool = Pool(processes=threads, maxtasksperchild=50)
            results = pool.map(process_current_image, args['<image>'])
            pool.close()
            pool.join()

    # drop all empty dictionaries (image processing was aborted because of high sun)
    # and merge the remaining files
    i = 0
    while i < len(results):
        if not results[i]:
            results.pop(i)
        else:
            i += 1

    imgCount = len(results)
    log.info('{} images were processed successfully.'.format(imgCount))

    #####################################################
    # no more processing if only a few images were processed successfully
    if len(args['<image>']) == 1:
        log.info('Done')
        sys.exit(0)

    if len(results) <= 5:
        log.info('Stop because only {} image(s) were processed. And we don\'t have enough data for further steps.'.format(len(results)))
        sys.exit(0)

    if args['--low-memory']:
        log.info('Option \'low-memory\' was activated. No data for further processing')
        sys.exit(0)

    star_list = list(map(lambda x: x['stars'], results))
    timestamp_list = list(map(lambda x: x['timestamp'], results))
    if args['--cloudtrack']:
        cloudmap_list = list(map(lambda x: x['cloudmap'], results)), timestamp_list

    # df = pd.concat(star_list, keys=timestamp_list, names=['date','HIP'])
    df = pd.concat(star_list)
    df = df.groupby('HIP').filter(lambda x: len(x.index) > 5)
    mean = df.groupby('HIP').mean()
    std = df.groupby('HIP').std()

    del results
    del star_list
    del timestamp_list

    df.sortlevel(inplace=True)

    #####################################################

    if args['--kernel']:
        if imgCount == 1:
            gr = df.groupby('HIP')
            res = list()
            for i, stars in gr:
                res.append(
                    stars.query(
                        'response_orig == {}'.format(stars.response_orig.max())
                    )[['HIP', 'vmag', 'kernel', 'response']].values
                )
            b = np.array(res)
            plt.plot(b[:, 0, 0], b[:, 0, 2])
            plt.show()

    if args['--airmass']:
        r = []
        grouped = df.groupby('HIP')
        # fit transmission for each star
        for group in grouped:
            # dont use stars for fit that do not span a wide distance
            if group[1].altitude.max() - group[1].altitude.min() < np.deg2rad(float(config['image']['openingangle']))/2:
                continue
            x = group[1].sort_values('altitude').altitude.values
            y = group[1].sort_values('altitude').response_orig.values
            try:
                popt, pcov = curve_fit(skycam.transmission3, x, y, p0=[0, 0.57])
            except RuntimeError:
                # skip group if fit does not converge
                # this might happen in a few cases where we have a small amount of data points with dark stars that fluctuate very much
                continue
            r.append((group[1].vmag.max(), popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1]), group[1]['B-V'].max(), group[1]['BTmag'].max(), group[1]['VTmag'].max() ))

        # do a fit of all transmissions
        r = np.array(r)
        coefficient = np.average(r[:,3], weights=1/r[:,4])
        print(coefficient)
        popt, pcov = curve_fit(skycam.lin, r[:,0], r[:,3], sigma=r[:,4], p0=[0,0])
        x = np.linspace(-1,7,5)
        y = skycam.lin(x, popt[0], popt[1])

        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        plt.errorbar(r[:,0], r[:,3], yerr=r[:,4], linestyle='', color='blue', marker='o', ms=6)
        plt.plot(x, y, color='red', label='linear regression')
        plt.xlim((-0.5,6.5))
        plt.ylim((-1,2))
        plt.xlabel('Star Magnitude')
        plt.ylabel('Air mass coefficient')
        '''
        plt.gca().text(0.98, 0.02, 'Weighted average: {}'.format(round(coefficient,4)),
            verticalalignment='bottom', horizontalalignment='right',
            transform=plt.gca().transAxes,
            backgroundcolor='white',
            color='black', fontsize=25,
        )
        '''
        plt.grid()
        plt.legend()
        if args['-v']:
            plt.show()
        plt.close('all')


        plt.figure(figsize=(16,9))
        plt.hist(r[:,3], bins=100, range=(-1,2))
        plt.xlabel('Air mass coefficient')
        plt.xlim((-0.5,1))
        print(popt, pcov)
        if args['-s']:
            plt.savefig('airmass_fit_{}'.format(config['properties']['name']), dpi=200)
        if args['-v']:
            plt.show()
        plt.close('all')


        # now show how good the fit looks like
        fig = plt.figure(figsize=(16,9))
        ax = fig.add_subplot(111)
        vmag_to_plot = np.arange(0, 6.1, 0.5)
        color = cm.jet(np.linspace(0,1,len(vmag_to_plot)))

        #remove stars that do not have a wide enough span of their altitude for plotting
        plot_df = grouped.filter(lambda x, df=df: x.altitude.max() > np.percentile(df.altitude, 95) and x.altitude.min() < np.percentile(df.altitude, 5))


        for i, vmag in enumerate(vmag_to_plot):
            c = color[i]
            # get star with magnitude closest to 'vmag'
            to_plot = plot_df.query(
                    'HIP == {}'.format(
                    max(plot_df.ix[(plot_df.vmag-vmag).abs().sort(inplace=False).index[0]].HIP)
                )
            ).sort_values('altitude')

            if abs(to_plot.vmag.max() - vmag) >= 0.25:
                # dont plot, if magnitude differs too much
                continue

            #maxVal = to_plot.query('altitude > {}'.format(np.percentile(to_plot.altitude, 90))).response_orig.mean()
            #popt, pcov = curve_fit(skycam.transmission2, to_plot.altitude.values, to_plot.response_orig.values, p0=[0,0.57])
            popt, pcov = curve_fit(skycam.transmission3, to_plot.altitude.values, to_plot.response_orig.values, p0=[0,0.57])

            ax.scatter(x=90-np.rad2deg(to_plot['altitude']), y=to_plot['response_orig'], c=c, label='mag = {}'.format(to_plot.vmag.max()))
            ax.plot(90-np.rad2deg(to_plot.altitude), skycam.transmission3(to_plot.altitude.values, popt[0], popt[1]), c=c)
            print(vmag, to_plot.HIP.max(), popt)

        ax.semilogy()
        ax.set_xlabel('Zenith angle')
        ax.set_ylabel('Response')
        ax.set_xlim((0,115))
        ax.set_ylim((47,42000))
        ax.grid()
        ax.legend(loc='upper right')
        plt.setp(plt.gca().get_legend().get_texts(), fontsize='22')
        plt.gca().text(0.98, 0.02, 'Fit result: $T(X)=e^{{-{}\cdot X}}$'.format(round(coefficient,4)),
            verticalalignment='bottom', horizontalalignment='right',
            transform=plt.gca().transAxes,
            backgroundcolor='white',
            color='black', fontsize=25,
        )
        if args['-s']:
            plt.savefig('airmass_absorbtion_{}'.format(config['properties']['name']), dpi=200)
        if args['-v']:
            plt.show()
        plt.close('all')
        del plot_df, to_plot, grouped


        fig = plt.figure(figsize=(16,9))
        x = np.linspace(0.1, 90.0, 500)

        # calculate absolut airmass for all 3 formulas
        y1 = np.log(skycam.transmission((90-x)/180*np.pi,1, coefficient))/-coefficient+1
        y2 = np.log(skycam.transmission2((90-x)/180*np.pi,1, coefficient))/-coefficient+1
        y3 = np.log(skycam.transmission3((90-x)/180*np.pi,1, coefficient))/-coefficient+0.7684

        ax = fig.add_subplot(111)
        ax.plot(x, y1, label='Planar')
        ax.plot(x, y2, label='Planar Corrected (Young 1967)')
        ax.plot(x, y3, label='Spherical elevated observer')
        ax.grid()
        ax.set_xlim((75,90))
        ax.set_ylim((-15,50))
        ax.set_xlabel('Zenith angle $[^\circ]$')
        ax.set_ylabel('Air mass')
        ax.legend(loc='best')
        if args['-s']:
            plt.savefig('airmass_curve', dpi=200)
        if args['-v']:
            plt.show()
        plt.close('all')



        fig = plt.figure(figsize=(16,9))
        x = np.linspace(0.1, 90.0, 200)

        y1 = skycam.transmission((90-x)/180*np.pi,1, coefficient)
        y2 = skycam.transmission2((90-x)/180*np.pi,1, coefficient)
        y3 = skycam.transmission3((90-x)/180*np.pi,1, coefficient)

        ax = fig.add_subplot(111)
        ax.plot(x, y1, label='Planar')
        ax.plot(x, y2, label='Planar Corrected')
        ax.plot(x, y3, label='Spherical elevated observer')
        ax.scatter(x=90-np.rad2deg(good_fit['altitude']), y=good_fit['response_orig']/good_fit.query('altitude > 1.3')['response_orig'].mean(), c='red', label='mag = {}'.format(good_fit.vmag.max()))
        ax.grid()
        ax.set_xlim((0,90))
        ax.set_ylim((-0.05,1.1))
        ax.set_xlabel('Zenith angle $[^\circ]$')
        ax.set_ylabel('Transmission')
        ax.legend(loc='upper right')
        ax.text(0.05, 0.01, 'Transmission coefficient = {}'.format(round(coefficient,4)),
            verticalalignment='bottom', horizontalalignment='left',
            transform=ax.transAxes,
            backgroundcolor='white',
            color='black', fontsize=15
        )
        if args['-s']:
            plt.savefig('transmission_curve', dpi=200)
        if args['-v']:
            plt.show()
        plt.close('all')


    if args['--response']:
        fig = plt.figure(figsize=(16,9))
        ax = plt.subplot(111)
        ax.semilogy()
        if args['-c'] == 'GTC':
            ax.axhspan(ymin=11**2/255**2, ymax=13**2/255**2, color='red', alpha=0.5, label='Old threshold - Gradient')
        mean.plot.scatter(x='vmag',y='response', yerr=std['response'].values, color='blue',
                ax=ax, logy=True, grid=True, vmin=0, vmax=1, label='{} Response'.format(args['--function']))
        #mean.plot.scatter(x='vmag',y='response_grad', yerr=std['response_grad'].values, color='red', ax=ax, logy=True, grid=True, label='Gradient Response')
        #ax.set_xlim((-1, max(mean['vmag'])+0.5))
        ax.set_ylim(bottom=10**(np.log10(np.nanpercentile(mean.response.values,10.0))//1-1),
            top=10**(np.log10(np.nanpercentile(mean.response.values,99.9))//1+1))
        x = np.linspace(-5+mean.vmag.min(), mean.vmag.max()+5, 20)
        lim = (list(map(float, split('\\s*,\\s*', config['analysis']['visibleupperlimit']))), list(map(float, split('\\s*,\\s*', config['analysis']['visiblelowerlimit']))))
        y1 = 10**(x*lim[1][0] + lim[1][1])
        y2 = 10**(x*lim[0][0] + lim[0][1])
        ax.plot(x, y1, c='red', label='lower limit')
        ax.plot(x, y2, c='green', label='upper limit')
        ax.legend(loc='best')
        ax.set_ylabel('Kernel Response')
        ax.set_xlabel('Star Magnitude')
        plt.show()
        if args['-s']:
            plt.savefig('response_{}_mean.png'.format(args['--function']))
        if args['-v']:
            plt.show()
        plt.close('all')

    if args['--cloudtrack']:
        ct = cloud_tracker.CloudTracker(config)
        for cmap,timestamp in zip(cloudmap_list[0], cloudmap_list[1]):
            ct.update(cmap,timestamp)


        print('Tracking done')


    '''
    def logf(x, m, b):
        return 10**(m*x+b)

    sys.exit(0)
    fit_stars = df.query('0 < vmag < {}'.format(float(data['vmagLimit'])))
    popt, pcov = curve_fit(logf, fit_stars.vmag.values, fit_stars.response.values, sigma=1/fit_stars.vmag.values, p0=(-0.2, 2))
    x = np.linspace(-3+fit_stars.vmag.min(), fit_stars.vmag.max(), 20)
    y = logf(x, popt[0], popt[1]-0.3)
    lim = (split('\\s*,\\s*', config['analysis']['visibleupperlimit']), split('\\s*,\\s*', config['image']['visiblelowerlimit']))
    y1 = 10**(x*float(lim[1][0]) + float(lim[1][1]))
    y2 = 10**(x*float(lim[0][0]) + float(lim[0][1]))
    '''

    '''
    fig = plt.figure(figsize=(16,9))
    ax = plt.subplot(111)
    #if args['--function'] == 'Grad':
    #ax.axhspan(ymin=11**2/255**2, ymax=13**2/255**2, color='red', alpha=0.5, label='old threshold range')
    df.loc[df.index.get_level_values(0).unique()[0]].plot.scatter(
            x='vmag',y='response', color='green', ax=ax, logy=True, grid=True, label='Kernel Response - no Moon')
    df.loc[df.index.get_level_values(0).unique()[1]].plot.scatter(
            x='vmag',y='response', color='red', ax=ax, logy=True, grid=True, label='Kernel Response - Moon')
    df.loc[df.index.get_level_values(0).unique()[2]].plot.scatter(
            x='vmag',y='response', color='k', ax=ax, logy=True, grid=True, label='Kernel Response - cloud')
    ax.set_ylim((1e-5,1))
    plt.show()
    '''


if __name__ == '__main__':
    try:
        main()
    except (KeyboardInterrupt, SystemExit) as e:
        logging.getLogger('starry_night').info('Exit')
