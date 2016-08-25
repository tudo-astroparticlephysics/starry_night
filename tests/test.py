from starry_night import skycam
from nose.tools import eq_
import numpy as np
import pandas as pd

def test_findLocalMaxPos():
    img = np.zeros((480,640))
    img[10,30] = 1

    # find max value if we are on top of it
    pos = skycam.findLocalMaxPos(img, 30,10,1)
    eq_((pos.maxX,pos.maxY), (30,10), 'Right on Maximum')

    # return center, if max is not in range
    pos = skycam.findLocalMaxPos(img, 3,12,1)
    eq_((pos.maxX,pos.maxY), (3,12), 'Max not in range')

    # radius too big should not be a problem
    pos = skycam.findLocalMaxPos(img, 0,0,300)
    eq_((pos.maxX,pos.maxY), (30,10), 'Array overflow1')

    # nans in image should not be a problem...
    img = np.ones((480,640))*np.nan
    img[10,30] = 1
    pos = skycam.findLocalMaxPos(img, 0,0,300)
    eq_((pos.maxX,pos.maxY), (30,10), 'Nan in range')

    # ... even if there is no max in range
    pos = skycam.findLocalMaxPos(img, 0,0,3)
    eq_((pos.maxX,pos.maxY), (0,0), 'Nan outside range. x,y:{}'.format((pos.maxX,pos.maxY)))

def test_isInRange():
    star1 = pd.Series({'x':2, 'y':3})
    star2 = pd.Series({'x':6, 'y':6})

    b = skycam.isInRange(star1, star2, rng=5, unit='pixel')
    eq_(b, True, 'On Range failed')

    b = skycam.isInRange(star1, star2, rng=4.999, unit='pixel')
    eq_(b, False, 'Outside Range failed')

    b = skycam.isInRange(star1, star2, rng=5.001, unit='pixel')
    eq_(b, True, 'In Range failed')

    star1 = pd.Series({'ra':0, 'dec':0})
    star2 = pd.Series({'ra':0, 'dec':5/180*np.pi})

    b = skycam.isInRange(star1, star2, rng=5, unit='deg')
    eq_(b, True, 'On Range failed')

    b = skycam.isInRange(star1, star2, rng=4.999, unit='deg')
    eq_(b, False, 'Outside Range failed')

    b = skycam.isInRange(star1, star2, rng=5.001, unit='deg')
    eq_(b, True, 'In Range failed')

def test_getBlobsize():
    image = np.ones((51,51))
    b = skycam.getBlobsize(image, 0.5, limit=10)
    eq_(b, 10, 'Exeed limit failed: {}'.format(b))

    b = skycam.getBlobsize(image, 0.5)
    eq_(b, 51*51, 'Total blob failed')

    image[25,25] = 20
    image[26,24] = 20
    image[27,24] = 20
    b = skycam.getBlobsize(image, 2)
    eq_(b, 3, 'Diagonal blob failed: {}'.format(b))

    image[:,22:26] = 20
    b = skycam.getBlobsize(image, 2)
    eq_(b, 204, 'Blob at border failed: {}'.format(b))

    image[25,24]=np.NaN
    image[26,25]=-np.NaN
    b = skycam.getBlobsize(image, 2)
    eq_(b, 202, 'Blob at border failed: {}'.format(b))

def test_invert_radius_theta():
    theta,r = 0.2, 5
    b = skycam.r2theta(skycam.theta2r(theta, r), r)
    eq_(round(b,4), 0.2, 'Invert radius 1 failed: {}'.format(b))
    
    theta,r = 0, 5
    b = skycam.r2theta(skycam.theta2r(theta, r, how='notLin'), r, how='notLin')
    eq_(b, 0, 'Invert radius 2 failed: {}'.format(b))

    theta,r = np.pi, 5
    b = skycam.r2theta(skycam.theta2r(theta, r, how='notLin'), r, how='notLin')
    eq_(round(b,4), round(np.pi,4), 'Invert radius 3 failed: {}'.format(b))
