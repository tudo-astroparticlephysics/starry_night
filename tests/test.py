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

    b = skycam.isInRange(star1, star2, rng=5)
    eq_(b, True, 'On Range failed')

    b = skycam.isInRange(star1, star2, rng=4.999)
    eq_(b, False, 'Outside Range failed')

    b = skycam.isInRange(star1, star2, rng=5.001)
    eq_(b, True, 'In Range failed')
