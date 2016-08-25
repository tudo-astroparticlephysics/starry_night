import numpy as np
import matplotlib as plt
from skimage.segmentation import active_contour
from IPython import embed
from starry_night.skycam import r2theta, theta2r

class Cloud:
    def __init__(self, _id):
        self.id = _id
        self.height = None
        self.position = (None, None)
        self.speed = None
        self.direction = None
        self.covered = True     #if False, then this is not a cloud but a free spot in a covered sky

    def print_info(self):
        print('''Cloud {}:\nID:{}\nHeight:{}\nPosition:{}\nSpeed:{}\n
                Direction:{}\nCovered:{}'''.format(
                    self.id,
                    self.height,
                    self.position,
                    self.speed,
                    self.direction,
                    self.covered,
                    )
                )

class CloudTracker:
    maxMaps = 5
    atmosphere_height = 9
    def __init__(self, conf):
        self.maps = []
        self.transMaps= []
        self.maxID = 0
        self.wind_speed = None
        self.wind_direction = None
        self.config = conf

    def __add_cloud(self, cloudMap):
        self.maps.append(cloudMap)
        self.transMaps.append(self.__transformMap(cloudMap))
        self.maxID += 1
        if (len(self.maps) > self.maxMaps):
            self.maps.pop(0)
            self.transMaps.pop(0)

    def __transformMap(self, cloudMap):
        '''
        Project fish eye lens image onto flat atmosphere
        transformed = np.zeros(1000,1000)
        X,Y = np.meshgrid(range(int(self.config['zenith_X'])), range(int(self.config['zenith_X'])))

        # coordinates in radius and azimuth
        r = np.sqrt((Y-float(self.config['zenith_Y']))**2 + (X-float(self.config['zenith_X']))**2)
        phi = np.arctan2(Y-float(self.config['zenith_Y']),X-float(self.config['zenith_X']))

        theta, mask = r2theta(r, float(self.config['radius']), self.config['angleProjection'], mask=True)
        r = np.tan(theta) * np.max(r)
        '''
        transformed = np.zeros([1000,1000])
        X,Y = np.meshgrid(range(transformed.shape[0]), range(transformed.shape[1]))
        pixel_dist = np.sqrt((Y-transformed.shape[1]//2)**2 + (X-transformed.shape[0]//2)**2)
        theta = np.arctan(pixel_dist/np.max(pixel_dist)*4)
        phi = np.arctan2(Y-transformed.shape[1]//2,X-transformed.shape[0]//2)
        r = theta2r(theta, float(self.config['radius']), self.config['angleProjection'])
        x,y = r * np.sin(phi) + cloudMap.shape[1]//2, r*np.cos(phi) + cloudMap.shape[0]//2
        return cloudMap[x.astype(int),y.astype(int)]

    def __remove_cloud(self, _id):
        try:
            self.maps.pop(_id)
        except:
            raise

    def print_clouds(self):
        print('Currently {} images loaded'.format(len(self.maps)))

    # update detects clouds in cloudmap and updates old cloud positions
    def update(self, cloudMap):
        transformed = 3
        self.__transformMap(cloudMap)
        self.__add_cloud(cloudMap)
        # detect clouds




