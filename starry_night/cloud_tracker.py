import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import logging
from skimage.segmentation import active_contour
from IPython import embed
from starry_night.skycam import r2theta, theta2r
from starry_night import skycam

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
    max_maps = 5
    atmosphere_height = 9
    def __init__(self, config):
        self.log = logging.getLogger(__name__)
        self.maps = []
        self.trans_maps= []
        self.pred_map = []
        self.maxID = 0
        self.wind_speed = None
        self.wind_direction = []
        self.crop = None
        self.trans_crop = None
        self.config = config

    def __add_cloud(self, cloudMap):
        if not self.maps:
            self.crop = skycam.get_crop_mask(cloudMap, self.config['crop'])
            self.trans_crop = self.__transform_map(self.crop)
        cloudMap[self.crop] = np.NaN
        self.maps.append(cloudMap)
        self.trans_maps.append(self.__transform_map(cloudMap))
        self.maxID += 1
        

    def __transform_map(self, cloudMap):
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
        r = theta2r(theta, float(self.config['image']['radius']), self.config['image']['angleProjection'])
        x,y = r * np.sin(phi) + cloudMap.shape[1]//2, r*np.cos(phi) + cloudMap.shape[0]//2
        return cloudMap[x.astype(int),y.astype(int)]

    def __remove_cloud(self, _id):
        try:
            self.maps.pop(_id)
        except:
            raise

    def __calculate_movement(self, map1, map2):
        self.log.debug('Calculate cloud movement')
        print('calc')
        minVal = np.inf
        rotation = (0,0)
        for i,j in product(np.arange(0, map1.shape[0], map1.shape[0]//20), np.arange(0, map1.shape[1], map1.shape[1]//20)):
            temp = np.nansum((map1 - np.roll(np.roll(map2, i, axis=0), j, axis=1))**2)
            if temp < minVal:
                minVal = temp
                rotation = (i,j)
            if temp <= 0:
                break
        return rotation, minVal



    def print_clouds(self):
        print('Currently {} images loaded'.format(len(self.maps)))
        fig = plt.figure(figsize=(16,9))
        for cnt, (i,j) in enumerate(zip(self.maps, self.trans_maps)):
            print(100*len(self.maps)+20 + 2*cnt+1,100*len(self.maps)+20 + 2*cnt+2)
            fig.add_subplot(len(self.maps), 2, 2*cnt+1)
            plt.gca().imshow(i, vmin=0, vmax=1)
            fig.add_subplot(len(self.trans_maps), 2, 2*cnt+2)
            try:
                plt.gca().imshow(self.pred_map[cnt+2], vmin=0, vmax=1)
            except:
                pass
        plt.tight_layout()
        plt.show()

    # update detects clouds in cloudmap and updates old cloud positions
    def update(self, cloudMap):
        print('update')
        self.log.debug('Update')
        # store projection of clouds on atmosphere
        self.__add_cloud(cloudMap)

        if len(self.maps) > 1:
            direct, val = self.__calculate_movement(self.trans_maps[-2], self.trans_maps[-1])
            self.wind_direction.append(direct)
            self.pred_map.append(np.roll(np.roll(self.maps[-1], self.wind_direction[-1][0], axis=0), self.wind_direction[-1][1], axis=1))

        if (len(self.maps) > self.max_maps):
            self.maps.pop(0)
            self.trans_maps.pop(0)
            #self.wind_direction.pop(0)





