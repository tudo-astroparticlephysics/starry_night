import numpy as np
import matplotlib.pyplot as plt
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
        self.maxID = 0
        self.timestamps = []
        self.maps = []
        self.trans_maps= []
        self.pred_map = [None, None, ] # can only calculate for third image
        self.wind_direction = [None, ] # can only calculate after 2 images
        self.wind_speed = [None, ] # can only calculate after 2 images

        self.crop = None
        self.trans_crop = None
        self.config = config

    def __enlarge(self, img):
        a = np.full(np.array(img.shape)//2*2+np.array(img.shape), np.nan)
        a[img.shape[0]//2:-(img.shape[0]//2), img.shape[1]//2:-(img.shape[1]//2)] = img
        return a

    def __add_cloud(self, cloudmap, timestamp):
        if not self.maps:
            self.crop = skycam.get_crop_mask(cloudmap, self.config['crop'])
            self.trans_crop = self.__transform_map(self.crop)
        cloudmap[self.crop] = np.NaN
        self.maps.append(cloudmap)
        #self.trans_maps.append(self.__transform_map(cloudmap))
        self.trans_maps.append(cloudmap)
        self.timestamps.append(timestamp)
        self.maxID += 1

    def clear(self):
        self.maxID = 0
        self.maps = []
        self.trans_maps= []
        self.pred_map = [None, None, ]
        self.wind_direction = [None, ]
        self.wind_speed = [None, ]
        

    def __transform_map(self, cloudmap):
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
        x,y = r * np.sin(phi) + cloudmap.shape[1]//2, r*np.cos(phi) + cloudmap.shape[0]//2
        return cloudmap[x.astype(int),y.astype(int)]

    def __shift_and_crop(self, img, shift_x, shift_y):
        '''
        Shift img by shift_x and shift_y and crop all pixels that were wrapped around the image border.
        '''
        shifted_map = np.roll(np.roll(img, shift_x, axis=1), shift_y, axis=0)
        xmin, xmax = max(0, shift_x), min(img.shape[1]+shift_x, img.shape[1])
        ymin, ymax = max(0, shift_y), min(img.shape[0]+shift_y, img.shape[0])
        mask = np.ones(img.shape, dtype=bool)
        mask[ymin:ymax, xmin:xmax] = False
        shifted_map[mask] = np.NaN
        return shifted_map


    def __calculate_movement(self, map1, map2):
        '''
        return x and y shift of clouds and min difference between images
        map1 is the more recent image
        '''
        self.log.debug('Calculate cloud movement')
        print('calc')
        minVal = np.inf
        rotation = (0,0)
        # shift the image and calculate the overlap value
        for shift_x in np.arange(-map1.shape[0]//2//2 , map1.shape[0]//2//2, max([map1.shape[0]//20, 1])):
            for shift_y in np.arange(-map1.shape[1]//2//2 , map1.shape[1]//2//2, max([map1.shape[1]//20, 1])):
                shifted_map = self.__shift_and_crop(map2, shift_x, shift_y)
                temp = np.nanmean((map1 - shifted_map)**2)
                if temp < minVal:
                    minVal = temp
                    rotation = (shift_x,shift_y)
                if temp <= 0:
                    break
        return rotation[::-1], minVal


    def print_clouds(self):
        print('Currently {} images loaded'.format(len(self.maps)))
        fig = plt.figure(figsize=(6,15))
        plt.title('Current image - previous image shifted')
        plt.axis('off')
        print('Wind:', self.wind_direction)
        for cnt, (i,j) in enumerate(zip(self.maps, self.trans_maps)):
            fig.add_subplot(len(self.maps), 2, 2*cnt+1)
            plt.gca().imshow(i, vmin=0, vmax=1)
            if self.wind_direction[cnt] != None:
                plt.gca().arrow(i.shape[0]//2, i.shape[1]//2, self.wind_direction[cnt][0], self.wind_direction[cnt][1], color='yellow', lw=5, label='Wind direction estimated')
                plt.gca().set_ylabel(self.timestamps[cnt], rotation='horizontal', labelpad=55)
            fig.add_subplot(len(self.trans_maps), 2, 2*cnt+2)
            if self.pred_map[cnt] != None:
                pred = self.pred_map[cnt]
                plt.gca().imshow(self.pred_map[cnt], vmin=0, vmax=1)
            else:
                plt.gca().imshow(np.ones(self.maps[0].shape), vmin=0, vmax=1)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('cloud_movement.png') 
        plt.show()

    def predict_next_cloudmap(self, wind_x, wind_y, prev_map):
        '''
        Predict next cloud map by shifting the previous map in wind direction and filling empty pixels
        with values from previous map.
        '''
        shifted_map = self.__shift_and_crop(prev_map, wind_x, wind_y)
        wind_speed = np.sqrt(wind_x**2+wind_y**2)
        wind_steps = wind_speed//self.config['image']['radius']
        shifted_map[np.isnan(shifted_map)] = prev_map[np.isnan(shifted_map)]
        self.pred_map.append(shifted_map)

    def update(self, cloudmap, timestamp):
        '''
        Update the cloud tracker.
        Insert new cloudmap and timestamp and the cloud tracker will calculate wind speed/direction
        and predict the next cloud map for you.
        '''
        print('update')
        self.log.debug('Update')
        # store projection of clouds on atmosphere
        self.__add_cloud(cloudmap, timestamp)

        if len(self.maps) > 1:
            direct, val = self.__calculate_movement(self.trans_maps[-1], self.trans_maps[-2])
            self.wind_direction.append(direct)
            self.predict_next_cloudmap(direct[0], direct[1], self.trans_maps[-1])

        # drop oldest value if more than max_maps are in memory
        if (len(self.maps) > self.max_maps):
            self.maps.pop(0)
            self.trans_maps.pop(0)
            self.timestamps.pop(0)
        if (len(self.wind_direction) > self.max_maps):
            self.wind_direction.pop(0)
            self.wind_speed.pop(0)
        if (len(self.pred_map) > self.max_maps):
            self.pred_map.pop(0)





