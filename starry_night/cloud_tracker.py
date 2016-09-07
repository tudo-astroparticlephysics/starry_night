import numpy as np
import matplotlib as plt
from skimage.segmentation import active_contour

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
    def __init__(self):
        self.clouds = []
        self.maxID = -1
        self.wind_speed = None
        self.wind_direction = None

    def add_cloud(self):
        self.add_cloud(self, self.maxID + 1)
        self.maxID += 1

    def remove_cloud(self, _id):
        popped = False
        for i in enumerate(clouds):
            if clouds[i].id == _id:
                clouds.pop(i)
                popped = True
                break
        if not popped:
            raise IndexError('No Cloud with id={} in CloudTracker!'.format(_id))

    def print_clouds(self):
        print('Currently {} clouds in the sky'.format(len(self.clouds)))
        for c in enumerate(self.clouds):
            c.print_info()

    # update detects clouds in cloudmap and updates old cloud positions
    def update(self, cloudMap):
        # detect clouds
        snake = active_contour(cloudMap, init, alpha=0.015, beta=10, gamma=0.001)




