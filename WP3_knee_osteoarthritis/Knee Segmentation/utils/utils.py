import matplotlib.pyplot as plt
import numpy as np

class average_meter(object):
    """Computes and stores the average and current value"""

    def __init__(self,remember=100):
        self.reset()
        self.remember=remember

    def reset(self):
        self.data=[]
        self.avg=None
        self.count=0

    def update(self, val):
        if self.count==self.remember:
            self.data=self.data[1:]
        self.data.append(val)
        self.count+=1
        self.avg = np.mean(self.data)