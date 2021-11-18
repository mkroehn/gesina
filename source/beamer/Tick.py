import time
import numpy as np

class Tick:
    framecount = 0
    info = ''

    def __init__(self, num_of_tickers):
        self.num_of_tickers = num_of_tickers
        self.tsum = np.zeros(num_of_tickers - 1)
        self.ticker = np.zeros(num_of_tickers)

    def tick(self, i):
        if i >= self.num_of_tickers:
            return
        self.ticker[i] = time.perf_counter()

    def update_info(self):
        self.info = ''
        self.framecount += 1
        for i in range(0, self.num_of_tickers - 1):
            self.tsum[i] += self.ticker[i+1] - self.ticker[i]
            self.info += '[' + str(i) + ']: ' + str(int(self.tsum[i]/self.framecount*1000)) + 'ms '

    def get_info(self):
        return self.info