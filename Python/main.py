from utils import *
from seqKmeans import *
from parKmeans import *
import time
import statistics

if __name__ == '__main__':
    n = 10000000
    c = 3
    d = 3
    ntest = 21
    speedups = []
    for i in range(ntest):
        print('test', i+1)
        ds = dataGenerator(n, c, d)
        centers = initCenters(ds, c, d)

        start = time.time()
        assign = sKmeans(ds, centers)
        end = time.time()
        pTime2 = end - start
        #print(f'sequential time: {pTime2}')

        start = time.time()
        assign_par = pKmeans(ds, centers)
        end = time.time()
        pTime = end - start
        #print(f'parallel time: {pTime}')

        speedUp = pTime2 / pTime
        print("speedup: ", speedUp)
        speedups.append(speedUp)

    median = statistics.median(speedups)
    print("speedup median: " + str(median))
