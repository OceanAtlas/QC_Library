import numpy as np
from scipy.stats import iqr

def outlierDetector(timeArray, data, flags, minWindowN=4, windowInDays=1):
    if data.ndim != 1 or timeArray.ndim != 1 or flags.ndim != 1:
        raise ValueError("Inputs must be 1D")
    variances = np.zeros_like(data)

    if windowInDays != 'all':
        deltaTime = pd.Timedelta(windowInDays, 'D').to_timedelta64()
    windowsN = np.zeros_like(data, dtype=np.int)
    means = np.zeros_like(data)
    numDevs = np.zeros_like(data)

    for indx,time in enumerate(timeArray):
        if (flags[indx] == 4) or (flags[indx] == 9):
            numDevs[indx] = np.nan
            continue

        if windowInDays == 'all':
            window = np.arange(timeArray.size)
        else:
            window = np.where((timeArray > time-deltaTime) &  (timeArray < time+deltaTime) &
                              (timeArray != time) & (flags != 4))[0]
        windowsN[indx] = window.size
        if windowsN[indx] < minWindowN:
            flags[indx] = 9
            numDevs[indx] = np.nan
            continue
        iQRange = iqr(data[window], nan_policy='omit')
        variances[indx] = (3/4) * iQRange
#         variances[indx] = np.nanvar(data[window])
        means[indx] = np.nanmean(data[window])
        numDevs[indx] = (data[indx] - means[indx])/np.sqrt(variances[indx])
        #return dict
        retdict = {'numdevs':numDevs, 'flags': flags, 'means': means, 'variances':variances}
    return retdict

def outlierRemoval(timeArray, data, windowInDays=1, minWindowN=4, numStdDevs=5, maxIterations=None, flags=None, verbosity=0):
    # create the flags array
    if flags is None:
        localflags = np.zeros_like(data, dtype=np.int) + 2
    else:
        localflags = np.copy(flags)

    # detect first outlier
    #print(np.where(flags != 2)[0].size)
    aDict = outlierDetector(timeArray, data, localflags, minWindowN=minWindowN, windowInDays=windowInDays)
    numDevs = aDict['numdevs']
    localflags = aDict['flags']

    cnt = 0
    if (maxIterations is None):
        maxIterations = data.size
    while ((np.nanmax(np.abs(numDevs)) > numStdDevs) and (cnt < maxIterations)):
        if verbosity > 0:
            print(cnt, " ", np.where(localflags != 2)[0].size," points flagged")

        localflags[np.where(np.abs(numDevs) > numStdDevs)] = 4
        aDict = outlierDetector(timeArray, data, localflags, minWindowN=minWindowN, windowInDays=windowInDays)
        numDevs = aDict['numdevs']
        localflags = aDict['flags']
        cnt += 1
    print(cnt, " ", np.where(localflags != 2)[0].size," points flagged")
    retdict = {'numdevs':numDevs, 'flags': localflags, 'means': aDict['means'], 'variances':aDict['variances']}
    return retdict

def grossRangeFlag(timeArray, data, min, max, flags=None):
    if data.ndim != 1 or timeArray.ndim != 1:
        raise ValueError("Inputs must be 1D")
    if flags is None:
        localflags = np.zeros_like(data, dtype=np.int) + 2
    else:
        localflags = np.copy(flags)

    # iterate
    for indx,time in enumerate(timeArray):
        if data[indx] < min or data[indx] > max:
            localflags[indx] = 4
    return localflags
