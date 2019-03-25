import numpy as np
from scipy.stats import iqr
import pandas as pd
import json
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

DEFAULT_MIN_N = 20
DEFAULT_NUM_SD = 5
DEFAULT_NUM_PRE_POST_PTS = 5
DEFAULT_WINDOW_IN_DAYS = 7

def outlierDetector(timeArray, data, flags, minWindowN=DEFAULT_MIN_N, windowInDays=DEFAULT_WINDOW_IN_DAYS,
					minPrePostPts=DEFAULT_NUM_PRE_POST_PTS):
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
			variances[indx] = np.nan
			means[indx] = np.nan
			continue

		if windowInDays == 'all':
			window = np.arange(timeArray.size)
		else:
			window = np.where((timeArray > time-deltaTime) &  (timeArray < time+deltaTime) &
							  (timeArray != time) & (flags != 4))[0]
		windowsN[indx] = window.size

		window_before = np.where(window < indx)[0]
		window_after = np.where(window > indx)[0]

		if (windowsN[indx] < minWindowN or window_before.size < minPrePostPts or
			window_after.size < minPrePostPts):
			flags[indx] = 9
			numDevs[indx] = np.nan
			variances[indx] = np.nan
			means[indx] = np.nan
			continue
		iQRange = iqr(data[window], nan_policy='omit')
		#variances[indx] = (3/4) * iQRange
		variances[indx] = (iQRange/1.35)*(iQRange/1.35)
		#         variances[indx] = np.nanvar(data[window])
		means[indx] = np.nanmean(data[window])
		numDevs[indx] = (data[indx] - means[indx])/np.sqrt(variances[indx])
		#return dict
		retdict = {'numdevs':numDevs, 'flags': flags, 'means': means, 'variances':variances}
	return retdict

def outlierRemoval(timeArray, data, windowInDays=DEFAULT_WINDOW_IN_DAYS, minWindowN=DEFAULT_MIN_N, numStdDevs=DEFAULT_NUM_SD, maxIterations=None, flags=None, verbosity=0):
	# create the flags array
	if flags is None:
		localflags = np.zeros_like(data, dtype=np.int) + 2
		if verbosity > 0:
			print("Initializing flags array")
	else:
		localflags = np.copy(flags)
		if verbosity > 0:
			print("Starting with gross-range flags array")

	# detect first outlier
	#print(np.where(flags != 2)[0].size)
	aDict = outlierDetector(timeArray, data, localflags, minWindowN=minWindowN, windowInDays=windowInDays)
	numDevs = aDict['numdevs']
	origNumDevs = np.copy(numDevs)
	localflags = aDict['flags']
	orig_means = aDict['means']
	orig_vars = aDict['variances']

	cnt = 0
	if (maxIterations is None):
		maxIterations = data.size

	numStartingFlags = np.where(localflags != 2)[0].size
	while ((np.nanmax(np.abs(numDevs)) > numStdDevs) and (cnt < maxIterations)):
		if verbosity > 0:
			if cnt == 0:
				print("Starting with " + str(numStartingFlags) + " flags")
			else:
				print(cnt, " ", str(np.where(localflags != 2)[0].size - numStartingFlags)," total points flagged by Spike Test")

		localflags[np.where(np.abs(numDevs) > numStdDevs)] = 4
		aDict = outlierDetector(timeArray, data, localflags, minWindowN=minWindowN, windowInDays=windowInDays)
		numDevs = aDict['numdevs']
		localflags = aDict['flags']
		cnt += 1
	if verbosity > 0:
		print(cnt, " ", np.where(localflags != 2)[0].size," total points flagged")
	retdict = {'orig_vars': orig_vars, 'orig_means' : orig_means, 'orig_num_devs': origNumDevs, 'numdevs':numDevs, 'flags': localflags, 'means': aDict['means'], 'variances':aDict['variances']}
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

def diagnostic_plots(parameter, alldata, times, flag_arrays=None, time_range=None,
					 yrange=None, figsize=[20, 4], title=None, xlabel=None, ylabel=None,
					 marker=None):
	# This plots time series plots
	# First just plot the raw data
	plt.figure(figsize=figsize)

	#set the time range
	if time_range is not None:
		subset_times = np.array(time_range, dtype='datetime64[s]')
		plt.xlim(subset_times)

	# set the y Range
	if yrange is not None:
		# Trange is a list
		plt.ylim(yrange)

	allgooddata = np.copy(alldata)
	allnonflaggeddata = np.copy(alldata)

	#Now plot the flags
	if flag_arrays is not None:
		test_order = ['gross', 'spike']
		plot_order = []
		for test in test_order:
			for key in flag_arrays:
				if test in key.lower():
					plot_order.append(test)

		goodafter_dict = {}
		color_dict = {}
		for key,flagvalues in flag_arrays.items():
			if 'spike' in key.lower():
				spikeddata = np.copy(allgooddata)
				goodafterspike = np.copy(allgooddata)
				allgooddata[np.where(flagvalues != 2)] = np.nan
				allnonflaggeddata[np.where(flagvalues == 4)] = np.nan
				goodafterspike[np.where(flagvalues != 2)] = np.nan
				spikeddata[np.where(flagvalues == 2)] = np.nan
				goodafter_dict['spike'] = goodafterspike
				color_dict['spike'] = 'r'
			if 'gross' in key.lower():
				grossdata = np.copy(allgooddata)
				goodaftergross = np.copy(allgooddata)
				allgooddata[np.where(flagvalues != 2)] = np.nan
				allnonflaggeddata[np.where(flagvalues == 4)] = np.nan
				goodaftergross[np.where(flagvalues != 2)] = np.nan
				grossdata[np.where(flagvalues == 2)] = np.nan
				goodafter_dict['gross'] = goodaftergross
				color_dict['gross'] = 'y'

		for index, test in enumerate(plot_order):
			if index == 0:
				# if this is the first one, use raw data
				plt.plot(times, alldata, c=color_dict[test] ,marker=marker)
			else:
				previous_test = plot_order[index-1]
				plt.plot(times, goodafter_dict[previous_test], c=color_dict[test], marker=marker)

	plt.plot(times, allnonflaggeddata, c='greenyellow', marker=marker)
	plt.plot(times, allgooddata, c='b', marker=marker)


	if title is not None:
		plt.title(title)
	else:
		plt.title(parameter)
	if xlabel is not None:
		plt.xlabel(xlabel)
	if ylabel is not None:
		plt.ylabel(ylabel)
	else:
		plt.ylabel(parameter)


def analyze(file, parameter, time_param='time_vals', gross_range=None, window_days=DEFAULT_WINDOW_IN_DAYS, min_n=DEFAULT_MIN_N, num_stdevs=DEFAULT_NUM_SD,
			max_iterations=None, missing_values=[-999, -99, -9.99, -9.999],
			verbosity=0):

	with open(file) as f:
		params_data = json.load(f)

	params = params_data['deployment_meta']['params']
	if verbosity > 1:
		print(params)
	times = np.array(params_data[time_param], dtype='datetime64[s]')
	paramsArray = np.array(params)
	param_index = np.where(paramsArray == parameter)[0][0]  #makes a temporary array from params dictionary
	if verbosity > 1:
		print(param_index)

	obs_vals = params_data['obs_vals']
	data_vals = np.array(obs_vals[str(param_index)])

	for abadval in missing_values:
		data_vals[np.where(data_vals==abadval)] = np.nan

	# Call Gross Range Test
	if gross_range is not None:
		gross_range_flags = grossRangeFlag(times, data_vals,
													   gross_range[0], gross_range[1])
	else:
		gross_range_flags = None

	# Call Spike Test
	spike_test_dict = outlierRemoval(times, data_vals, windowInDays=window_days,
								maxIterations=max_iterations, minWindowN=min_n,numStdDevs=num_stdevs,
								flags=gross_range_flags, verbosity=verbosity)
	# returns: {'numdevs':numDevs, 'flags': localflags, 'means': aDict['means'],
	# 'variances':aDict['variances']}

	flags = spike_test_dict['flags']
	return {'parameter' : parameter,
			'data': data_vals,
			'times' : times,
			'orig_num_devs' : spike_test_dict['orig_num_devs'],
			'num_devs' : spike_test_dict['numdevs'],
			'orig_vars' : spike_test_dict['orig_vars'],
			'variances' : spike_test_dict['variances'],
			'orig_means': spike_test_dict['orig_means'],
			'means' : spike_test_dict['means'],
			'gross_range_flags': gross_range_flags,
			'spike_flags': flags}

def summarizeWOAFile(inFile, param):
	rootgrp = netCDF4.Dataset(inFile, "r", format="NETCDF4")
	print("Root Group: ", rootgrp.variables, end="\n")
	print("Data Model: ", rootgrp.data_model, end="\n")
	print("Dimensions:", rootgrp.dimensions, end="\n")
#     print(len(rootgrp.dimensions.get("lat")))
#     print(len(rootgrp.dimensions.get("lon")))
#   Get the Range of the 'o_an' variable
	var_vals = rootgrp.variables.get(param)
	print("#######", var_vals)

#TODO add vmin, vmax
def plotGridFile(grid, lat_grid, lon_grid, xlim=[-180, 180], ylim=[-90, 90], title='Global Map'):
	colors = [(0.765, 0.765, 0.074), (1, 1, 1), (0.514, 0.074, 1)]
	cm = LinearSegmentedColormap.from_list(title, colors, 16)
	plt.imshow(grid, extent=(lon_grid.min(), lon_grid.max(), lat_grid.max(), lat_grid.min()),
		  interpolation='nearest', cmap=cm) #LRBT vmin=0.2, vmax=8.2
	plt.xlim(xlim[0], xlim[1])
	plt.ylim(ylim[0], ylim[1])
	plt.colorbar(cmap=cm)
	plt.title(title)
	plt.show()