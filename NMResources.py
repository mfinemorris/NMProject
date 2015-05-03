import operator
import os, sys, re, shutil, math, time

import numpy as np
from numpy import NaN, Inf, arange, isscalar, asarray, array

import scipy
from scipy import spatial, signal, fft, arange

import pandas as pd
from pandas import Series, DataFrame

debug = True
element_sep = "-"
#period_sub = "o" #may use in future







def delta_tuner(dataframe, epsilon, rate): #choose which data to use to tune. can be either selected list or full ist. AD reccoments full list.
    '''
    this function takes a dataframe of time series data and runs peak detection iteritvely. 
    since peak detection always has the the range of delta values of 0 to the max of stack,
    epsiolon is used to be the number of divisions of that range to test. 1 being the minimum for epsilon, which is the exact middle of the range.
    the function will return a results table (average # of events) and (# ROIs with events>1) on their own axis.
    the graph will be click able, as to obtain the delta value that generated that point.
    data results are not saved.
    '''
    
    range_array = np.linspace(0, max(dataframe.max())/2, num = epsilon) #create the array of which delta values to test. the range is from zero (although zero is not used) to half of the max value from the entire data frame. epsilon is used to determine the number of slices to make
    
    results_average = Series(index = range_array) #the empty series to store results
    results_num = Series (index = range_array) #an empty series to store results
    #results_perc = Series (index = range_array)
    
    for delta in range_array[1:]: #for each delta value in the array
        
        peak_amp_temp, peak_sets_temp_x, peak_sets_temp_y = event_detection(dataframe,delta, rate) #perform event detection with the delta

        event_counts = peak_amp_temp.loc['count'] #count the number of events, which is a row in the peak_amp_temp array
        average_num_events = event_counts.mean() #average the counts, to obtain (average # events/roi)
        num_roi = event_counts[event_counts>=1].count() #count the number of ROIs with more than one event
        
        #perc_roi = num_roi/len(data_smooth.columns)

        results_average[delta] = average_num_events 
        results_num[delta] = num_roi
        #results_perc[delta]= perc_roi
    return results_average, results_num

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def peakdet(v, delta, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    
    Returns two arrays
    
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %      
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    
    """
    maxtab = []
    #mintab = []
       
    if x is None:
        x = arange(len(v))
    
    v = asarray(v)
    
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    
    lookformax = True
    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                #mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab)#, array(mintab)


def event_detection(data, delta, rate):
    '''
    Do peak detect on a dataframe. Takes a delta value and the rate.
    A point is considered a maximum peak if it has the maximal value, and was preceded (to the left) by a value lower by DELTA.
    Returns a DataFrame of statistics on the peak amplitudes, and two dictionaries, one for the time and one for amplitude of each peak.
    '''
    #results storage
    peak_amp_temp = DataFrame()
    rr_int_temp = DataFrame()
    peak_sets_temp_x = {} #time
    peak_sets_temp_y = {} #amplitude

    for label, column in data.iteritems(): #for each column in the data frame
        start_time = time.clock()
        time_arr = column.index.tolist() #time array
        col = column.tolist() #amplitude array

        #peakdet
        maxtab = peakdet(col, delta,None)

        maxtab = np.array(maxtab)

        if maxtab.size == 0: #how to handle empty np array, which occurs if no events are detected
            maxptime = []
            maxpeaks = []
        
        else:
            maxptime = maxtab[:,0] #all of the rows and only the first column are time
            maxpeaks = maxtab[:,1] #all of the rows and only the second column are amp.

        maxptime_true = (np.multiply(maxptime,rate) + time_arr[0]) #get the real time for each peak, since the peak is given in index not time
        peak_sets_temp_x[label] = maxptime_true #store array of event time in the dictionary with the ROI name as the key

        peak_sets_temp_y[label] = np.array(maxpeaks) #store array of events in the dictionary with the ROI name as the key
        #RR = rrinterval(maxptime_true)
        peak_amp_temp[label] = Series(data = maxpeaks, index=maxptime_true).describe() #store summary data series in the summary dataframe
        #rr_int_temp[label] = Series(data = RR, index=maxptime_true[:-1]).describe()
        end_time = time.clock()
        print label, 'took', (end_time - start_time), 'seconds to run'
    return peak_amp_temp, peak_sets_temp_x, peak_sets_temp_y



def rrinterval(maxptime): 
    """
    find time from r peak to r peak, called the R-R interval. Input array must be a list of numbers (float).
    """
    
    rrint = [] #empty array for ttot to go into
    
    for time in maxptime[1:]: #for each r peak, starting with the second one
        s2time = maxptime.index(time) 
        s2 = maxptime[s2time-1]
        meas = time - s2 #measure the interval by subtracting
        rrint.append(meas) #append the measurement to the ttotal array
    return rrint #return array



def sliding_freq(dictionary, data_raw, window, key_word):
    '''
    takes a window size in ms and does non-overlapping count
    (which is freq, since the windows are all the same size) and average ibi.
    key_word is the key used to access the section in each dictionary value
    over which the count should occur.
    returns two dataframes with this information for each 'test' case.
    '''
    import math
    
    num = math.trunc(data_raw.index[-1]/window) #get the number of windows to do the counts and averages over.
    
    sliding_count = DataFrame(index= (np.arange(num)*window)) #where we're storing the results for each timeseries
    sliding_mean = DataFrame(index= (np.arange(num)*window)) #where we're storing the results for each timeseries
    
    for key, value in dictionary.iteritems():
        
        temp_count = Series(index=(np.arange(num)*window)) #temp storage
        temp_mean = Series(index=(np.arange(num)*window)) #temp storage
        
        for i in (np.arange(num)*window):
            temp_count[i] = value[key_word][i:(i+window)].count() #get the count in the window
            temp_mean[i] = value[key_word][i:(i+window)].mean() #get the mean in the window
        
        temp_mean = temp_mean.fillna(0) #temp mean returns NaN for windows with no events. make it zero for graphing
        
        sliding_count[key] = temp_count #store series in results table
        sliding_mean[key] = temp_mean #store series in results table
    
    sliding_count = sliding_count.sort_index(axis = 1)
    sliding_mean = sliding_mean.sort_index(axis = 1) #my attempt at reordering so the columns are in increaing order
    return sliding_mean, sliding_count


def histent(data_list):
    '''
    Input must be a list. returns the entropy score and binarray. 
    '''
    NumBin = int(2 * (math.log(len(data_list), 2)))
    binarray = np.linspace(min(data_list), max(data_list), NumBin)
    no, xo = np.histogram(data_list, binarray); #print no

    # normalize the distribution to make area under distribution function unity
    no = no/sum(no); #print no

    # find the bins that have no samples to prevent math.log(0) in calculation
    no = no[np.nonzero(no)]    

    # if all of the samples fall in one bin regardless of the bin size
    # means we have the most predictable sitution and the entropy is 0
    # if we have uniformly dist function, the max entropy will be 1

    HistEntropy = [-1*(x * math.log(x, 2))/(math.log(NumBin,2)) for x in no]
    #print 'HistEntropy array=', HistEntropy
    HistEntropy = sum(HistEntropy)
    #print 'HistEntropy=', HistEntropy
    
    return HistEntropy, binarray


def peak_amplitude_interval(peak_sets_temp_x, peak_sets_temp_y):
    #all events detected
    peaks = {}
    temp = []

    for key, value in output_dict['peak_sets_temp_x'].iteritems():
        start_time = time.clock()

        xlist = value.tolist()
        ylist = output_dict['peak_sets_temp_y'][key].tolist()
        
        RR = NMResources.rrinterval(xlist)
        RR.append(NaN)
        #make Dataframe with index as xlist (time of peak) and store amplitude of peak (ylist) and RR-interval
        peaks[key] =  DataFrame({'Amplitude':ylist,'Interval':RR}, index = xlist)

        #### Collect peak data for saving
        temp.append(xlist)
        temp.append(ylist)
        temp.append(RR[:-1])

        end_time = time.clock()
        print key, "finished in", np.round(end_time - start_time, 3)
    '''
    #at some point finish function up so that if ouput_dir is provided temp is saved immediately
    #and only peaks is returned.
    
    cols=pd.MultiIndex.from_product([data_raw.columns,[u'rr', u'x', u'y']],names=['variable','peak_data'])
    max_peaks = max([len(peak_x[key]) for key in peak_x])
    peak_x_y = pd.DataFrame(data=temp, index=cols, columns=range(0,max_peaks)).T.to_sparse()
    save_file = os.path.join(output_dict['output_directory'],'peaks-'+file_info_str+'.txt')
    print save_file

    # IDK why this works and peak_x_y.to_csv() doesn't, but you can't argue with results...
    peak_dict = peak_x_y.to_dict()
    pd.DataFrame().from_dict(peak_dict).to_csv(save_file)
    '''
    
    return peaks, temp



#File Name Functions

def preprocess_params(parameters):
    parameters = sorted(parameters.items(), key=operator.itemgetter(1)) #sort dictionary so that items with None will appear first
    params = element_sep.join([key+str(val) if val else key for key,val in parameters]).replace('.','_') #process params
    return params



def make_folderpaths(model, time, parameters={}, path=''):
    params = preprocess_params(parameters)
    time_seconds = str(int(round(time/1000.))) #convert time to seconds and string
    name_parts = [model, "sec"+ time_seconds]
    folder_name = element_sep.join((name_parts+[params] if params else name_parts))
    if path and os.path.isdir(path):
        return os.path.join(path, folder_name)
    else:
        return folder_name



def make_filepaths(info_type, model, time, parameters={}, extension='txt', path=''):
    '''
    Creates a complete path name for a file, where the file name has the following format:

    info_type '-' model '-sec' time '-' parameters '.' extension

    info_type = string indicating what data is being saved. 'voltage' or 'timing' or something else.
    model = the model that was simulated
    time = number of milliseconds that were simulated
    parameters = is a dictionary where any parameter changes can be specified as key:value pairs (key is the name of the parameter!). 
    If the data comes from runs with different values of the same parameter, the value should be None.
    extension = the extension WITHOUT the '.' i.e. 'txt', 'csv', etc.
    path = the path if different from the current working directory
    '''
    #prepare file name components
    params = preprocess_params(parameters)
    time_seconds = str(int(round(time/1000.))) #convert time to seconds and string
    file_name_parts = [info_type, model, "sec"+ time_seconds]

    file_name = element_sep.join((file_name_parts+[params] if params else file_name_parts)) + "." + extension
    if path and os.path.isdir(path):
        return os.path.join(path, file_name)
    else:
        return file_name

def deconstruct_filepaths(file_path, element_seperator=None):
    '''
    Deconstructs a filepath of format:

    info_type '-' model '-sec' time '-' parameters '.' extension

    and returns 

    '''
    
    ## break up the path into its components ##
    #separate the directory from the file name
    base = os.path.basename(file_path)
    directory = os.path.dirname(file_path)
    #separate the file extension and file path
    file_info, delim, ext = base.partition('.')
    #break the file name into its components
    if element_seperator:
        file_info = file_info.split(element_seperator)
    else:
        file_info = file_info.split(element_sep)

    stored_data_type = file_info[0]
    model = file_info[1]
    
    params = {}
    for i in file_info[2:]:
        param_name = re.sub("[^A-z]", "", i).replace("_","") #extract secondary parameter from file name
        param_val = re.sub("[^0-9.]", "", i.replace("_",".")) #extract secondary parameter from file name
        params[param_name] = param_val
    seconds = float(params['sec'])*1000.
    del params['sec']
    
    return stored_data_type, model, seconds, params, ext, directory

def test():
    #test that it works
    print "Folder: ",make_folderpaths('BRSModel', 1000*60*6, parameters={'IP':None, 'gL':2.3, 'gnaps':None}, path = os.getcwd())

    print "Calling make_filepaths('voltage', 'BRSModel', 1000*60*6, parameters={'IP':None, 'gL':2.3, 'gnaps':None}) creates name:"
    print
    path = make_filepaths('voltage', 'BRSModel', 1000*60*6, parameters={'IP':None, 'gL':2.3, 'gnaps':None}, path = os.getcwd())
    print "constructed path: ",path
    print
    path_parts = deconstruct_filepaths(path)
    print path_parts
    print
    print make_filepaths(*path_parts)


def convert_to_new_scheme(file_dir, pattern):

    #get all files in file_dir matching pattern
    dir_contents = os.listdir(file_dir) #get list of all files in file_dir
    matching_files =  [data_file for data_file in dir_contents if re.search(pattern, data_file)] #make list of files matching pattern
    
    for datafile in matching_files:
        #separate the file extension and file path
        name_stem, delim, ext = datafile.partition('.')
        #break the file name into its components
        file_info = name_stem.split("-")
        #format: info_type '-' model '-sec' time '-' parameters '.' extension
        try:
            new_file = "-".join([file_info[0],file_info[2],file_info[3],file_info[1]]+file_info[4:])+delim+ext
        except:
            print "Could not gen new file name for", datafile
            continue
        
        old_path, new_path = os.path.join(file_dir, datafile), os.path.join(file_dir, new_file)
        
        try:
            os.rename(old_path, new_path)
            print "Renamed", old_path, "to", new_path
        except:
            print "Failed to rename", old_path, "to", new_path


if __name__ == "__main__":
    if debug:
        test()
        sys.exit()
    #convert_to_new_scheme(os.path.join(os.getcwd(), 'Results'), 'voltages-IP-TBModel')
    

