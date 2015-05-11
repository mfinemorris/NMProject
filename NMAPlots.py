import pandas as pd
import NMResources
import NMUtility
import os, sys
from matplotlib import pyplot as plt


#heat plots
from heatmap import *
from multigroup_barchart import *

#raster plot
def raster_plot(peak_sets_temp_x, variable, x_min = 10000, x_max=20000, output_path=''):
    
    plt.figure()
    
    for key, value in peak_sets_temp_x.iteritems():
        key = float(key)
        temp_y = []
        for n in value:
            temp_y.append(key)
        plt.plot(value, temp_y, marker = '.', color = 'k', linestyle = 'None', markersize = 2)
    # title and labels
    plt.xlabel('Time (ms)')
    plt.ylabel('%s' %variable)
#    plt.title('Raster plot (%s)'%output_dict['exp'])

    #plot range
    keys = sorted(peak_sets_temp_x.keys(),cmp=lambda x,y: cmp(float(x), float(y)))
    y_min = float(keys[0])
    y_max = float(keys[-1])
    buff = abs(y_max-y_min)/6.0
    plt.ylim(ymax= y_max+buff, ymin=y_min-buff)
    plt.xlim(xmin=x_min, xmax=x_max)
    #save plot
    if output_path:
        #fig.set_canvas(plt.gcf().canvas) #might fix savefig problem?
        plt.savefig(output_path)
        print output_path
        return plt.gca()
    plt.show()


    
def time_series(data, label, exp='', peaks=None, bursts=None, xmin=30000, xmax=45000, output_directory=''):
    
    plt.figure()
    plt.plot(data.index, data, color = 'k')
    plt.xlabel('Time (ms)')
    plt.ylabel('Voltage (mV)')

    try:
        plt.title(label)
        plt.title('%s \n(%s)' %(label, exp))
    except:
        pass #no title
    
    #try to mark the peaks if peaks DataFrame was passed in
    if peaks is not None:
        #plot peaks with green triangles
        plt.plot(peaks.index, peaks['Amplitude'], marker ='^', color = 'g', linestyle = 'None')

        #try to mark the burst start and end peaks if bursts and peaks DataFrames were passed
        if bursts is not None:  
            #mark burst start (magenta triangle)
            for index, row in bursts['Start'].iteritems():
                plt.plot(row, peaks['Amplitude'].loc[row], marker ='^', color = 'm', linestyle = 'None')
            #mark burst end (yellow triangle)
            for index, row in bursts['End'].iteritems():
                plt.plot(row, peaks['Amplitude'].loc[row], marker ='^', color = 'y', linestyle = 'None')

    plt.xlim(xmin, xmax) #you must set limits AFTER bursts and peaks are plotted!
    
    if output_directory:
        plt.savefig(output_directory+"time series "+str(label)+'.png')
        pass #save fig to its own png file!
    
    return plt.gca()


def all_time_series(data_raw, peaks=None, bursts=None, exp='', output_directory='', xmin=30000, xmax=45000):
    '''
    create and save all time series if output_directory is supplied. 
    requires that event detection has been performed. 
    saves out plots of graphs with the same scale, 30 s - 45 s.
    '''
    if output_directory:
        try:
            from matplotlib.backends.backend_pdf import PdfPages
            f = os.path.join(output_directory,'TimeSeries.pdf')
            print f
            pp = PdfPages(f)
        except IOError as e:
            if errno.EACCES == e.errno:
                #replace file
                os.chmod(e.filename, 0777)
                os.remove(e.filename)
                pp=PdfPages(e.filename)

    for label, column in data_raw.iteritems():
        
        if peaks and bursts:
            plot_time_series(column, label, exp, peaks[label], bursts[label], xmin=xmin, xmax=xmax)
        elif peaks:
            plot_time_series(column, label, exp, peaks[label], xmin=xmin, xmax=xmax)
        elif bursts:
            plot_time_series(column, label, exp, bursts[label],xmin=xmin, xmax=xmax)
        else:
            plot_time_series(column, label, exp, xmin=xmin, xmax=xmax)

        if output_directory:
            pp.savefig()
            
    if output_directory:
        pp.close()

    plt.show()
    
def plot_histEntropy(hist_ent_series):
    
    for key, value in hist_ent_series.iteritems():
        plt.plot(value)
    

def plot_hist(data_col_labels, data_dict, label, plot_title='', plot_file=''):
    
    series = pd.Series(index = data_col_labels)
    
    for key, results in data_dict.iteritems():
        
        temp = results[label].tolist()
        try:
            HistEntropy, binarray = NMResources.histent(temp)
            series[key] = HistEntropy
        except Exception as e:
            print "Skipping %s %s because %s"%(variable, key, e)
            continue
        plt.hist(temp,binarray)
        plt.xlabel('%s (s)'%label)
        plt.ylabel('Count')
        
        try:
                plt.title(plot_title % key)
        except TypeError or ValueError:
            plt.title(plot_title)

        if plot_file:
                plt.savefig(plot_file%key)
                plt.close()
        else:
            plt.show()    
    return series




def line_plots(data_orignal, data_smooth, events_x, events_y, peak_sets_temp_x, peak_sets_temp_y, event_summary,folder):
    '''
    creates the plots with two lines:
    original data and smoothed data.
    it also overlays the events from LCpro and RAIN.
    '''
    lcpro_events_select_list = event_summary[event_summary['LCpro, select'] >= 1].index.tolist() #list of only roi's found by RAIN
    
    for label, column in data_orignal.iteritems():
        
        plt.figure()
        plt.xlabel('Time (s)')
        plt.ylabel('Intensity')
        plt.title(label)
        plt.ylim(ymin = min(data_orignal.min()), ymax = max(data_orignal.max()))
        plt.xlim(xmin = data_orignal.index[0], xmax = data_orignal.index[-1])
        
        plt.plot(data_orignal.index, data_orignal[label], label = 'original', color = 'r')
        plt.plot(data_orignal.index, data_smooth[label], label = 'smooth', color = 'b')
        if label in data_orignal.columns:     
            plt.plot(events_x[label], events_y[label], marker = "^", color="r", linestyle= "None")
        if label in lcpro_events_select_list:
            plt.plot(events_x[label], events_y[label], marker = "^", color="g", linestyle= "None")
        if label in peak_sets_temp_x.keys():
            plt.plot(peak_sets_temp_x[label], peak_sets_temp_y[label], marker = "^", color="y", linestyle= "None")
        #plt.savefig(r'%s/plots/%s.pdf' %(folder,label))
        #plt.close()
    plt.show()
    
if __name__ == "__main__":

    data_raw = pd.read_csv(r'NMResults\TBModel-sec300-eL-IP0_96-gnaps2_8\voltage-TBModel-sec300-eL-IP0_96-gnaps2_8.txt', index_col= 0)
    raster_plot(output_dict['peak_sets_temp_x'],'eL',output_dir = output_dict['output_directory'])
