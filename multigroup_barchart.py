# a bar plot with errorbars
import numpy as np
import matplotlib.pyplot as plt
import itertools

def bar_chart(mean_lists, std_lists, group_labels, tick_labels, colors_list=None, set_separation=.5, side_buffer=0.1):
    '''
    modified from http://matplotlib.org/examples/api/barchart_demo.html (May 10, 2015)
    
    The axis of the plot is returned so that the user may add axes labels, plot title, and legend, and make modifications.
    

    -------------------------------------------------------------------------------------

    Example:
    
    menMeans = (20, 35, 30, 35, 27)
    menStd =   (2, 3, 4, 1, 2)
    womenMeans = (25, 32, 34, 20, 25)
    womenStd =   (3, 5, 2, 3, 3)
    
    
    colors = ['r','purple']
    means = [menMeans, womenMeans]
    stds = [menStd, womenStd]
    group_labels = ('Men','Women')
    tick_labels = ('G1', 'G2', 'G3', 'G4', 'G5')
    
    ax = bar_chart(means, stds, group_labels, tick_labels, colors)
    ax.legend(loc=8)
    plt.title("Men vs. Women")
    plt.show()
    
    -------------------------------------------------------------------------------------
    '''
    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')
    
    N = max([len(i) for i in mean_lists]) #number of values in each group
    num_groups = len(mean_lists)
    
    if not colors_list:
        colors_list = ['b' for i in np.arange(num_groups)]
    
    width=1.0/(num_groups+set_separation)
    ind = np.arange(N)+set_separation  # the x locations for the groups. note each value is the left most edge of a bar.
    fig, ax = plt.subplots()
    bar_groups = [] #collect for legend and labeling purposes
    
    x_max = -np.inf
    x_min = np.inf
    for n, (means, stds, labels, color) in enumerate(itertools.izip(mean_lists, std_lists, group_labels, colors_list)):
        left = ind+(n*width)
        bar_set = ax.bar(left, means, width, color=color, yerr=stds, label=labels)
        autolabel(bar_set)
        bar_groups.append(bar_set)
        if x_max < left[-1]:
            x_max = left[-1]
        if x_min > left[0]:
            x_min = left[0]

    ax.set_xlim(left=x_min-side_buffer, right=x_max+width+side_buffer)

    tick_positions = ind + (width*num_groups/2)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels( tick_labels )

    return plt.gca()

if __name__ == '__main__':
    # fake data
    menMeans = (20, 35, 30, 35, 27)
    menStd =   (2, 3, 4, 1, 2)
    
    womenMeans = (25, 32, 34, 20, 25)
    womenStd =   (3, 5, 2, 3, 3)

    intersexMeans = (23,22,22,15,31)
    intersexStd = (0,0,0,0,0)#(1,4,1,5,2)
    
    # group data + setup
    colors = ['r','purple','g']
    means = [menMeans, womenMeans, intersexMeans]
    stds = [menStd, womenStd, intersexStd]
    
    # 3 group bar chart
    ax = bar_chart(means,stds,('Men','Women','Intersex'),('G1', 'G2', 'G3', 'G4', 'G5'), colors)
    ax.legend(loc=8)
    
    # 2 group bar chart
    ax = bar_chart(means[:2],stds[:2],('Men','Women'),('G1', 'G2', 'G3', 'G4', 'G5'), colors[:2])
    ax.legend(loc=8)
    
    plt.show()
