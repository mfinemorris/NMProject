import os, errno, sys, itertools, contextlib, shutil, itertools
import pandas as pd

def mkdir_p(path):
    '''
    This function creates a folder at the end of the specified path, unless the folder already exsists.
    
    '''
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass #do nothing if the error occurs because the path already exists
        else: raise #re-raises the error

##############################################################################
########### extract voltage data from autosaves ##############################

def find_break_index(filepath, break_symbol =  "-----"):
    index = 0
    with open(filepath) as f:
        for line in f.readlines():
            index += 1
            if break_symbol in line:
                return index



def convert_col_label_to_index(header_line, col_labels_list):
    col_index_list = []
    split_header_line = line.split(sep)
    for n,i in enumerate(split_header_line):
        for ii in col_labels_list:
            if i.strip().lower() == ii.strip().lower():
                col_index_list.append(n)
    return col_index_list


def extract_target_data_from_file(filepath, **kwargs):
    '''target_columns, sep=',', index_col=0,
    parse_dates=True, encoding=None, tupleize_cols=False,
    infer_datetime_format=False):'''
    '''
    usecols can be a list of one or more column headers, a single column label, or column number(s) indexed from 0
    '''
    
    #at some point make it possible for target_columns to be a tuple
    #with a delimiter symbol and a string of column labels separated by delimiter
    usecols_is_string = False
    if isinstance(usecols, basestring):
        #if target_columns is a single string, put it a list
        usecols = [usecols]
        usecols_is_string = True
    elif isinstance(usecols, list) and isinstance(usecols[0],int):
        pass
    elif isinstance(usecols, list) and isinstance(usecols[0],basestring):
        usecols_is_string = True
    else:
        raise TypeError('usecols as passed to function extract_target_data_from_file() is of wrong type.')
    
    if usecols_is_string:
        #determine which column numbers correspond to the requested column labels
        column_label_index_lookup = {}
        column_index_label_lookup = {}
        with open(filepath) as f:
            #iterate thru until linenum specified by header
            row_counter = 0
            for line in f:
                if row_counter == header:
                    #upon finding header line, find position of strings in usecols
                    usecols = convert_col_label_to_index(line, usecols)
                    break #stop iterating thru file
                row_counter += 1

    column_index_label_lookup = {v: k for k, v in column_label_index_lookup.iteritems()}

    #extract columns
    df = pd.read_table(filepath, header=header, sep=sep,
                  parse_dates=parse_dates, index_col=index_col,
                  encoding=encoding, tupleize_cols=tupleize_cols,
                  infer_datetime_format=infer_datetime_format)

    return df

def extract_target_data_from_csv(filepath, column_target_label, header_indx=0):
    df = pd.DataFrame.from_csv(filepath, header_indx)
    return df[column_target_label]


'''
def extract_column_from_file(filepath, header_num, column_label):
    #get columns using index
    #determine which column number column_label is located at
    pd.read_table(filepath,',',header=header_num,usecols=[column_label])
'''


def search_param_header(filepath, params_list, break_symbol = "-----"):
    '''
    Open file and read thru parameters until break symbol found (store line num)
    Collect parameters specified in params_list and their values into label dictionary
    Return line num of break symbol and the parameter dictionary
    '''
    index = 0
    label = {}
    
    with open(filepath) as f:
        for line in f.readlines():
            index += 1
            if break_symbol in line:
                break
            for i in params_list:
                if i in line:
                    #separate key and value and remove whitespace
                    key, val = [ii.strip() for ii in line.split(":")]
                    label[key] = val

    return index, label

def load_param_header(filepath, break_symbol='-----'):
    """
    Load parameters in header into dictionary and return it. Also return index of header.
    """
    index = 0
    params = {}
    
    with open(filepath) as f:
        for line in f.readlines():
            index += 1
            if break_symbol in line:
                break
            else:
                key, val = [ii.strip() for ii in line.split(":")]
                params[key] = val
    return index, params

def load_data_file(filepath):
    index, params = load_param_header(filepath)
    pd.DataFrame(filepath, header=index)


def collate_target_data(file_list, key, param_names):
    '''
    Iterate thru files in file_list, extracting data type corresponding to key and the parameter values for each parameter in param_names.
    '''
    series_list = []
    for i in file_list:
        
        #find the parameters and the end of the parameter list
        index, label = parse_param_header(i, param_names)
        
        #make label for this file
        s_label = " ".join([v+k for k,v in label.iteritems()])
        
        #make dataframe
        df = pd.DataFrame.from_csv(i,index)
        
        #extract Vms and make a series with s_label as the heading
        Vms = df[key]
        s = pd.Series(data = Vms.values, index = Vms.index, name = s_label)
        series_list.append(s.iloc[0:-1])
    
    return series_list



##############################################################################


def alert(Freq = 1500,Dur = 500):
    '''
    On windows os, make beep at Freq Hz for Dur milliseconds.
    default is Freq @ 2500 Hertz and Dur @ 500 ms
    '''
    try:
        from winsound import Beep
        Beep(Freq,Dur)
    except ImportError as e:
        sys.stdout.write('\a')
        sys.stdout.flush()
    except Exception as e:
        pass

   
def get_folders_files(folder):
    dir_content = os.listdir(folder)
    dir_files = []
    dir_folders = []
    for f in dir_content:
        f = os.path.join(folder,f)
        if os.path.isdir(f):
            dir_folders.append(f)
        else:
            dir_files.append(f)
    return dir_folders, dir_files


def display_content(dir_files, data_type=''):
    #get dirname
    print "Folder: ",os.path.dirname(dir_files[0]), '\n'
    for n,i in enumerate(dir_files):
        file_name = os.path.basename(i)
        if data_type in file_name:
            print n,": ",i


def strings_containing(strings, elements):
    
    strings_w_all_elems = []
    for string in strings:
        #print 
        #print string
        num_elems = 0
        for elem in elements:
            #print "\t", elem,
            if elem in string:
                num_elems += 1
           
        if num_elems == len(elements):
            strings_w_all_elems.append(string)
            
    return strings_w_all_elems
def test_strings_containing():
    dummy = ['misc-TBModel-sec132-eL-IP0_9-gnaps1_0.txt.csv',\
    'misc-TBModel-sec1320-eL-IP0_95-gnaps1_0.txt.csv',\
    'misc-TBModel-sec1320-eL-IP0_9-gnaps1_0.txt.csv']

    print strings_containing(dummy, ['0_95','1320'])
    print strings_containing(dummy, ['1320'])
    print strings_containing(dummy, ['0_95'])

"""
def get_files(start, folder_crit=[], file_crit=[], verbose = True):
    folders = []
    curr_dir = os.getcwd()
    folder_num = 0
    for n, (root, dirs, files) in enumerate(os.walk(start)):
        
        #if root does not contain folder criteria, skip it
        if not strings_containing([root], folder_crit):
            continue
        
        if verbose: print "["+str(folder_num)+"]", root
        folder_num += 1
        file_paths = []
        
        goodfiles = strings_containing(files,file_crit)
        for nn,f in enumerate(goodfiles):
            #remove zip files
            if '.zip' in f:
                files.remove(f)
                continue
            file_path = "r'"+os.path.join(curr_dir,root, f).replace('C:','')+"'"
            if verbose: 
                print " ["+str(nn)+"]", file_path
            file_paths.append(file_path.lstrip('r').replace('\'',''))
        if verbose: 
            print
        if len(file_paths) > 0:
            folders.append(file_paths)
    return list(itertools.chain(*folders))
"""

@contextlib.contextmanager
def stdout_redirected(new_stdout):
    '''
    Redirect print statements and other standard output
    to a new standard output new_stdout
    '''
    save_stdout = sys.stdout
    sys.stdout = new_stdout
    try:
        yield None
    finally:
        sys.stdout = save_stdout


def load_multiCols(file_name):
    return pd.read_csv(file_name, index_col=[0], header=[0,1], skipinitialspace=True)

def extract_misc_data(file_name, show = False):
    '''
    Extracts labeled data from the passed file and returns a dict of labels: data.
    '''
    import json
    #Extract Data
    with open(file_name,'r') as f:
        misc_data = json.load(f)
    if show: print 'Items in misc file: ',misc_data.keys(),'\n'

    #put extracted data in a dict
    temp = {}
    for k,v in misc_data.iteritems():
        try:
            temp[k] = pd.DataFrame.from_dict(v)
            if show: print k,': ', type(temp[k])
        except:
            temp[k] = v
            if show: print k, ": ", v
    return temp


def extract_var_label(file_name, var_name):
    basename = os.path.basename(file_name)
    f_parts = os.path.splitext(basename)[0].split('-') #remove extension and split on '-'
    for part in f_parts:
        if var_name in part:
            label = part.replace(var_name,'').replace("_",".")
            return label
    return None

def load_label_multiCols(files, label_var):
    data = {}
    for f in files:
        basename = os.path.basename(f)
        label = extract_var_label(basename,'IP')
        data[label] = load_multiCols(f)
    return data

def count_bursts_all(bursts_dict, time=0):
    all_bursts_freq = {}
    for key, value in bursts_dict.iteritems():
        all_bursts_freq[key] = count_bursts(value,time)
    return all_bursts_freq

def count_bursts(burst_df, time=0):
    '''
    If you do not provide a time >= 0 only the count will be returned, not the overall frequency
    '''
    burst_start_all_keys = burst_df.xs(key='start',axis=1, level=1)
    bursts_per_sec = {}
    for k,v in burst_start_all_keys.iteritems():
        if time <= 0:
            bursts_per_sec[k] = len(v.dropna())
            print k, len(v.dropna())
        else:
            bursts_per_sec[k] = len(v.dropna())/time
    return bursts_per_sec



def get_files(start, folder_crit=[], file_crit=[], verbose = True):
    
    if not os.path.isdir(start):
        raise IOError("Directory '%s' not found"%start)
    
    folders = []
    curr_dir = os.getcwd()
    folder_num = 0
    for n, (root, dirs, files) in enumerate(os.walk(start)):
        
        #if root does not contain folder criteria, skip it
        if not strings_containing([root], folder_crit):
            continue
        
        if verbose: print "["+str(folder_num)+"]", root
        folder_num += 1
        file_paths = []
        
        goodfiles = strings_containing(files,file_crit)
        for nn,f in enumerate(goodfiles):
            #remove zip files
            if '.zip' in f:
                files.remove(f)
                continue
            file_path = "r'"+os.path.join(curr_dir,root, f).replace('C:','')+"'"
            if verbose: 
                print " ["+str(nn)+"]", file_path
            file_paths.append(file_path.lstrip('r').replace('\'',''))
        if verbose: 
            print
        if len(file_paths) > 0:
            folders.append(file_paths)
    return list(itertools.chain(*folders))



def copy_all_files(new_dir, files, verbose = False):

    #let user know that files must be a list of files
    #(b/c otherwise there will be an error below that is difficult to diagnose)
    if not type(files) is list:
        raise TypeError("Argument 'files' must be a list. Currently it is %s"%type(files))
    
    #will be passed to user upon completion
    new_files = []
    
    for i in files:
        base = os.path.basename(i)
        new_location = os.path.join(new_dir, base)
        try:
            #copy to new location, print success message, add new file location to list
            shutil.copy(i, new_location)
            if verbose: print "Made file %s" % new_location
            new_files.append(new_location)
            
        except IOError as e:
            if e.errno == errno.ENOENT and verbose: print 'Cannot find file "%s". '% i
                
        except Exception as e: 
            if verbose: print 'Copy of "%s" to new dir "%s" failed because %s.'%(i, new_dir, e)
                
        finally:
            if verbose: print "Made file %s" % new_location
            new_files.append(new_location)
            
    return new_files


#def process_path(path):
    
