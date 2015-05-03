from numpy import *
import matplotlib.pyplot as plt
import matplotlib
import pandas
from scipy import integrate
from time import clock, strftime, localtime
import os, time, sys
import NMResources
import NMUtility

set_printoptions(4)
'''
want to make an option to run from most recent sim, but retain the original initial var values. (So make _initial_state in self.simulate).
'''
#Define BRS and TB Models
class BRSModel(object):

    #### MEMBRANE CAPACITANCE (pF) ####
    Cms=21.

    #### REVERSAL POTENTIALS (mV) ####
    # sodium potential (for normal sodium current and persistent sodium current)
    ena=50.
    # potassium potential
    ek=-85.
    # leak reversal potential
    eL=-60.0 # suggested value for vleak from Butera paper are -60,-57.5,-54
    # reversal potential of non-NMDA glutamatergic synaptic currents.
    eSyn=0.0

    #### CONDUCTANCE (nS) ####
    # potassium current conductance
    gk=11.2
    # sodium current conductance
    gna=28.
    # persistent sodium current conductance
    gnaps=2.8 #should this be 1.5 or 2.0 as in Toparikova and Butera 2011?
    # leak current conductance
    gL=2.8
    # tonic conductance
    gtonic=0.0

    #### APPLIED CURRENT (pA) ####
    Iaps=0

    #### HALF (IN)ACTIVATION VOLTAGE (mV) ####
    vm=-34
    vn=-29
    vmp=-40
    vh=-48

    #### INVERSE SLOPE MULTIPLIER (mV) ####
    sm=-5
    sn=-4
    smp=-6
    sh=5

    #### TIME CONSTANT MAX VALUES (ms) #####
    taunb=10 #tau max for n
    tauhb=10000 #tau max for h
    
    # Integration timestep (milliseconds)
    dt = .1
    
    #### INITIAL VALUES ####
    Vs0=-60.0 #initial somatic memberane voltage
    ns0=0.004 #initial value of gating var n
    hs0=0.33 #initial value of gating var h
    
    #### MODEL PARAMETER LIST ####
    #set here so that only the variables are in param_names
    _param_names = [i for i in dir() if "__" not in i]
    _initial_vars = [name for name in _param_names if name[-1]=='0']
    list.sort(_initial_vars, key = str.lower)
    
    #let user know if parameter dictionary passed in to load_vars is incomplete
    warn_for_missing_vars = False
    
    def __init__(self, param_dict={}, warn=False):
        """
        Initialize a model with a set of parameters, or use the defaults.
        If true, warn will inform the user if the dict they passed (or any future parameter dict)
        is incomplete.
        """
        #load model with parameters
        self.load_vars(param_dict)
        self.warn_for_missing_vars = warn
        self.sim_data = None #holder for past simulation
        self.curr_time = 0.0
    
    @classmethod
    def _load_param_header(cls, filepath, break_symbol='-----'):
        """Load parameters in header into dictionary and return it. Also return index of header."""
    
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

    @classmethod
    def load_data_file(cls, filepath, warn = False):
        
        #get parameters from file and locate header index
        index, params = cls.load_param_header(filepath)
        model = params.pop('Model')
        if model != cls.__name__:
            raise TypeError('Save file is for model class {}, not model class {}'.format(model, cls.__name__))
        
        model_object = cls(params, warn)
        model_object.sim_data = pd.DataFrame(filepath, header=index)
        #check to make sure the number of columns is equal to the number of items in _initial_vars

        return model_object
    
    
    def load_vars(self, param_dict, warn = warn_for_missing_vars):
        """
        Set model parameter values.
        Param_dict should be a dictionary where the keys are names for the class variables,
        and the values are whatever you want those variables set too.
        This is mostly for loading many parameter values at once, if you only want to change one,
        consider using simple dot notation because this is not very efficient.
        """
        
        for i in self._param_names:
            # if the dict is empty don't do anything
            if len(param_dict) < 1:
                break
            try:
                self.__dict__[i] = float(param_dict[i])
            except (KeyError, AttributeError) as e:
                #if is missing from var dictionary, let user know if warn is True
                if  warn:
                    print ("Param "+i+" is either missing or invalid. Using default.")


    def print_vars_helper(self, cols=1, exclude=[]):
        '''
        Prints parameters with less formatting.
        Cols is how many columns into which to organize the output,
        exclude is a list of strings, which should be parameter keys to exclude from the output.
        '''
        print "{:<9s}: {:<10} \t ".format('Model', self.__class__.__name__)
        
        i = 1
        for key in self._param_names:
            if key in exclude:
                continue
            try:
                item = vars(self)[key]
            except (KeyError, AttributeError):
                item = self.__class__.__dict__[key]

            print "{:<9s}: {:<10} \t ".format(key, item),
            if i%cols == 0:
                print
            i +=1
    
    def print_vars(self,cols = 3):
        '''
        Print the model parameters in an easy to read format. 
        Set cols to the number of columns in which you would like the variables to appear.
        '''
        
        line = "-----------------------"
        
        #print string to indicate the start of the param list
        print line + "\n"+"Model Parameters: "+"\n"+line+"\n"
        
        #collect and print
        iv = []
        for i in self._initial_vars:
            try:
                val = str(round(self.__dict__[i],3))
            except:
                val = str(round(self.__class__.__dict__[i],3))
            iv.append("%s = %s"%(i,val))
        print "Initial state: "+(", ".join(iv))
        print
        self.print_vars_helper(cols, self._initial_vars) #print all constant params
        
        #print string to indicate the end of the param list
        print "\n"+line + "\n"+"End Parameters "+"\n"+line+'\n'
    
    
    def get_parameter_keys(self):
        '''
        Returns a list of the model parameters that can be set with setVars. 
        You can use printVars to see a formatted list of parameter keys and values.
        '''
        return deepcopy(self._param_names)
    
    def _model(self, y, t):
        '''
        Primarily for internal use by the simulate function. y is the cell state from the previous time step 
        (consisting of an array with h,n,and Vs, in that order) and t is the time at which this timestep occurs.
        '''
        
        hs, ns, Vs = y
        
        # SOMATIC FUNCTIONS
        minfs = 1/(1+exp((Vs-self.vm) /self.sm))
        ninfs = 1/(1+exp((Vs-self.vn) /self.sn))
        minfps = 1/(1+exp((Vs-self.vmp)/self.smp))
        hinfs = 1/(1+exp((Vs-self.vh) /self.sh))

        tauns = self.taunb/cosh((Vs-self.vn)/(2*self.sn))
        tauhs = self.tauhb/cosh((Vs-self.vh)/(2*self.sh))

        # CURRENT EXPRESSIONS
        # currents in Soma
        I_nas = self.gna*(minfs**3)*(1-ns)*(Vs-self.ena) #sodium current
        I_ks = self.gk*(ns**4)*(Vs-self.ek) #potassium current
        I_naps = self.gnaps*minfps*hs*(Vs-self.ena) #persistent sodium current
        I_L = self.gL*(Vs-self.eL) #pA
        I_tonic = self.gtonic*(Vs-self.eSyn)

        #### DIFFERENTIAL EQUATIONS
        # SOMATIC EQUATIONS
        dVs = (-I_ks-I_nas-I_naps-I_L-I_tonic+self.Iaps)/self.Cms
        dns = (ninfs-ns)/tauns
        dhs = (hinfs-hs)/tauhs        
        
        return [dhs, dns, dVs]



    def _jacobian(self,y,t):
        '''
        Primarily for internal use by the simulate function. y is the cell state from the previous time step 
        (consisting of an array with Vs, n, and h) and t is the time at which this timestep occurs. 
        '''
        h, n, Vs = y
        j33 = (-h*self.gnaps/(exp((Vs - self.vmp)/self.smp) + 1) + h*self.gnaps*(Vs - self.ena)*exp((Vs - self.vmp)/self.smp)/(self.smp*(exp((Vs - self.vmp)/self.smp) + 1)**2) - n**4*self.gk - self.gL - self.gna*(-n + 1)/(exp((Vs - self.vm)/self.sm) + 1)**3 + 3*self.gna*(Vs - self.ena)*(-n + 1)*exp((Vs - self.vm)/self.sm)/(self.sm*(exp((Vs - self.vm)/self.sm) + 1)**4))/self.Cms
        j32 =  (-4*n**3*self.gk*(Vs - self.ek) + self.gna*(Vs - self.ena)/(exp((Vs - self.vm)/self.sm) + 1)**3)/self.Cms
        j31 = -self.gnaps*(Vs - self.ena)/(self.Cms*(exp((Vs - self.vmp)/self.smp) + 1))
        j23 = (-n + 1/(exp((Vs - self.vn)/self.sn) + 1))*sinh((Vs - self.vn)/(2*self.sn))/(2*self.sn*self.taunb) - exp((Vs - self.vn)/self.sn)*cosh((Vs - self.vn)/(2*self.sn))/(self.sn*self.taunb*(exp((Vs - self.vn)/self.sn) + 1)**2)
        j22 = -cosh((Vs - self.vn)/(2*self.sn))/self.taunb
        j21 = 0
        j13 = (-h + 1/(exp((Vs - self.vh)/self.sh) + 1))*sinh((Vs - self.vh)/(2*self.sh))/(2*self.sh*self.tauhb) - exp((Vs - self.vh)/self.sh)*cosh((Vs - self.vh)/(2*self.sh))/(self.sh*self.tauhb*(exp((Vs - self.vh)/self.sh) + 1)**2)
        j12 = 0
        j11 = -cosh((Vs - self.vh)/(2*self.sh))/self.tauhb
        
        return [[j11, j12, j13], [j21, j22, j23], [j31,j32,j33]] 
    
    
    def _autosave(self, autosave_dir):
        '''
        Make a folder for autosaves if none exists in this directory. 
        Dump all simulated data and parameters for the run into a txt file.
        '''
        
        NMUtility.mkdir_p(autosave_dir)
        
        #make time-stamped file name (include model name)
        filename = os.path.join(autosave_dir, str(self.__class__.__name__)+time.strftime("_%Y_%m_%d_%H_%M_%S.csv", time.localtime()))
        self.save(filename)
    

    def save(self, filepath):
        
        #get a string representation of the dataframe (DO NOT PASS A FILENAME TO to_csv()!!!)
        str_formatted_data = self.sim_data.to_csv()
        
        #save all model info (parameters) and output to txt file
        with open(filepath, "w") as f:
            
            #redirect print statements so that they print to a file
            with NMUtility.stdout_redirected(f):
                self.print_vars_helper()
                print '--------------------'
                print str_formatted_data
            print "Wrote data to file at {}".format(filepath)


    def _make_initial_state_array(self):
        try:
            return list(self.sim_data.iloc[-1])
        except:
            return [(eval("self."+i)) for i in self._initial_vars]


    def simulate(self, simulationTime, use_jacobian=True, autosave_dir = 'autosaved_sim_data'):
        '''
        Simulates the model for number of milliseconds indicated by simulationTime. 
        Returns two arrays, V and t: t is the array of time points at which the system was calculated
        and V is the membrane voltage found at each of the time points
        Create dir specified in autosave_dir (do not overwrite if it already exists),
        autosaves all traces and model parameters to uniquely named, time-stamped files in that dir.
        Note that a value for autosave_dir is already specified. To prevent autosaving set
        autosave_dir to '' or None.
        '''
        t = linspace(self.curr_time,self.curr_time+simulationTime, simulationTime/self.dt) #make time array and initial state array for odeint function
        initial_state = self._make_initial_state_array() #Save initial state variables (needs to be a list to preserve order)
        column_labels =[name.strip('0') for name in self._initial_vars] #column names for saving the data
        print
        print "t = ",self.curr_time
        print initial_state
        print
        #must test jacobian and model functions upfront b/c odeint doesn't break out of loop upon error
        if use_jacobian:
            try:
                self._jacobian(initial_state,0.)
                self._model(initial_state, 0.)
                y = integrate.odeint(self._model, initial_state, t, Dfun=self._jacobian)
            except NotImplementedError as e:
                print "Could not run simulation with the jacobian matrix because '{}'. Attempting simulation without jacobian matrix.".format(e)
                use_jacobian = False
                sys.stdout.flush()

        if not use_jacobian:
            try:
                y = integrate.odeint(self._model, initial_state, t)
            except Exception as e:
                print "Could not run simulation. Exceptions: {}".format(e)
                raise

        self.curr_time = t[-1]
        print "t = ",self.curr_time

        try:
            temp = pandas.DataFrame(data=y, index=t, columns=column_labels)
            self.sim_data.append(temp)
            print "append to sim data"
        except AttributeError:
            
            self.sim_data = pandas.DataFrame(data=y, index=t, columns=column_labels)
            print "sim data"
        
        #extract membrane voltage calculated during this run
        Vs_index = self._initial_vars.index('Vs0')
        V = y.T[Vs_index] #extract membrane voltage
        
        #save data to filename based on date and time
        if autosave_dir:
            self._autosave(autosave_dir)

        return V, t  
'''
brs = BRSModel()
V,t = brs.simulate(1000.)
print brs.sim_data

V,t = brs.simulate(1000.)
#print brs.sim_data
'''


















class TBModel(BRSModel):
        
    #### MEMBRANE CAPACITANCE (pF) ####
    Cms=21.
    #dendrite membrane capacitance
    Cmd=5.

    #### REVERSAL POTENTIALS (mV) ####
    # sodium potential (for normal sodium current and persistent sodium current)
    ena=50.
    # potassium potential
    ek=-85.
    # leak reversal potential
    eL=-60.0 # suggested value for vleak from Butera paper are -60,-57.5,-54
    # reversal potential of non-NMDA glutamatergic synaptic currents.
    eSyn=0.0
    
    #### CONDUCTANCE (nS) ####
    # potassium current conductance
    gk=11.2
    # sodium current conductance
    gna=28.
    # persistent sodium current conductance
    gnaps=2.8#1.5#2.8 #should this be 1.5 or 2.0 as in Toparikova and Butera 2011?
    # leak current conductance
    gL=2.3 #in Toporikova_Butera_2010_code.py
    # tonic conductance
    gtonic=0.0
    # calcium channel conductance
    gcan = 1.5
    # gc (the conductance for the link terms)
    gc = 1.0

    #### APPLIED CURRENT (pA) ####
    Iaps=0

    #### HALF (IN)ACTIVATION VOLTAGE (mV) ####
    vm=-34.
    vn=-29.
    vmp=-40.
    vh=-48.

    #### INVERSE SLOPE MULTIPLIER (mV) ####
    sm=-5.
    sn=-4.
    smp=-6.
    sh=5.

    #### TIME CONSTANT MAX VALUES (ms) #####
    taunb=10. #tau max for n
    tauhb=10000. #tau max for h

    #### Constants for calculating Ca Flux: ER --> Cytosol ####
    IP=1. #IP3 concentration
    LL=0.37 #ER leak permeability
    P=31000. #maximum total permeability of IP3 channels
    Ki=1.0 #dissociation consts for IP3 receptor activity by IP3
    Ka=0.4 #dissociation consts for IP3 receptor activity by Ca

    #### Constants for calculating Ca Flux: Cytosol --> ER ####
    Ve=400. #Maximal SERCA pump rate
    Ke=0.2 #coefficient for SERCA pumps

    #### ER Ca CONCENTRATION ####
    Ct=1.25 #Total Ca
    sigma=0.185 #ratio of cytosolic to ER volume

    #The ER parameters
    fi=0.0001 #bound Ca concentration in cytosol
    Vi=4. #free Ca concentration in cytosol
    A=0.005 #scaling const.
    Kd=0.4 #dissociation constant for IP3 receptor inactivation by Ca

    #### Calcium current activation ####
    Kcan=0.74 # microM 
    ncan=0.97

    #### Ratio of somatic to total area ####
    k=0.3
    
    # In milliseconds
    dt = 0.1
    
    Vs0=-60.0 #initial somatic memberane voltage
    ns0=0.004 #initial value of gating var n
    hs0=0.33 #initial value of gating var h
    Vd0=-50. #initial dendritic membrane voltage
    Ca0=0.03 #initial calcium 2+ balance
    l0=0.93 # initial value of IP3 channel gating variable
    
    #set here so that only the variables are in param_names
    _param_names = sorted([i for i in dir() if "__" not in i])
    _initial_vars = [name for name in _param_names if name[-1]=='0']
    list.sort(_initial_vars, key = str.lower)
    
    #let user know if parameter dictionary passed in to setVars is incomplete
    warn_for_missing_vars = False
    
    def __init__(self, param_dict={}):
        super(TBModel, self).__init__(param_dict)
        self.load_vars(param_dict)

    def _model(self, y, t):

        Ca, hs, l, ns, Vd, Vs = y
    
        # SOMATIC FUNCTIONS
        # gating variables
        minfs = 1/(1+exp((Vs-self.vm) /self.sm))
        ninfs = 1/(1+exp((Vs-self.vn) /self.sn))
        minfps = 1/(1+exp((Vs-self.vmp)/self.smp))
        hinfs = 1/(1+exp((Vs-self.vh) /self.sh))
        
        # time constants
        tauns = self.taunb/cosh((Vs-self.vn)/(2*self.sn))
        tauhs = self.tauhb/cosh((Vs-self.vh)/(2*self.sh))

        # DENDRITIC FUNCTIONS
        # Calculate ER Ca
        Ce = (self.Ct - Ca)/self.sigma
        # Flux of Ca from ER to cytosol(regulated by IP3 receptors)
        J_ER_in=(self.LL + self.P*((self.IP*Ca*l/((self.IP+self.Ki)*(Ca+self.Ka)))**3))*(Ce - Ca)
        # Flux from cytosol back to ER (controlled by SERCA pumps)
        J_ER_out=self.Ve*(Ca**2)/((self.Ke**2)+(Ca**2))
        # Activation of calcium current (I_can)
        caninf = 1/(1+((self.Kcan/Ca)**self.ncan))

        # CURRENT EXPRESSIONS
        # currents in Soma
        I_nas = self.gna*(minfs**3)*(1-ns)*(Vs-self.ena) #sodium current
        I_ks = self.gk*(ns**4)*(Vs-self.ek) #potassium current
        I_naps = self.gnaps*minfps*hs*(Vs-self.ena) #persistent sodium current
        I_L = self.gL*(Vs-self.eL) #pA
        I_sd = self.gc*(Vs-Vd)/(1-self.k) # modification of dendritic current due to somatic current
        I_tonic = self.gtonic*(Vs-self.eSyn)
        # currents in Dendrite
        I_can = self.gcan*caninf*(Vd-self.ena) # calcium current
        I_ds = self.gc*(Vd-Vs)/self.k # modification of somatic current due to dendritic current

        #### DIFFERENTIAL EQUATIONS
        # SOMATIC EQUATIONS
        dVs = (-I_ks-I_nas-I_naps-I_L-I_sd-I_tonic+self.Iaps)/self.Cms
        dns = (ninfs-ns)/tauns
        dhs = (hinfs-hs)/tauhs
        
        # DENDRITIC EQUATIONS
        dVd = (-I_can-I_ds)/self.Cmd
        dCa = (self.fi/self.Vi)*( J_ER_in - J_ER_out)
        dl = self.A*( self.Kd - (Ca + self.Kd)*l )

        dy = [dCa, dhs, dl, dns, dVd, dVs]
        return dy

    def _jacobian(self, y, t):
        raise NotImplementedError('Jacobian not implemented for {}'.format(self.__class__.__name__))



class YanModel(TBModel):
    
    #### MEMBRANE CAPACITANCE (pF) ####
    Cms=21.
    #dendrite membrane capacitance
    Cmd=5.
    
    #### REVERSAL POTENTIALS (mV) ####
    # sodium potential (for normal sodium current and persistent sodium current)
    ena=50.
    # potassium potential
    ek=-85.
    # leak reversal potential
    eL=-60.0 # suggested value for vleak from Butera paper are -60,-57.5,-54
    # reversal potential of non-NMDA glutamatergic synaptic currents.
    eSyn=0.0
    #P2X7 channel reversal potential
    eatp = 0.0 #mV
    
    #### CONDUCTANCE (nS) ####
    # potassium current conductance
    gk=11.2
    # sodium current conductance
    gna=28.
    # persistent sodium current conductance
    gnaps=2.8#1.5#2.8 #should this be 1.5 or 2.0 as in Toparikova and Butera 2011?
    # leak current conductance
    gL=2.3 #in Toporikova_Butera_2010_code.py
    # tonic conductance
    gtonic=0.0
    # calcium channel conductance
    gcan = 1.5
    # gc (the conductance for the link terms)
    gc = 1.0
    #conductance of unsensitized P2X7 channels
    g12 = 15 # nS
    #conductance of sensitized P2X7 channels
    g34 = 45 # nS
    
    #### APPLIED CURRENT (pA) ####
    Iaps=0
    
    #### HALF (IN)ACTIVATION VOLTAGE (mV) ####
    vm=-34.
    vn=-29.
    vmp=-40.
    vh=-48.
    
    #### INVERSE SLOPE MULTIPLIER (mV) ####
    sm=-5.
    sn=-4.
    smp=-6.
    sh=5.
    
    #### TIME CONSTANT MAX VALUES (ms) #####
    taunb=10. #tau max for n
    tauhb=10000. #tau max for h
    
    #### P2X7 RECEPTOR ####
    # RATE CONSTANTS
    # Back rates (ms)**-1
    k1 = 0.3e-3
    k3 = 2.4e-3
    k5 = 1.58e-3
    # Forward rates (ms*M)**-1
    k2 = 40000.0e-3
    k4 = 50000.0e-3
    k6 = 7000.0e-3
    # Sensitized/unsensitized rates
    L1 = 0.0001e-3
    L2 = 0.004e-3
    L3 = 0.5e-3
    
    #### ICAN CURRENT ####
    ## Constants for calculating Ca Flux: ER --> Cytosol ##
    IP=1. #IP3 concentration
    LL=0.37 #ER leak permeability
    P=31000. #maximum total permeability of IP3 channels
    Ki=1.0 #dissociation consts for IP3 receptor activity by IP3
    Ka=0.4 #dissociation consts for IP3 receptor activity by Ca
    
    ## Constants for calculating Ca Flux: Cytosol --> ER ##
    Ve=400. #Maximal SERCA pump rate
    Ke=0.2 #coefficient for SERCA pumps
    
    ## ER Ca CONCENTRATION ##
    Ct=1.25 #Total Ca
    sigma=0.185 #ratio of cytosolic to ER volume
    
    ##The ER parameters
    fi=0.0001 #bound Ca concentration in cytosol
    Vi=4. #free Ca concentration in cytosol
    A=0.005 #scaling const.
    Kd=0.4 #dissociation constant for IP3 receptor inactivation by Ca
    
    ## Calcium current activation ##
    Kcan=0.74 # microM
    ncan=0.97
    
    ## Ratio of somatic to total area ##
    k=0.3
    
    ## Time step ##
    dt = 0.1 #(ms)
    
    #### INITIAL VARIABLES
    Q10=0
    Q20=0
    Q30=0
    Q40=0
    ATP0=3.2
    C10=1
    C20=0
    C30=0
    C40=0
    
    Vs0=-60.0 #initial somatic memberane voltage
    ns0=0.004 #initial value of gating var n
    hs0=0.33 #initial value of gating var h
    Vd0=-50. #initial dendritic membrane voltage
    Ca0=0.03 #initial calcium 2+ balance
    l0=0.93 # initial value of IP3 channel gating variable
    
    
    #set here so that only the variables are in param_names
    _param_names = sorted([i for i in dir() if "__" not in i])
    _initial_vars = [name for name in _param_names if name[-1]=='0']
    list.sort(_initial_vars, key = str.lower)
    
    #let user know if parameter dictionary passed in to setVars is incomplete
    warn_for_missing_vars = False
    
    def __init__(self, param_dict={}):
        super(TBModel, self).__init__(param_dict)
        self.load_vars(param_dict)
        self._atp_derivative = lambda atp, t: 0.0
    
    def _model(self, y, t):
        
        atp, c1, c2, c3, c4, Ca, hs, l, ns, q1, q2, q3, q4, Vd, Vs = y
        
        # SOMATIC FUNCTIONS
        minfs = 1/(1+exp((Vs-self.vm) /self.sm))
        ninfs = 1/(1+exp((Vs-self.vn) /self.sn))
        minfps = 1/(1+exp((Vs-self.vmp)/self.smp))
        hinfs = 1/(1+exp((Vs-self.vh) /self.sh))
        
        tauns = self.taunb/cosh((Vs-self.vn)/(2*self.sn))
        tauhs = self.tauhb/cosh((Vs-self.vh)/(2*self.sh))
        
        # DENDRITIC FUNCTIONS
        #Calculate ER Ca
        Ce = (self.Ct - Ca)/self.sigma
        # Flux of Ca from ER to cytosol(regulated by IP3 receptors)
        J_ER_in=(self.LL + self.P*((self.IP*Ca*l/((self.IP+self.Ki)*(Ca+self.Ka)))**3))*(Ce - Ca)
        # Flux from cytosol back to ER (controlled by SERCA pumps)
        J_ER_out=self.Ve*(Ca**2)/((self.Ke**2)+(Ca**2))
        # Activation of calcium current (I_can)
        caninf = 1/(1+((self.Kcan/Ca)**self.ncan))
        
        # CURRENT EXPRESSIONS
        # currents in Soma
        I_nas = self.gna*(minfs**3)*(1-ns)*(Vs-self.ena) #sodium current
        I_ks = self.gk*(ns**4)*(Vs-self.ek) #potassium current
        I_naps = self.gnaps*minfps*hs*(Vs-self.ena) #persistent sodium current
        I_L = self.gL*(Vs-self.eL) #pA
        I_sd = self.gc*(Vs-Vd)/(1-self.k) # modification of dendritic current due to somatic current
        I_atp = (self.g12*(q1+q2)*(Vs-self.eatp) + self.g34*(q3+q4)*(Vs-self.eatp))
        I_tonic = self.gtonic*(Vs-self.eSyn)
        # currents in Dendrite
        I_can = self.gcan*caninf*(Vd-self.ena) # calcium current
        I_ds = self.gc*(Vd-Vs)/self.k # modification of somatic current due to dendritic current
        
        # atp concentration
        datp = self._atp_derivative(atp, t)
        dq1 = 2.*self.k4*atp*c2 + 3.*self.k5*q2 - (2.*self.k3 + self.k6*atp)*q1
        dq2 = self.k6*atp*q1 + self.L2*q3 - (3.*self.k5 + self.L3)*q2
        dq3 = self.k2*atp*q4 + self.L3*q2 -(3.*self.k1 + self.L2)*q3
        dq4 = 2.*self.k2*atp*c3 + 3.*self.k1*q3 - (2.*self.k1 + self.k2*atp)*q4
        dc1 = self.k1*c2 + self.L1*c4 - 3.*self.k2*atp*c1
        dc2 = 3.*self.k2*atp*c1 + 2.*self.k3*q1 - (self.k1+2.*self.k4*atp)*c2
        dc3 = 3*self.k2*atp*c4 + 2.*self.k1*q4 - (self.k1 + 2.*self.k2*atp)*c3
        dc4 = self.k1*c3 - (self.L1 + 3.*self.k2*atp)*c4
        
        #### DIFFERENTIAL EQUATIONS
        # SOMATIC EQUATIONS
        dVs = (-I_ks-I_nas-I_naps-I_L-I_atp-I_sd-I_tonic+self.Iaps)/self.Cms
        dns = (ninfs-ns)/tauns
        dhs = (hinfs-hs)/tauhs
        
        # DENDRITIC EQUATIONS
        dVd = (-I_can-I_ds)/self.Cmd
        dCa = (self.fi/self.Vi)*( J_ER_in - J_ER_out)
        dl = self.A*( self.Kd - (Ca + self.Kd)*l )

        
        dy = [datp, dc1, dc2, dc3, dc4, dCa, dhs, dl, dns, dq1, dq2, dq3, dq4, dVd, dVs]
        return dy
    



plt.close('all')

def plot_states(ax, Qt, t):
    labels = Qt.index
    q1,q2,q3,q4 = Qt.values
    ax.plot(t, q1, label=labels[0])
    ax.plot(t, q2, label=labels[1])
    ax.plot(t, q3, label=labels[2])
    ax.plot(t, q4, label=labels[3])
    
    ax.legend()
    
    return plt.gca()



def test(ym):
    
    runtime = 1000.*100

    V, t = ym.simulate(runtime)
    df = ym.sim_data
    print "Sim done"

    Q = df.loc[0.0:,'Q10':'Q40']
    C = df.loc[0.0:,'C10':'C40']
    
    Qt = Q.T
    Ct = C.T
    
    fig = plt.figure()
    '''
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(223)
    '''
    ax1 = plt.subplot2grid((2,2), (0,0), colspan=2)
    ax2 = plt.subplot2grid((2,2), (1,0))
    ax3 = plt.subplot2grid((2,2), (1, 1))


    ax1.plot(t,V)
    
    plot_states(ax2,Qt,t)
    plot_states(ax3,Ct,t)

    plt.tight_layout()
    return plt.gca()
    
    '''
    plt.subplot(121)
    plot_states(Qt,t)
    plt.subplot(122)
    plot_states(Ct,t)
    plt.savefig("testing_p2x7/QC_BzATP"+str(BzATP)+".jpeg")
    plt.close()
    '''

'''

ligand = .32
for i in [1,10,20,30,40,50]:
    BzATP=ligand/i
    ym = YanModel({'ATP0':BzATP, 'gnaps': 2.8, 'IP':0.97})
    test(ym)
    plt.savefig("testing_p2x7/BzATP"+str(BzATP)+".jpeg")
    plt.close()
#plt.show()
'''

def continue_from_file(filepath):
    pd.DataFrame(filepath)


def compare_yan_tb(time, yan_params, tb_params):

            ym = YanModel(yan_params)
            ym.simulate(runtime, False)
            
            tb = TBModel(tb_params)
            tb.simulate(runtime, False)






if __name__ == "__main__":

    first_list = [1.8, 2.4, 3.0, 3.6, 4.2]
    second_list = [-65, -60, -55, -50]
    runtime = 1000.*60*14
    
    runs_remaining = len(first_list)*len(second_list)
    for i in first_list:
        for ii in second_list:
            print runs_remaining, i," :\t", ii
            compare_yan_tb(runtime, yan_params = {'eL':ii, 'gnaps': i, 'IP':0.97, 'ATP0':10**-6},tb_params = {'eL':ii,'gnaps': i, 'IP':0.97})
            runs_remaining -= 1



