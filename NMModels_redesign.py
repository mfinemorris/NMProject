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


#Define BRS and TB Models
class BRSModel(object):
    
    def __init__(self, param_dict={}):
        """
        Initialize a model with a set of parameters, or use the defaults, which are as follows:
        
        #### MEMBRANE CAPACITANCE (pF) ####
        Cms=21.
        
        #### REVERSAL POTENTIALS (mV) ####
        ena=50. # sodium potential (for normal sodium current and persistent sodium current)
        ek=-85. # potassium potential
        eL=-60.0 # leak reversal potential
        eSyn=0.0 # reversal potential of non-NMDA glutamatergic synaptic currents.
        
        #### CONDUCTANCE (nS) ####
        gk=11.2 # potassium current conductance
        gna=28. # sodium current conductance
        gnaps=2.8 # persistent sodium current conductance
        gL=2.8 # leak current conductance
        gtonic=0.0 # tonic conductance
        
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
        taunb=10 # tau max for n
        tauhb=10000 # tau max for h
        
        # Integration timestep (milliseconds)
        dt = .1
        
        #### INITIAL VALUES ####
        Vs0=-60.0 #initial somatic memberane voltage
        ns0=0.004 #initial value of gating var n
        hs0=0.33 #initial value of gating var h
        """
        self.load_vars(**param_dict)
    
    
    def load_vars(self, Cms=21., ena=50., ek=-85., eL=-60.0, eSyn=0.0, gk=11.2, gna=28., gnaps=2.8, gL=2.8, gtonic=0.0, vm=-34, vn=-29, vmp=-40, vh=-48, sm=-5, sn=-4, smp=-6, sh=5, taunb=10, tauhb=10000, Vs0=-60.0, ns0=0.004, hs0=0.33, Iaps=0, dt = .1):
   
        """
        Set model parameter values.
        """
   
        #### MEMBRANE CAPACITANCE (pF) ####
        self.Cms = Cms
        
        #### REVERSAL POTENTIALS (mV) ####
        # sodium potential (for normal sodium current and persistent sodium current)
        self.ena = ena
        # potassium potential
        self.ek = ek
        # leak reversal potential
        self.eL = eL
        # reversal potential of non-NMDA glutamatergic synaptic currents.
        self.eSyn = eSyn
        
        #### CONDUCTANCE (nS) ####
        # potassium current conductance
        self.gk = gk
        # sodium current conductance
        self.gna = gna
        # persistent sodium current conductance
        self.gnaps = gnaps
        # leak current conductance
        self.gL = gL
        # tonic conductance
        self.gtonic = gtonic
        
        #### APPLIED CURRENT (pA) ####
        self.Iaps = Iaps
        
        #### HALF (IN)ACTIVATION VOLTAGE (mV) ####
        self.vm = vm
        self.vn = vn
        self.vmp = vmp
        self.vh = vh
        
        #### INVERSE SLOPE MULTIPLIER (mV) ####
        self.sm = sm
        self.sn = sn
        self.smp = smp
        self.sh = sh
        
        #### TIME CONSTANT MAX VALUES (ms) #####
        self.taunb = taunb #tau max for n
        self.tauhb = tauhb #tau max for h
        
        # Integration timestep (milliseconds)
        self.dt = dt
        
        #### INITIAL VALUES ####
        self.Vs0 = Vs0 #initial somatic memberane voltage
        self.ns0 = ns0 #initial value of gating var n
        self.hs0 = hs0 #initial value of gating var h

        
        #### INITIAL STATE PARAMETER LIST ####
        self._initial_vars = [param for param in self.get_parameter_names() if param.endswith('0')]
        list.sort(self._initial_vars, key = str.lower)

    def get_parameter_names(self):
        '''
        Return a list of parameter names.
        '''
        return [attr for attr in dir(self) if not callable(attr) and not (attr.startswith("__") or attr.startswith("_"))]

    def print_vars_helper(self, cols=1, exclude=[]):
        '''
        prints parameters with less formatting
        Cols is how many columns to print the output in, 
        exclude is a list of strings, which should be parameter keys to exclude from the output
        '''
        print "{:<9s}: {:<10} \t ".format('Model', self.__class__.__name__)
        
        i = 1
        for key in self.get_parameter_names():
            if key in exclude:
                continue
            try:
                item = vars(self)[key]
            except (KeyError, AttributeError):
                item = self.__class__.__dict__[key]#is this necessary/does it do what I want?

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
        You can use print_vars to see a formatted list of parameter keys and values.
        '''
        return deepcopy(self._param_names)
    
    def model(self, y, t):
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
        #I_tonic = self.gtonic*(Vs-eSyn)

        #### DIFFERENTIAL EQUATIONS
        # SOMATIC EQUATIONS
        dVs = (-I_ks-I_nas-I_naps-I_L+self.Iaps)/self.Cms
        dns = (ninfs-ns)/tauns
        dhs = (hinfs-hs)/tauhs        
        
        return [dhs, dns, dVs]



    def jacobian(self,y,t):
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
    
    
    def _autosave(self, autosave_dir, sim_data):
        '''
        Make a folder for autosaves if none exists in this directory. 
        Dump all simulated data and parameters for the run into a txt file.
        '''
        
        NMUtility.mkdir_p(autosave_dir)
        
        #make time-stamped file name (include model name)
        filename = os.path.join(autosave_dir, str(self.__class__.__name__)+time.strftime("_%Y_%m_%d_%H_%M_%S.csv", time.localtime()))

        #get a string representation of the dataframe (DO NOT PASS A FILENAME TO to_csv()!!!)
        str_formatted_data = sim_data.to_csv()
        
        #save all model info (parameters) and output to txt file
        with open(filename, "w") as f:
            #redirect print statements so that they print to a file
            with NMUtility.stdout_redirected(f):
                self.print_vars_helper()
                print '--------------------'
                print str_formatted_data

                
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
        #make time array and initial state array for odeint function
        t = linspace(0,simulationTime, simulationTime/self.dt)
        initial_state = self._initial_state

        #must test jacobian upfront b/c odeint doesn't break out of loop upon error
        try:
            self.jacobian(initial_state,0.)
        except Exception as e:
            print "Could not use jacobian matrix because '{}'. Attempting simulation without jacobian matrix.".format(e)
            use_jacobian = False
            sys.stdout.flush()

        try:
            if use_jacobian:
                y = integrate.odeint(self.model, initial_state, t, Dfun=self.jacobian)
            else:
                y = integrate.odeint(self.model, initial_state, t)
            
            self.sim_data = pandas.DataFrame(y, index = t, columns = self._initial_vars)

        except KeyboardInterrupt:
            #kinda pointless because odeint doesn't actually exit upon keyboard interrupt
            #this section exists because someday I would like to make own integrator that
            #does actually exit and return the data upon interrupt (or if odeint ever gets fixed).
            if autosave_dir:
                self._autosave(autosave_dir, self.sim_data)
            raise
        
        except Exception as e:
            print "Could not run simulation. Exceptions: {}".format(e)
            raise

        #extract membrane voltage
        Vs_index = self._initial_vars.index('Vs0')
        V = y.T[Vs_index] #extract membrane voltage
        
        #save data to filename based on date and time
        if autosave_dir:
            sim_data = pandas.DataFrame(data=y, index=t, columns=self._initial_vars)
            self._autosave(autosave_dir, self.sim_data)

        return V, t

BRSModel()


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
    #eSyn=0.0
    
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
    #gtonic=0.0
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

    def model(self, y, t):

        Ca, hs, l, ns, Vd, Vs = y
    
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
        #I_tonic = self.gtonic*(Vs-eSyn)
        # currents in Dendrite
        I_can = self.gcan*caninf*(Vd-self.ena) # calcium current
        I_ds = self.gc*(Vd-Vs)/self.k # modification of somatic current due to dendritic current

        #### DIFFERENTIAL EQUATIONS
        # SOMATIC EQUATIONS
        dVs = (-I_ks-I_nas-I_naps-I_L-I_sd+self.Iaps)/self.Cms
        dns = (ninfs-ns)/tauns
        dhs = (hinfs-hs)/tauhs
        
        # DENDRITIC EQUATIONS
        dVd = (-I_can-I_ds)/self.Cmd
        dCa = (self.fi/self.Vi)*( J_ER_in - J_ER_out)
        dl = self.A*( self.Kd - (Ca + self.Kd)*l )

        dy = [dCa, dhs, dl, dns, dVd, dVs]
        return dy

    def jacobian(self, y, t):
        raise NotImplementedError('Jacobian not implemented for {}.'.format(self.__class__.__name__))


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
    #eSyn=0.0
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
    #gtonic=0.0
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
    
    def model(self, y, t):
        
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
        
        #I_tonic = self.gtonic*(Vs-eSyn)
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
        dVs = (-I_ks-I_nas-I_naps-I_L-I_atp-I_sd+self.Iaps)/self.Cms
        dns = (ninfs-ns)/tauns
        dhs = (hinfs-hs)/tauhs
        
        # DENDRITIC EQUATIONS
        dVd = (-I_can-I_ds)/self.Cmd
        dCa = (self.fi/self.Vi)*( J_ER_in - J_ER_out)
        dl = self.A*( self.Kd - (Ca + self.Kd)*l )

        
        dy = [datp, dc1, dc2, dc3, dc4, dCa, dhs, dl, dns, dq1, dq2, dq3, dq4, dVd, dVs]
        return dy
