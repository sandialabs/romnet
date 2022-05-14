import sys
print(sys.version)
import os
import numpy as np
import pandas as pd
import time

import pyDOE
import cantera as ct
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# WORKSPACE_PATH = os.environ['WORKSPACE_PATH']
WORKSPACE_PATH = os.getcwd()+'/../../../../../'

# import matplotlib.pyplot as plt
# plt.style.use(WORKSPACE_PATH+'/ROMNet/romnet/extra/postprocessing/presentation.mplstyle')




##########################################################################################
### Input Data

### HYDROGEN
OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_5000Cases_H2_Iter/'
Fuel0              = 'H2:1.0'         
Oxydizer0          = 'O2:1.0, N2:4.0'
#t0                 = 1.e-5
#Deltat             = 1.e-5
DeltatMax          = 1.e-6
DeltatMin          = 1.e-4
tEnd               = 1.e-1
KeepVec            = None #['H2','H','O','O2','OH','H2O','HO2','H2O2','N','NH','NH2','NH3','NNH','NO','NO2','N2O','HNO','N2']

# ### METHANE
# OutputDir          = WORKSPACE_PATH + '/ROMNet/Data/0DReact_Isobaric_500Cases_CH4/'
# Fuel0              = 'CH4:1.0'
# Oxydizer0          = 'O2:0.21, N2:0.79'
# t0                 = 1.e-6
# tEnd               = 1.e2
# KeepVec            = None

MixtureFile        = 'gri30.yaml'
P0                 = ct.one_atm

NtInt              = 5000
Integration        = 'Canteras'
delta_T_max        = 1.
# Integration        = ''
# rtol               = 1.e-12
# atol               = 1.e-8
# SOLVER             = 'BDF'#'RK23'#'BDF'#'Radau'

# FIRST TIME
DirName            = 'train'
n_ics              = 5000
T0Exts             = np.array([900., 1100], dtype=np.float64)
EqRatio0Exts       = np.array([0.90, 1.10], dtype=np.float64)
# T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# EqRatio0Exts       = np.array([.5, 4.], dtype=np.float64)
X0Exts             = None #np.array([0.05, 0.95], dtype=np.float64)
SpeciesVec         = None #['H2','H','O','O2','OH','N','NH','NO','N2']
NPerT0             = 10000

# ## SECOND TIME
# DirName            = 'test'
# n_ics              = 10
# # T0Exts             = np.array([1000., 2000.], dtype=np.float64)
# # EqRatio0Exts       = np.array([.5, 4.], dtype=np.float64)
# T0Exts             = np.array([950., 1050.], dtype=np.float64)
# EqRatio0Exts       = np.array([0.95, 1.05], dtype=np.float64)
# X0Exts             = None
# SpeciesVec         = None
# NPerT0             = 5000




##########################################################################################

try:
    os.makedirs(OutputDir)
except:
    pass
# try:
#     os.makedirs(FigDir)
# except:
#     pass
try:
    os.makedirs(OutputDir+'/Orig/')
except:
    pass
try:
    os.makedirs(OutputDir+'/Orig/'+DirName+'/')
except:
    pass
try:
    os.makedirs(OutputDir+'/Orig/'+DirName+'/ext/')
except:
    pass




##########################################################################################
### Defining ODE and its Parameters
def IdealGasConstPressureReactor_SciPY(t, y):
    #print(t)

    Y          = y[1:]
    # YEnd     = np.array([1.-np.sum(y[1:])], dtype=np.float64)
    # Y        = np.concatenate((y[1:], YEnd), axis=0)
    gas_.TPY = y[0], P_, Y
    
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y, dtype=np.float64)
    ydot[0]  = - np.dot(wdot, gas_.partial_molar_enthalpies) / gas_.cp / gas_.density
    ydot[1:] = wdot * gas_.molecular_weights / gas_.density
    # ydot[1:] = wdot[0:-1] * gas_.molecular_weights[0:-1] / gas_.density
    
    return ydot


def IdealGasConstPressureReactor(t, T, Y):

    gas_.TP   = T, P_
    if (KeepVec):
        gas_sub = gas_[KeepVec]
    else:
        gas_sub = gas_
    gas_sub.Y = Y 

    
    wdot     = gas_sub.net_production_rates

    Tdot     = - np.dot(wdot, gas_sub.partial_molar_enthalpies) / gas_sub.cp / gas_sub.density
    Ydot     = wdot * gas_sub.molecular_weights / gas_sub.density

    HR       = - np.dot(gas_sub.net_production_rates,gas_sub.partial_molar_enthalpies)

    return Tdot*t, Ydot*t, HR


def IdealGasReactor_SciPY(t, y):
    #print(t)

    yPos     = np.maximum(y, 0.)
    ySum     = np.minimum(np.sum(y[1:]), 1.)
    YEnd     = np.array([1.-ySum], dtype=np.float64)
    Y        = np.concatenate((y[1:], YEnd), axis=0)
    gas_.TDY = y[0], density_, Y
    
    wdot     = gas_.net_production_rates

    ydot     = np.zeros_like(y, dtype=np.float64)
    ydot[0]  = - np.dot(wdot, gas_.partial_molar_int_energies) / gas_.cv / density_
    ydot[1:] = wdot[0:-1] * gas_.molecular_weights[0:-1] / density_
    
    return ydot


def IdealGasReactor(t, T, Y):

    gas_.TDY = T, density_, np.maximum(Y, 0.)
    
    wdot     = gas_.net_production_rates

    Tdot     = - np.dot(wdot, gas_.partial_molar_int_energies) / gas_.cv / density_
    Ydot     = wdot * gas_.molecular_weights / density_

    HR       = - np.dot(gas_.net_production_rates, gas_.partial_molar_enthalpies)
    
    return Tdot, Ydot, HR



##########################################################################################
### Generating Training Data


if (DirName == 'train'):

    if (EqRatio0Exts is not None):
        MinVals = np.array([EqRatio0Exts[0], T0Exts[0]], dtype=np.float64)
        MaxVals = np.array([EqRatio0Exts[1], T0Exts[1]], dtype=np.float64)
        NDims   = 2

        ICs     = pyDOE.lhs(2, samples=n_ics, criterion='center')

        for i in range(NDims):
            ICs[:,i] = ICs[:,i] * (MaxVals[i] - MinVals[i]) + MinVals[i]
        ICs = np.concatenate([P0*np.ones((n_ics,1)),ICs], axis=1)

        ### Writing Initial Temperatures
        FileName = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
        Header   = 'P,EqRatio,T'
        np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')


    elif (X0Exts is not None) and (SpeciesVec is not None):
        NSpecies = len(SpeciesVec)

        MinVals  = np.array([T0Exts[0], X0Exts[0]], dtype=np.float64)
        MaxVals  = np.array([T0Exts[1], X0Exts[1]], dtype=np.float64)
        NDims    = NSpecies + 1

        ICs      = pyDOE.lhs(NDims, samples=n_ics, criterion='center')

        ICs[:,0] = ICs[:,0] * (T0Exts[1] - T0Exts[0]) + T0Exts[0]
        for i in range(1, NDims):
            ICs[:,i] = ICs[:,i] * (X0Exts[1] - X0Exts[0]) + X0Exts[0]
        ICs = np.concatenate([P0*np.ones((n_ics,1)),ICs], axis=1)

        ### Writing Initial Temperatures
        FileName   = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
        SpeciesStr = SpeciesVec[0]
        for Spec in SpeciesVec[1:]:
            SpeciesStr += ','+Spec
        Header   = 'P,T,'+SpeciesStr
        np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')

    else:
        print('Please, specify (EqRatio0Exts) OR (X0Exts and SpeciesVec)!')



elif (DirName == 'test'):
    # NDims    = 2
    # ICs      = np.zeros((n_ics,NDims))
    # # ICs[:,0] = [2.5, 1.9, 3.5, 1., 3.6]
    # # ICs[:,1] = [1200., 1900., 1300., 1600., 1700.]
    # ICs[:,0] = [0.0.8, 0.9, 1.0, 1.1, 1.2]
    # ICs[:,1] = [1300., 1200., 1400., 1500., 1250.]
    # ICs = np.concatenate([P0*np.ones((n_ics,1)), ICs], axis=1)
    MinVals = np.array([EqRatio0Exts[0], T0Exts[0]], dtype=np.float64)
    MaxVals = np.array([EqRatio0Exts[1], T0Exts[1]], dtype=np.float64)
    NDims   = 2

    ICs     = pyDOE.lhs(2, samples=n_ics, criterion='center')

    for i in range(NDims):
        ICs[:,i] = ICs[:,i] * (MaxVals[i] - MinVals[i]) + MinVals[i]
    ICs = np.concatenate([P0*np.ones((n_ics,1)),ICs], axis=1)

    ### Writing Initial Temperatures
    FileName = OutputDir+'/Orig/'+DirName+'/ext/ICs.csv'
    Header   = 'P,EqRatio,T'
    np.savetxt(FileName, ICs, delimiter=',', header=Header, comments='')



### Iterating Over Residence Times
DataMat         = None
iStart          = np.zeros(n_ics)
iEnd            = np.zeros(n_ics)
AutoIgnitionVec = np.zeros((n_ics,1))
for iIC in range(n_ics):
    
    if (EqRatio0Exts is not None):
        P0       = ICs[iIC,0]
        EqRatio0 = ICs[iIC,1]
        T0       = ICs[iIC,2]
        print('Pressure = ', P0, 'Pa; EqRatio0 = ', EqRatio0, '; Temperature = ', T0, 'K')

    elif (X0Exts is not None) and (SpeciesVec is not None):
        P0       = ICs[iIC,0]
        T0       = ICs[iIC,1]
        print('Pressure = ', P0, 'Pa; Temperature = ', T0, 'K')
    

    ### Create Mixture
    gas     = ct.Solution(MixtureFile)

    Mask_ = []
    if (KeepVec):
        for Keep in KeepVec:
            Mask_.append(gas.species_names.index(Keep))
        Mask_ = np.array(Mask_)
    else:
        Mask_ = np.arange(len(gas.species_names))

    ### Create Reactor
    gas.TP  = T0, P0

    if (EqRatio0Exts is not None):
        gas.set_equivalence_ratio(EqRatio0, Fuel0, Oxydizer0)

    elif (X0Exts is not None) and (SpeciesVec is not None):
        SpecDict = {}
        for iS, Spec in enumerate(SpeciesVec):
            SpecDict[Spec] = ICs[iIC,iS+2]
        gas.X    = SpecDict
        print('   Mole Fractions = ', SpecDict)

    r       = ct.IdealGasConstPressureReactor(gas)
    sim     = ct.ReactorNet([r])
    sim.verbose = False

    gas_    = gas
    mass_   = r.mass
    # print('   Mass = ', mass_)
    density_= r.density
    P_      = P0
    y0      = np.array(np.hstack((gas_.T, gas_.Y)), dtype=np.float64)
    # y0      = np.array(np.hstack((gas_.T, gas_.Y[0:-1])), dtype=np.float64)

    ############################################################################
    # ### Initialize Integration 
    # tAuto    = 10**( (8.75058755*(1000/T0) -9.16120796) )
    # tMin     = tAuto * 1.e-1
    # tMax     = tAuto * 1.e1
    # dt0      = tAuto * 1.e-3

    # tStratch = 1.01
    # tVec     = [0.0]
    # t        = tMin
    # dt       = dt0
    # while (t <= tMax):
    #     tVec.append(t)
    #     t  =   t + dt
    #     dt = dt0 * tStratch
    ############################################################################

    ############################################################################
    # A
    #tVec     = np.logspace(np.log10(t0), np.log10(tEnd), NtInt)
    
    # B
    #tVec     = np.linspace(t0, tEnd, NtInt)
    
    # C
    # NtInt    = int(tEnd/Deltat)
    # t0       = np.random.rand() * Deltat
    # tVec     = np.append([0.], np.arange(NtInt-1) * Deltat + t0)

    # D
    tVec = []
    t    = 0.
    while (t<=tEnd):
        tVec.append(t)
        t += np.random.rand() * (DeltatMax-DeltatMin)+DeltatMin
    #############################################################################


    gas_             = gas
    if (KeepVec):
        gas_kept     = gas[KeepVec]
    else:
        gas_kept     = gas
    states           = ct.SolutionArray(gas_kept, 1, extra={'t': [0.0]})

    
    if (Integration == 'Canteras'):
        #r.set_advance_limit('temperature', delta_T_max)
        TT               = r.T
        YY               = r.thermo.Y[Mask_]
        Vec              = np.concatenate(([TT],YY), axis=0)
        TTdot, YYdot, HR = IdealGasConstPressureReactor(tVec[0], TT, YY)
        Vecdot           = np.concatenate(([TTdot],YYdot), axis=0)
        Mat              = np.array(Vec[np.newaxis,...])
        Source           = np.array(Vecdot[np.newaxis,...])
        it0              = 1
        tVecFinal        = np.array(tVec, dtype=np.float64)
        HRVec            = [HR]
    else:
        output           = solve_ivp( IdealGasConstPressureReactor_SciPY, (tVec[0],tVec[-1]), y0, method=SOLVER, t_eval=tVec, rtol=rtol, atol=atol )
        it0              = 0
        tVecFinal        = output.t
        HRVec            = []

    ### Integrate
    it           = it0
    for t in tVecFinal[it:]:

        if (Integration == 'Canteras'):
            sim.advance(t)
            TT               = r.T
            YY               = r.thermo.Y[Mask_]
        else:
            TT               = output.y[0,it]
            YY               = output.y[1:,it]
            # YY               = np.concatenate((output.y[1:,it], [1.0-np.sum(output.y[1:,it])]), axis=0)

        Vec                  = np.concatenate(([TT],YY), axis=0)

        TTdot, YYdot, HR     = IdealGasConstPressureReactor(t, TT, YY)
        Vecdot               = np.concatenate(([TTdot],YYdot), axis=0)

        if (it == 0):
            Mat              = np.array(Vec[np.newaxis,...])
            Source           = np.array(Vecdot[np.newaxis,...])
        else:
            Mat              = np.concatenate((Mat, Vec[np.newaxis,...]),       axis=0)
            Source           = np.concatenate((Source, Vecdot[np.newaxis,...]), axis=0)

        HRVec.append(HR)
        it+=1 
        
    #AutoIgnitionVec[iIC,0]   = tVecFinal[HRVec.index(max(HRVec))+it0]   
    ### print('Auto Ignition Delay = ', auto_ignition)


    ### Storing Results
    Nt  = len(tVecFinal)
    if (Nt < NPerT0):
        Mask = np.arange(Nt)
        Ntt  = Nt
    else:
        Mask = np.linspace(0,Nt-1,NPerT0, dtype=int)
        Ntt  = NPerT0

    if (iIC == 0):
        T0All        = np.ones(Ntt)*T0
        yTemp        = np.concatenate((tVecFinal[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = yTemp

        ySourceTemp  = np.concatenate((tVecFinal[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = ySourceTemp
        
        iStart[iIC]  = 0
        iEnd[iIC]    = Ntt
    else:
        T0All        = np.concatenate((T0All, np.ones(Ntt)*T0), axis=0)

        yTemp        = np.concatenate((tVecFinal[Mask,np.newaxis], Mat[Mask,:]), axis=1)
        yMat         = np.concatenate((yMat, yTemp), axis=0)
        
        ySourceTemp  = np.concatenate((tVecFinal[Mask,np.newaxis], Source[Mask,:]), axis=1)
        SourceMat    = np.concatenate((SourceMat, ySourceTemp), axis=0) 
        
        iStart[iIC]  = iEnd[iIC-1]
        iEnd[iIC]    = iEnd[iIC-1]+Ntt
        

    ### Writing Results
    NSpec        = gas.n_species
    Header       = 't,T'
    SpeciesNames = []
    for iSpec in range(NSpec):
        Header += ','+gas.species_name(iSpec)
        SpeciesNames.append(gas.species_name(iSpec))

    # FileName = OutputDir+'/orig_data/States.csv.'+str(iIC+1)
    # np.savetxt(FileName, DataTemp,       delimiter=',', header=Header, comments='')


    ### Writing Results
    Header   = 't,T'
    if (KeepVec):
        for Keep in KeepVec:
            Header += ','+Keep
    else:
        for iSpec in range(NSpec):
            Header += ','+gas.species_name(iSpec)

    FileName = OutputDir+'/Orig/'+DirName+'/ext/y.csv.'+str(iIC+1)
    np.savetxt(FileName, yTemp,       delimiter=',', header=Header, comments='')

    FileName = OutputDir+'/Orig/'+DirName+'/ext/ySource.csv.'+str(iIC+1)
    np.savetxt(FileName, ySourceTemp, delimiter=',', header=Header, comments='')

    # FileName = OutputDir+'/orig_data/Jacobian.csv.'+str(iIC+1)
    # np.savetxt(FileName, JJTauMat,    delimiter=',')



FileName = OutputDir+'/Orig/'+DirName+'/ext/SimIdxs.csv'
Header   = 'iStart,iEnd'
np.savetxt(FileName, np.concatenate((iStart[...,np.newaxis], iEnd[...,np.newaxis]), axis=1), delimiter=',', header=Header, comments='')

FileName = OutputDir+'/Orig/'+DirName+'/ext/tAutoIgnition.csv'
Header   = 't'
np.savetxt(FileName, AutoIgnitionVec, delimiter=',', header=Header, comments='')



print('Original (', len(SpeciesNames), ') Species: ', SpeciesNames)
VarsName    = ['T']
if (DirName == 'train'):
    
    for iSpec in range(yMat.shape[1]-2):
        if (np.amax(np.abs(yMat[1:,iSpec+2] - yMat[:-1,iSpec+2])) > 1.e-10):
            VarsName.append(SpeciesNames[iSpec]) 

    print('Non-zeros (', len(VarsName), ') Variables: ', VarsName)
 

    ToOrig       = []
    OrigVarNames = ['T']+SpeciesNames
    for Var in VarsName:
        ToOrig.append(OrigVarNames.index(Var))
    ToOrig = np.array(ToOrig, dtype=int)

    FileName = OutputDir+'/Orig/ToOrig_Mask.csv'
    np.savetxt(FileName, ToOrig, delimiter=',')


    FileName = OutputDir+'/Orig/'+DirName+'/ext/CleanVars.csv'
    StrSep = ','
    with open(FileName, 'w') as the_file:
        the_file.write(StrSep.join(VarsName)+'\n')

# ##########################################################################################