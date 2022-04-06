import numpy                          as np
from scipy.integrate import solve_ivp as scipy_ode
import pandas                         as pd
import tensorflow                     as tf
import cantera                        as ct

from .system import System



class StiffKinetics(System):

    #===========================================================================
    def __init__(
        self,
        InputData
    ):  

        path_to_data_fld       = InputData.path_to_data_fld
        ROMNet_fld             = InputData.ROMNet_fld

        self.norm_output_flg   = InputData.norm_output_flg
        self.data_preproc_type = InputData.data_preproc_type

 
        self.order             = [1]
 
        self.ind_names         = ['t']
        self.other_names       = InputData.input_vars_all.copy()
        self.other_names.remove('t')
 
        self.ind_labels        = ['t [s]']
        self.other_labels      = ['s_{1_0}', 's_{2_0}', 's_{3_0}']

        self.get_variable_locations()

        # Initial/final time
        self.read_extremes(ROMNet_fld)
    
        # Get Parameters
        self.read_params(ROMNet_fld)

        self.f_call            = self.f
        self.n_dims            = 3

        self.n_batch           = InputData.batch_size 
 
        self.fROM_anti         = None

    #===========================================================================



    #===========================================================================
    def f(self, t, y, ICs):

        y_1, y_2, y_3 = tf.split(y, 3, axis=1)

        dy_dt_1       = - self.params[0]*y_1                         + self.params[2]*y_2*y_3
        dy_dt_2       =   self.params[0]*y_1 - self.params[1]*y_2**2 - self.params[2]*y_2*y_3
        dy_dt_3       =                        self.params[1]*y_2**2

        dy_dt         = tf.concat([dy_dt_1, dy_dt_2, dy_dt_3], axis=1)
   
        return dy_dt

    #===========================================================================



    #===========================================================================
    def jac(self, t, y, arg):
        return arg

    #===========================================================================



    #===========================================================================
    def solve(self, y_0, params=None, eval_jac=False, eval_f=False):
        '''Solving the ODE.'''
        return None

    #===========================================================================


    #===========================================================================
    def get_residual(self, ICs, t, grads):

        y, dy_dt = grads


        if (self.norm_output_flg):

            if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
                y = y * self.output_std + self.output_mean

            elif (self.data_preproc_type == '0to1'):
                y = y * self.output_range + self.output_min

            elif (self.data_preproc_type == 'range'):
                y = y * self.output_range

            elif (self.data_preproc_type == '-1to1'):
                y = (y + 1.)/2. * self.output_range + self.output_min

            elif (self.data_preproc_type == 'pareto'):
                y = y * np.sqrt(self.output_std) + self.output_mean


        dy_ct_dt = self.f_call(t, y, ICs)


        if (self.norm_output_flg):

            if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
                dy_ct_dt /= self.output_std

            elif (self.data_preproc_type == '0to1'):
                dy_ct_dt /= self.output_range

            elif (self.data_preproc_type == 'range'):
                dy_ct_dt /= self.output_range

            elif (self.data_preproc_type == '-1to1'):
                dy_ct_dt /= self.output_range*2.

            elif (self.data_preproc_type == 'pareto'):
                dy_ct_dt /= np.sqrt(self.output_std)


        return dy_dt - dy_ct_dt

    #===========================================================================



    #===========================================================================
    def read_extremes(self, ROMNet_fld):

        # PathToExtremes      = ROMNet_fld + '/database/MassSpringDamper/Extremes/'

        # Data                = pd.read_csv(PathToExtremes+'/t.csv')
        # self.t0             = Data.to_numpy()[0,:]
        # self.tEnd           = Data.to_numpy()[1,:]

        self.ind_extremes   = None #{'t': [self.t0, self.tEnd]}


        # Data                = pd.read_csv(PathToExtremes+'/xv.csv')
        # self.xvMin          = Data.to_numpy()[0,:]
        # self.xvMax          = Data.to_numpy()[1,:]

        self.other_extremes = None #{'x': [self.xvMin[0], self.xvMax[0]], 'v': [self.xvMin[1], self.xvMax[1]]}

        # self.from_extremes_to_ranges()

        self.other_ranges = None

    #===========================================================================



    #===========================================================================
    def read_params(self, ROMNet_fld):

        PathToParams = ROMNet_fld + '/database/StiffKinetics/Params/'

        k1 = pd.read_csv(PathToParams+'/k1.csv').to_numpy()[0,0]
        k2 = pd.read_csv(PathToParams+'/k2.csv').to_numpy()[0,0]
        k3 = pd.read_csv(PathToParams+'/k3.csv').to_numpy()[0,0]

        self.params = [k1, k2, k3]

    #===========================================================================


    # #===========================================================================
    # def preprocess_data(self, all_data, xstat):

    #     for i, now_data in enumerate(all_data):
    #         for data_id, xyi_data in now_data.items():

    #             # all_data[i][data_id][0]['HH'] = (all_data[i][data_id][0]['HH'] - xstat['min'].to_numpy()[0]) / (xstat['max'].to_numpy()[0] - xstat['min'].to_numpy()[0])
    #             # all_data[i][data_id][1]['HH'] = (all_data[i][data_id][1]['HH'] - xstat['min'].to_numpy()[0]) / (xstat['max'].to_numpy()[0] - xstat['min'].to_numpy()[0])

    #             # for var in list(all_data[i][data_id][0].columns)[1:]:
    #             #     all_data[i][data_id][0][var] = np.log10(all_data[i][data_id][0][var])          
    #             #     all_data[i][data_id][1][var] = np.log10(all_data[i][data_id][1][var])   
                
    #             all_data[i][data_id][0] = (all_data[i][data_id][0] - xstat['mean'].to_numpy()) / np.sqrt(xstat['std'].to_numpy())
    #             all_data[i][data_id][1] = (all_data[i][data_id][1] - xstat['mean'].to_numpy()) / np.sqrt(xstat['std'].to_numpy())

    #     return all_data       

    # #===========================================================================



# #=======================================================================================================================================
# class AutoEncoderLayer(tf.keras.layers.Layer):

#     def __init__(self, path_to_data_fld, NVars, trainable_flg=False, name='AutoEncoderLayer'):
#         super(AutoEncoderLayer, self).__init__(name=name, trainable=False)

#         self.path_to_data_fld = path_to_data_fld
#         Data               = pd.read_csv(self.path_to_data_fld+'/train/ext/Output_MinMax.csv')
#         var_min            = Data.to_numpy()[:,0]
#         var_max            = Data.to_numpy()[:,1]

#         self.HH_min        = var_min[0]
#         self.HH_max        = var_max[0]
#         self.HH_range      = self.HH_max - self.HH_min
#         self.NVars         = NVars
#         self.trainable_flg = trainable_flg

#     def call(self, inputs):

#         inputs_unpack    = tf.split(inputs, [1,self.NVars-1], axis=1)

#         inputs_unpack[0] = (inputs_unpack[0] - self.HH_min) / self.HH_range

#         #inputs_unpack[1] = tf.experimental.numpy.log10(inputs_unpack[1])
#         inputs_unpack[1] = tf.math.log(inputs_unpack[1])
        
#         return tf.concat(inputs_unpack, axis=1)

# #=======================================================================================================================================



# #=======================================================================================================================================
# class AntiAutoEncoderLayer(tf.keras.layers.Layer):

#     def __init__(self, path_to_data_fld, NVars, trainable_flg=False, name='AntiAutoEncoderLayer'):
#         super(AntiAutoEncoderLayer, self).__init__(name=name, trainable=False)

#         self.path_to_data_fld = path_to_data_fld
#         Data               = pd.read_csv(self.path_to_data_fld+'/train/ext/Output_MinMax.csv')
#         var_min            = Data.to_numpy()[:,0]
#         var_max            = Data.to_numpy()[:,1]

#         self.HH_min        = var_min[0]
#         self.HH_max        = var_max[0]
#         self.HH_range      = self.HH_max - self.HH_min
#         self.NVars         = NVars
#         self.trainable_flg = trainable_flg

#     def call(self, inputs):

#         inputs_unpack    = tf.split(inputs, [1,self.NVars-1], axis=1)

#         inputs_unpack[0] = inputs_unpack[0] * self.HH_range + self.HH_min

#         #inputs_unpack[1] = 10**(inputs_unpack[1])
#         inputs_unpack[1] = tf.math.exp(inputs_unpack[1])
        
#         return tf.concat(inputs_unpack, axis=1)

# #=======================================================================================================================================