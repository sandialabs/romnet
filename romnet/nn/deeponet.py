import numpy                   as np
import tensorflow              as tf
import pandas                  as pd
import itertools
from tensorflow.python.keras       import backend as K

from .nn                   import NN
from .building_blocks      import System_of_Components



#===================================================================================================================================
class DeepONet(NN):
    """Deep Operator Network 
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_output, system):
        super(DeepONet, self).__init__()

        self.structure_name            = 'DeepONet'
           
        self.attention_mask            = None
        self.residual                  = None
          
          
        self.input_vars                = InputData.input_vars_all
        self.n_inputs                  = len(self.input_vars)
                     
            
        self.output_vars               = InputData.output_vars
        self.n_outputs                 = len(self.output_vars)
          
        self.branch_vars               = InputData.input_vars['DeepONet']['Branch']
        self.trunk_vars                = InputData.input_vars['DeepONet']['Trunk']  

        self.n_branches                = len([name for name in InputData.structure['DeepONet'].keys() if 'Branch' in name])
        self.n_trunks                  = len([name for name in InputData.structure['DeepONet'].keys() if 'Trunk'  in name])

        try:
            self.n_rigids              = len([name for name in InputData.structure['DeepONet'].keys() if 'Rigid' in name]) 
            self.rigid_vars            = InputData.input_vars['DeepONet']['Rigid']
        except:
            self.n_rigids              = 0

        try:
            self.internal_pca_flg      = InputData.internal_pca_flg
        except:
            self.internal_pca_flg      = False

        if (norm_input is None):
            norm_input                 = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input                = norm_input

        self.trans_fun                 = InputData.trans_fun

        try:
            self.dotlayer_mult_flg     = InputData.dotlayer_mult_flg['DeepONet']
        except:
            self.dotlayer_mult_flg     = None
        try:
            self.dotlayer_bias_flg     = InputData.dotlayer_bias_flg['DeepONet']
        except:
            self.dotlayer_bias_flg     = None

        try:
            self.data_preproc_type     = InputData.data_preproc_type
        except:
            self.data_preproc_type     = None


        self.norm_output_flg           = InputData.norm_output_flg

        print("\n[ROMNet - deeponet.py               ]:   Constructing Deep Operator Network: ") 


        self.layers_dict      = {'DeepONet': {}, 'All': {}}
        self.layer_names_dict = {'DeepONet': {}, 'All': {}}
        for i_trunk in range(self.n_trunks):
            if (self.n_trunks > 1):
                temp_str = '_'+str(i_trunk+1)
            else:
                temp_str = ''
            self.layers_dict['DeepONet']['Trunk'+temp_str]       = {}
            self.layer_names_dict['DeepONet']['Trunk'+temp_str] = {}


        # PCA Layers
        if (self.internal_pca_flg):
            self.layers_dict['DeepONet']['PCALayer']    = PCALayer(system.A, system.C, system.D)
            self.layers_dict['DeepONet']['PCAInvLayer'] = PCAInvLayer(system.A, system.C, system.D)


        # Pre-Transforming Layer
        if (self.trans_fun):
            for i_trunk in range(self.n_trunks):
                if (self.n_trunks > 1):
                    temp_str = '_'+str(i_trunk+1)
                else:
                    temp_str = ''
                for ifun, fun in enumerate(self.trans_fun):
                    vars_list = self.trans_fun[fun]

                    indxs = []
                    for ivar, var in enumerate(self.trunk_vars):
                        if var in vars_list:
                            indxs.append(ivar)

                    if (len(indxs) > 0):
                        layer_name = 'PreTransformation' + fun + '-' + str(i_trunk+1)
                        layer      = InputTransLayer(fun, len(self.trunk_vars), indxs, name=layer_name)

                        self.layers_dict['DeepONet']['Trunk'+temp_str]['TransFun']      = layer
                        self.layer_names_dict['DeepONet']['Trunk'+temp_str]['TransFun'] = layer_name

        # Trunk-Rigid Blocks Coupling Layers
        if (self.n_rigids > 0):
            for i_trunk in range(self.n_trunks):
                if (self.n_trunks > 1):
                    temp_str = '_'+str(i_trunk+1)
                else:
                    temp_str = ''
                self.layers_dict['DeepONet']['Trunk'+temp_str]['Shift']      = tf.keras.layers.subtract
                self.layers_dict['DeepONet']['Trunk'+temp_str]['Stretch']    = tf.keras.layers.multiply
                self.layer_names_dict['DeepONet']['Trunk'+temp_str]['Shift'] = 'Subtract'
                self.layer_names_dict['DeepONet']['Trunk'+temp_str]['Shift'] = 'Multiply'


        # Main System of Components
        self.system_of_components                          = {}
        self.system_of_components['DeepONet']              = System_of_Components(InputData, 'DeepONet', self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


        # Adding Biases to the DeepONet's Dot-Layers
        if (self.dotlayer_mult_flg):
            self.layers_dict['DeepONet']['MultLayer']      = MultLayer()


            # Adding Biases to the DeepONet's Dot-Layers
        if (self.dotlayer_bias_flg):
            self.layers_dict['DeepONet']['BiasLayer']      = BiasLayer()


        # Output Normalizing Layer
        self.norm_output_flg             = InputData.norm_output_flg
        self.stat_output                 = stat_output
        if (self.norm_output_flg) and (self.stat_output):                    
            self.output_min                                = tf.constant(stat_output['min'],  dtype=tf.keras.backend.floatx())
            self.output_max                                = tf.constant(stat_output['max'],  dtype=tf.keras.backend.floatx())
            self.output_mean                               = tf.constant(stat_output['mean'], dtype=tf.keras.backend.floatx())
            self.output_std                                = tf.constant(stat_output['std'],  dtype=tf.keras.backend.floatx())
            
            self.layers_dict['All']['OutputTrans']         = OutputTransLayer(   self.data_preproc_type, self.output_min, self.output_max, self.output_mean, self.output_std)
            self.layer_names_dict['All']['OutputTrans']    = 'OutputTrans'

            self.layers_dict['All']['OutputInvTrans']      = OutputInvTransLayer(self.data_preproc_type, self.output_min, self.output_max, self.output_mean, self.output_std)
            self.layer_names_dict['All']['OutputInvTrans'] = 'OutputInvTrans'


    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, training=False):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)

        y                           = self.system_of_components['DeepONet'].call_deeponet([inputs_branch, inputs_trunk], self.layers_dict, training=training)
        if (self.internal_pca_flg) and (self.norm_output_flg) and (self.stat_output):
            y                       = self.layers_dict['All']['OutputTrans'](y)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def call_hybrid(self, inputs, training=False):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)

        y                           = self.system_of_components['DeepONet'].call_deeponet_hybrid([inputs_branch, inputs_trunk], self.layers_dict, training=training)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)
        if (self.internal_pca_flg):
            inputs_branch           = self.layers_dict['DeepONet']['PCALayer'](inputs_branch)

        y                           = self.system_of_components['DeepONet'].call([inputs_branch, inputs_trunk], self.layers_dict, training=False)

        if (not self.internal_pca_flg) and (self.norm_output_flg) and (self.stat_output):                    
            y                       = self.layers_dict['All']['OutputInvTrans'](y)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def get_graph(self):

        input_     = tf.keras.Input(shape=[self.n_inputs,])
        return tf.keras.Model(inputs=[input_], outputs=[self.call(input_)] )

    # ---------------------------------------------------------------------------------------------------------------------------

#===================================================================================================================================



#=======================================================================================================================================
class AllInputTransLayer(tf.keras.layers.Layer):

    def __init__(self, f, name='AllInputTransLayer'):
        super(AllInputTransLayer, self).__init__(name=name, trainable=False)
        self.f           = f

    def call(self, inputs):
        
        if (self.f == 'log10'):
            y = tf.experimental.numpy.log10(K.maximum(inputs, K.epsilon()))
        elif (self.f == 'log'):
            y = tf.math.log(K.maximum(inputs, K.epsilon()))
        
        return y
        
#=======================================================================================================================================


#=======================================================================================================================================
class InputTransLayer(tf.keras.layers.Layer):

    def __init__(self, f, NVars, indxs, name='InputTrans'):
        super(InputTransLayer, self).__init__(name=name, trainable=False)
        self.f           = f
        self.NVars       = NVars
        self.indxs       = indxs

    def call(self, inputs):

        inputs_unpack = tf.split(inputs, self.NVars, axis=1)
        
        if (self.f == 'log10'):
            for indx in self.indxs:
                inputs_unpack[indx] = tf.experimental.numpy.log10(inputs_unpack[indx])
        elif (self.f == 'log'):
            for indx in self.indxs:
                inputs_unpack[indx] = tf.math.log(inputs_unpack[indx])
        
        return tf.concat(inputs_unpack, axis=1)
        
#=======================================================================================================================================


#=======================================================================================================================================
class OutputTransLayer(tf.keras.layers.Layer):

    def __init__(self, data_preproc_type, output_min, output_max, output_mean, output_std, name='OutputTrans'):
        super(OutputTransLayer, self).__init__(name=name, trainable=False)
        self.data_preproc_type = type
        self.output_min        = output_min
        self.output_max        = output_max
        self.output_mean       = output_mean
        self.output_std        = output_std
        self.output_range      = self.output_max - self.output_min

        if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
            self.call = self.call_std
        elif (self.data_preproc_type == '0to1'):
            self.call = self.call_0to1
        elif (self.data_preproc_type == 'range'):
            self.call = self.call_range
        elif (self.data_preproc_type == '-1to1'):
            self.call = self.call_m1to1
        elif (self.data_preproc_type == 'pareto'):
            self.call = self.call_pareto

    def call_std(self, inputs):
        return (inputs -  self.output_mean) / self.output_std

    def call_0to1(self, inputs):
        return (inputs -  self.output_min) / self.output_range

    def call_range(self, inputs):
        return (inputs) / self.output_range

    def call_m1to1(self, inputs):
        return 2. * (inputs - self.output_min) / (self.output_range) - 1.

    def call_pareto(self, inputs):
        return (inputs -  self.output_mean) / np.sqrt(self.output_std)
        
#=======================================================================================================================================


#=======================================================================================================================================
class OutputInvTransLayer(tf.keras.layers.Layer):

    def __init__(self, data_preproc_type, output_min, output_max, output_mean, output_std, name='OutputInvTrans'):
        super(OutputInvTransLayer, self).__init__(name=name, trainable=False)
        self.data_preproc_type = type
        self.output_min        = output_min
        self.output_max        = output_max
        self.output_mean       = output_mean
        self.output_std        = output_std
        self.output_range      = self.output_max - self.output_min
        
        if (self.data_preproc_type == None) or (self.data_preproc_type == 'std') or (self.data_preproc_type == 'auto'):
            self.call = self.call_std
        elif (self.data_preproc_type == '0to1'):
            self.call = self.call_0to1
        elif (self.data_preproc_type == 'range'):
            self.call = self.call_range
        elif (self.data_preproc_type == '-1to1'):
            self.call = self.call_m1to1
        elif (self.data_preproc_type == 'pareto'):
            self.call = self.call_pareto

    def call_std(self, inputs):
        return inputs * self.output_std + self.output_mean

    def call_0to1(self, inputs):
        return inputs * self.output_range + self.output_min

    def call_range(self, inputs):
        return inputs * self.output_range

    def call_m1to1(self, inputs):
        return (inputs + 1.)/2. * self.output_range + self.output_min

    def call_pareto(self, inputs):
        return inputs * np.sqrt(self.output_std) + self.output_mean

#=======================================================================================================================================



#=======================================================================================================================================
class PCALayer(tf.keras.layers.Layer):

    def __init__(self, A, C, D, name='PCALayer'):
        super(PCALayer, self).__init__(name=name, trainable=False)
        self.AT = A.T
        self.C  = C
        self.D  = D

    def call(self, inputs):
        return tf.matmul( (inputs -  self.C) / self.D, self.AT ) 
        
#=======================================================================================================================================


#=======================================================================================================================================
class PCAInvLayer(tf.keras.layers.Layer):

    def __init__(self, A, C, D, name='PCAInvLayer'):
        super(PCAInvLayer, self).__init__(name=name, trainable=False)
        self.A  = A
        self.C  = C
        self.D  = D

    def call(self, inputs):
        return tf.matmul( inputs, self.A) * self.D +  self.C
        
#=======================================================================================================================================



#=======================================================================================================================================
class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x):
        return x + self.bias

#=======================================================================================================================================



#=======================================================================================================================================
class MultLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        super(MultLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.stretch = self.add_weight('stretch',
                                       shape=input_shape[1:],
                                       initializer='ones',
                                       trainable=True)
    def call(self, x):
        return x * self.stretch

#=======================================================================================================================================