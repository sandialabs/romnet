import itertools
import numpy                       as np
import tensorflow                  as tf
import pandas                      as pd
from tensorflow.python.keras   import backend as K
import tensorflow_probability      as tfp

from .nn                       import NN
from .building_blocks          import System_of_Components



#===================================================================================================================================
class VI_DeepONet(NN):
    """Deep Operator Network 
    """

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, norm_input, stat_output, system):
        super(VI_DeepONet, self).__init__()

        self.structure_name            = 'VI_DeepONet'
        self.structure                 = InputData.structure
        self.n_hypers                  = len(list(self.structure.keys()))

        self.attention_mask            = None
        self.residual                  = None
          
          
        self.input_vars                = InputData.input_vars_all
        self.n_inputs                  = len(self.input_vars)
                     
            
        self.output_vars               = InputData.output_vars
        self.n_outputs                 = len(self.output_vars)
          
        self.branch_vars               = InputData.input_vars['DeepONetMean']['Branch']    
        self.trunk_vars                = InputData.input_vars['DeepONetMean']['Trunk']  

        self.n_branches                = len([name for name in self.structure['DeepONetMean'].keys() if 'Branch' in name])
        self.n_trunks                  = len([name for name in self.structure['DeepONetMean'].keys() if 'Trunk'  in name])

        try:
            self.n_rigids              = len([name for name in self.structure['DeepONetMean'].keys() if 'Rigid' in name]) 
            self.rigid_vars            = InputData.input_vars['DeepONetMean']['Rigid']
        except:
            self.n_rigids              = 0

        try:
            self.internal_pca_flg      = InputData.internal_pca_flg
        except:
            self.internal_pca_flg      = False

        try:
            self.sigma_like            = InputData.sigma_like
        except:
            self.sigma_like            = None


        if (norm_input is None):
            norm_input                 = pd.DataFrame(np.zeros((1,self.n_inputs)), columns=self.input_vars)
        self.norm_input                = norm_input

        self.trans_fun                 = InputData.trans_fun

        try:
            self.dotlayer_mult_flg     = InputData.dotlayer_mult_flg['DeepONetMean']
        except:
            self.dotlayer_mult_flg     = None
        try:
            self.dotlayer_bias_flg     = InputData.dotlayer_bias_flg['DeepONetMean']
        except:
            self.dotlayer_bias_flg     = None

        try:
            self.data_preproc_type     = InputData.data_preproc_type
        except:
            self.data_preproc_type     = None

        self.norm_output_flg           = InputData.norm_output_flg


        print("\n[ROMNet - vi_deeponet.py            ]:   Constructing Variational-Inference Deep Operator Network: ") 


        self.layers_dict      = {'All': {}}
        self.layer_names_dict = {'All': {}}

        self.system_of_components              = {}
        for system_name in list(self.structure.keys()):
            self.layers_dict[system_name]      = {}
            self.layer_names_dict[system_name] = {}


            for i_trunk in range(self.n_trunks):
                if (self.n_trunks > 1):
                    temp_str = '_'+str(i_trunk+1)
                else:
                    temp_str = ''
                self.layers_dict[system_name]['Trunk'+temp_str]      = {}
                self.layer_names_dict[system_name]['Trunk'+temp_str] = {}


            # PCA Layers
            if (self.internal_pca_flg):
                self.layers_dict[system_name]['PCALayer']    = PCALayer(system.A, system.C, system.D)
                self.layers_dict[system_name]['PCAInvLayer'] = PCAInvLayer(system.A, system.C, system.D)


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
                            layer_name = system_name+'-PreTransformation' + fun + '-' + str(i_trunk+1)
                            layer      = InputTransLayer(fun, len(self.trunk_vars), indxs, name=layer_name)

                            self.layers_dict[system_name]['Trunk'+temp_str]['TransFun']      = layer
                            self.layer_names_dict[system_name]['Trunk'+temp_str]['TransFun'] = layer_name


            # Trunk-Rigid Blocks Coupling Layers
            if (self.n_rigids > 0):
                for i_trunk in range(self.n_trunks):
                    if (self.n_trunks > 1):
                        temp_str = '_'+str(i_trunk+1)
                    else:
                        temp_str = ''
                    self.layers_dict[system_name]['Trunk'+temp_str]['Shift']      = tf.keras.layers.subtract
                    self.layers_dict[system_name]['Trunk'+temp_str]['Stretch']    = tf.keras.layers.multiply
                    self.layer_names_dict[system_name]['Trunk'+temp_str]['Shift'] = 'Subtract'
                    self.layer_names_dict[system_name]['Trunk'+temp_str]['Shift'] = 'Multiply'


            # Main System of Components
            self.system_of_components[system_name]         = System_of_Components(InputData, system_name, self.norm_input, layers_dict=self.layers_dict, layer_names_dict=self.layer_names_dict)


            # Adding Biases to the DeepONet's Dot-Layers
            if (self.dotlayer_mult_flg):
                self.layers_dict[system_name]['MultLayer'] = MultLayer()


            # Adding Biases to the DeepONet's Dot-Layers
            if (self.dotlayer_bias_flg):
                self.layers_dict[system_name]['BiasLayer'] = BiasLayer()


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


        hypers_vec = []
        for system_name in list(self.structure.keys()): 
            if (self.internal_pca_flg):
                inputs_branch = self.layers_dict[system_name]['PCALayer'](inputs_branch)

            hyper             = self.system_of_components[system_name].call([inputs_branch, inputs_trunk], self.layers_dict, training=training)
            hypers_vec.append(hyper)

        if (self.n_hypers == 1):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0] 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                # dist = tfp.distributions.MultivariateNormalDiag(loc=tf.ones_like(mu), scale_diag=self.sigma_like)
                # bij  = tfp.bijectors.ScaleMatvecDiag(scale_diag=mu)
                # dist = tfp.distributions.TransformedDistribution(distribution=dist, bijector=bij)
                return dist
       
        elif (self.n_hypers == 2):

            def normal_sp(hypers_vec): 
                mu         = hypers_vec[0] 
                sigma_like = hypers_vec[1]
                dist       = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 
                # dist       = tfp.distributions.MultivariateNormalDiag(loc=tf.ones_like(mu), scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 
                # bij        = tfp.bijectors.ScaleMatvecDiag(scale_diag=mu)
                # dist       = tfp.distributions.TransformedDistribution(distribution=dist, bijector=bij)
                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers_vec) 
        if (self.internal_pca_flg) and (self.norm_output_flg) and (self.stat_output):
            y                       = self.layers_dict['All']['OutputTrans'](y)

        return y

    # ---------------------------------------------------------------------------------------------------------------------------


    # ---------------------------------------------------------------------------------------------------------------------------
    def call_predict(self, inputs):

        inputs_branch, inputs_trunk = tf.split(inputs, num_or_size_splits=[len(self.branch_vars), len(self.trunk_vars)], axis=1)


        hypers_vec = []
        for system_name in list(self.structure.keys()): 
            if (self.internal_pca_flg):
                inputs_branch = self.layers_dict[system_name]['PCALayer'](inputs_branch)
            hyper             = self.system_of_components[system_name].call([inputs_branch, inputs_trunk], self.layers_dict, training=False)
            hypers_vec.append(hyper)

        if (self.n_hypers == 1):

            def normal_sp(hypers_vec): 
                mu   = hypers_vec[0]
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                return dist
        
        elif (self.n_hypers == 2):

            def normal_sp(hypers_vec): 
                mu         = hypers_vec[0] 
                sigma_like = hypers_vec[1] 
                #dist       = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 
                dist       = mu * tfp.distributions.MultivariateNormalDiag(loc=tf.ones_like(sigma_like), scale_diag=1e-5 + tf.math.softplus(0.05 * sigma_like)) 

                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers_vec) 

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