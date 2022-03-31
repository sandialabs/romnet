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
            self.dotlayer_bias_flg     = InputData.dotlayer_bias_flg['DeepONetMean']
        except:
            self.dotlayer_bias_flg     = None


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
                            layer_name = 'PreTransformation' + fun + '-' + str(i_trunk+1)
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
            if (self.dotlayer_bias_flg):
                self.layers_dict[system_name]['BiasLayer'] = BiasLayer()


            # # Output Normalizing Layer
            # self.norm_output_flg             = InputData.norm_output_flg
            # self.stat_output                 = stat_output
            # if (self.norm_output_flg) and (self.stat_output):                    
            #     self.output_min                                = tf.constant(stat_output['min'],  dtype=tf.keras.backend.floatx())
            #     self.output_max                                = tf.constant(stat_output['max'],  dtype=tf.keras.backend.floatx())
            #     self.output_range                              = tf.constant(self.output_max - self.output_min,   dtype=tf.keras.backend.floatx())
                
            #     self.layers_dict['All']['OutputTrans']         = OutputTransLayer(   self.output_range, self.output_min)
            #     self.layer_names_dict['All']['OutputTrans']    = 'OutputTrans'

            #     self.layers_dict['All']['OutputInvTrans']      = OutputInvTransLayer(self.output_range, self.output_min)
            #     self.layer_names_dict['All']['OutputInvTrans'] = 'OutputInvTrans'


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
            # hypers = hypers_vec[0]

            def normal_sp(hypers_vec): 
                mu = hypers_vec[0] # params[:,0:self.n_outputs]

                #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                return dist
       
        elif (self.n_hypers == 2):
            #hypers = tf.keras.layers.Concatenate(axis=1)(hypers_vec)

            def normal_sp(hypers_vec): 
                mu = hypers_vec[0] # params[:,0:self.n_outputs]
                sd = hypers_vec[1] # params[:,self.n_outputs:]

                #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                #dist = tfp.distributions.MultivariateNormalDiag(loc=params[:,0:self.n_outputs], scale_diag=1e-8 + tf.math.softplus(0.05 * params[:,self.n_outputs:])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-8 + tf.math.softplus(0.05 * sd)) 

                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers_vec) 

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
                mu = hypers_vec[0] # params[:,0:self.n_outputs]

                #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=self.sigma_like)
                return dist
        
        elif (self.n_hypers == 2):
            #hypers = tf.keras.layers.Concatenate(axis=1)(hypers_vec)

            def normal_sp(hypers_vec): 
                mu = hypers_vec[0] # params[:,0:self.n_outputs]
                sd = hypers_vec[1] # params[:,self.n_outputs:]

                #dist = tfp.distributions.Normal(loc=params[:,0:1], scale=1e-3 + tf.math.softplus(0.05 * params[:,1:2])) 
                #dist = tfp.distributions.MultivariateNormalDiag(loc=params[:,0:self.n_outputs], scale_diag=1e-3 + tf.math.softplus(0.05 * params[:,self.n_outputs:])) 
                dist = tfp.distributions.MultivariateNormalDiag(loc=mu, scale_diag=1e-5 + tf.nn.relu(sd)) 
                return dist

        y = tfp.layers.DistributionLambda(normal_sp)(hypers_vec) 

        if (self.norm_output_flg) and (self.stat_output):                    
            y = self.layers_dict['All']['OutputInvTrans'](y)

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

    def __init__(self, output_range, output_min, name='OutputTrans'):
        super(OutputTransLayer, self).__init__(name=name, trainable=False)
        self.output_range = output_range
        self.output_min   = output_min

    def call(self, inputs):
        return (inputs -  self.output_min) / self.output_range
        
#=======================================================================================================================================


#=======================================================================================================================================
class OutputInvTransLayer(tf.keras.layers.Layer):

    def __init__(self, output_range, output_min, name='OutputInvTrans'):
        super(OutputInvTransLayer, self).__init__(name=name, trainable=False)
        self.output_range = output_range
        self.output_min   = output_min

    def call(self, inputs):
        return inputs * self.output_range + self.output_min
        
#=======================================================================================================================================



#=======================================================================================================================================
class PCALayer(tf.keras.layers.Layer):

    def __init__(self, A, C, D, name='PCALayer'):
        super(PCALayer, self).__init__(name=name, trainable=False)
        self.AT = A.T
        self.C  = C
        self.D  = D

        print('self.C = ', self.C)
        print('self.D = ', self.D)

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