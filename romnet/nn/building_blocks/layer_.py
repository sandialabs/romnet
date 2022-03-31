import numpy                    as np
import tensorflow               as tf

from tensorflow.keras       import regularizers
from tensorflow.keras       import activations
from tensorflow.keras       import initializers

from ...training            import L1L2Regularizer


#=======================================================================================================================================
class Layer(object):
	

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, 
    			 InputData, 
    			 layer_type       = 'TF',
    			 i_layer          = 1, 
    			 n_layers         = 1, 
    			 layer_name       = '', 
    			 n_neurons        = 1, 
    			 act_fun          = 'linear', 
    			 use_bias         = True, 
    			 trainable_flg    = 'all', 
    			 transfered_model = None):

        ### Weights L1 and L2 Regularization Coefficients 
    	self.weight_decay_coeffs = InputData.weight_decay_coeffs

    	self.i_layer             = i_layer
    	self.n_layers            = n_layers
    	self.layer_name          = layer_name
    	self.n_neurons           = n_neurons
    	self.act_fun             = act_fun
    	self.use_bias            = use_bias
    	self.trainable_flg       = trainable_flg
    	self.transfered_model    = transfered_model
    	self.last_flg            = True if (i_layer >= n_layers-1) else False

    	if (layer_type == 'TF'):
    		self.build = self.build_TF
		elif (layer_type == 'TFP'):
			self.build = self.build_TFP


    # ---------------------------------------------------------------------------------------------------------------------------
	def build_TF(self):

        # Parameters Initialization
        ### Biases L1 and L2 Regularization Coefficients 
        KW1 = self.weight_decay_coeffs[0]
        KW2 = self.weight_decay_coeffs[1]
        if (not self.last_flg):
            if (len(self.weight_decay_coeffs) == 2):
                kb1 = self.weight_decay_coeffs[0]
                kb2 = self.weight_decay_coeffs[1]
            else:
                kb1 = self.weight_decay_coeffs[2]
                kb2 = self.weight_decay_coeffs[3]
        else:
            kb1 = 0.
            kb2 = 0.

        if (self.transfered_model is not None):
            W0    = self.transfered_model.get_layer(self.layer_name).kernel.numpy()
            b0    = self.transfered_model.get_layer(self.layer_name).bias.numpy()
            W_ini = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
            b_ini = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            W_reg = L1L2Regularizer(kW1, kW2, W0)
            b_reg = L1L2Regularizer(kb1, kb2, b0)
        else:
            W_ini = 'he_normal' if (self.act_fun == 'relu') else 'glorot_normal'
            b_ini = 'zeros'
            W_reg = regularizers.l1_l2(l1=kW1, l2=kW2)
            b_reg = regularizers.l1_l2(l1=kb1, l2=kb2)

        # Constructing Kera Layer
        layer = tf.keras.layers.Dense(units              = self.n_neurons,
                                      activation         = self.act_fun,
                                      use_bias           = self.use_bias,
                                      kernel_initializer = W_ini,
                                      bias_initializer   = b_ini,
                                      kernel_regularizer = W_reg,
                                      bias_regularizer   = b_reg,
                                      name               = self.layer_name)


        # Trainable Layer?
        if (self.trainable_flg.lower() == 'none'):
            layer.trainable = False
        elif (self.trainable_flg.lower() == 'only_last'):
            if (not self.last_flg):
                layer.trainable = False

        return layer



    # ---------------------------------------------------------------------------------------------------------------------------
	def build_TFP(self):

        # Parameters Initialization
        ### Biases L1 and L2 Regularization Coefficients 
        KW1 = self.weight_decay_coeffs[0]
        KW2 = self.weight_decay_coeffs[1]
        if (not self.last_flg):
            if (len(self.weight_decay_coeffs) == 2):
                kb1 = self.weight_decay_coeffs[0]
                kb2 = self.weight_decay_coeffs[1]
            else:
                kb1 = self.weight_decay_coeffs[2]
                kb2 = self.weight_decay_coeffs[3]
        else:
            kb1 = 0.
            kb2 = 0.

        if (self.transfered_model is not None):
            W0    = self.transfered_model.get_layer(self.layer_name).kernel.numpy()
            b0    = self.transfered_model.get_layer(self.layer_name).bias.numpy()
            W_ini = tf.keras.initializers.RandomNormal(mean=W0, stddev=1.e-10)
            b_ini = tf.keras.initializers.RandomNormal(mean=b0, stddev=1.e-10)
            W_reg = L1L2Regularizer(kW1, kW2, W0)
            b_reg = L1L2Regularizer(kb1, kb2, b0)
        else:
            W_ini = 'he_normal' if (self.act_fun == 'relu') else 'glorot_normal'
            b_ini = 'zeros'
            W_reg = regularizers.l1_l2(l1=kW1, l2=kW2)
            b_reg = regularizers.l1_l2(l1=kb1, l2=kb2)

        # Constructing Kera Layer
        layer = tf.keras.layers.Dense(units              = self.n_neurons,
                                      activation         = self.act_fun,
                                      use_bias           = self.use_bias,
                                      kernel_initializer = W_ini,
                                      bias_initializer   = b_ini,
                                      kernel_regularizer = W_reg,
                                      bias_regularizer   = b_reg,
                                      name               = self.layer_name)


        # Trainable Layer?
        if (self.trainable_flg.lower() == 'none'):
            layer.trainable = False
        elif (self.trainable_flg.lower() == 'only_last'):
            if (not self.last_flg):
                layer.trainable = False

        return layer
