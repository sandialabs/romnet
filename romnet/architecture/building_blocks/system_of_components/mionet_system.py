import numpy                                  as np
import tensorflow                             as tf
import h5py

from tensorflow.keras                     import regularizers
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.python.keras              import backend
from tensorflow.python.keras.engine       import base_layer_utils
from tensorflow.python.keras.utils        import tf_utils
from tensorflow.python.util.tf_export     import keras_export

from ..component                          import Component
from ..normalization                      import CustomNormalization

from .                                    import System_of_Components



#=======================================================================================================================================
class MIONet_System(System_of_Components):

    # ---------------------------------------------------------------------------------------------------------------------------
    def __init__(self, InputData, name, norm_input, stat_input, layers_dict=[], layer_names_dict=[]):
        super(MIONet_System, self).__init__()
     
        self.name                          = name

        if ('_' in self.name):
            self.type, self.idx            = self.name.split('_')
            self.idx                       = int(self.idx)
        else:           
            self.idx                       = 1
            self.type                      = self.name 
                  
        self.structure                     = InputData.structure

        self.input_vars                    = []
        for input_vars_ in InputData.input_vars[self.name].values():
            self.input_vars               += input_vars_
        self.input_vars                    = list(set(self.input_vars))
        self.n_inputs                      = len(self.input_vars)
        
        if (norm_input is not None):
            self.norm_input                = norm_input
        else:
            self.norm_input                = None
        if (stat_input is not None):
            self.stat_input                = stat_input
        else:
            self.stat_input                = None

        try:
            self.data_preproc_type         = InputData.data_preproc_type
        except:
            self.data_preproc_type         = 'std'

        self.output_vars                   = InputData.output_vars
        self.n_outputs                     = len(self.output_vars)

        try:
            self.internal_pca_flg          = InputData.internal_pca_flg
        except:
            self.internal_pca_flg          = False


        self.n_branches                = len([name for name in self.structure[self.name].keys() if 'branch1' in name.lower()])
        # self.branch1_vars              = InputData.input_vars[self.name]['Branch1']
        # self.branch2_vars              = InputData.input_vars[self.name]['Branch2']

        self.n_pre_blocks              = 0
        try:
            self.n_shifts              = len([name for name in self.structure[self.name].keys() if 'shift' in name.lower()]) 
            self.n_pre_blocks         += 1 
            # self.shift1_vars           = InputData.input_vars[self.name]['Shift1']
            # self.shift2_vars           = InputData.input_vars[self.name]['Shift2']
        except:
            self.n_shifts              = 0
        try:
            self.n_stretches           = len([name for name in self.structure[self.name].keys() if 'stretch' in name.lower()]) 
            self.n_pre_blocks         += 1 
            # self.stretch1_vars         = InputData.input_vars[self.name]['Stretch1']
            # self.stretch2_vars         = InputData.input_vars[self.name]['Stretch2']
        except:
            self.n_stretches           = 0
        try:
            self.n_rotations           = len([name for name in self.structure[self.name].keys() if 'rotation' in name.lower()]) 
            self.n_pre_blocks         += 1 
            # self.rotation1_vars        = InputData.input_vars[self.name]['Rotation1']
            # self.rotation2_vars        = InputData.input_vars[self.name]['Rotation2']
        except:
            self.n_rotations           = 0
        try:
            self.n_prenets             = len([name for name in self.structure[self.name].keys() if 'prenet' in name.lower()]) 
            self.n_pre_blocks         += 1
            # self.prenet1_vars          = InputData.input_vars[self.name]['PreNet1']
            # self.prenet2_vars          = InputData.input_vars[self.name]['PreNet2']
        except:
            self.n_prenets             = 0

        self.n_trunks                  = len([name for name in self.structure[self.name].keys() if 'trunk' in name.lower()])
        # self.trunk_vars                = InputData.input_vars[self.name]['Trunk']

        try:
            self.branch_to_trunk       = InputData.branch_to_trunk[self.name]
            if (self.branch_to_trunk   == 'multi_to_one'):
                self.branch_to_trunk   = [0]*self.n_branches
            elif (self.branch_to_trunk == 'one_to_one'):
                self.branch_to_trunk   = np.arange(self.n_branches)
        except:    
          self.branch_to_trunk         = [0]*self.n_branches
        print("[ROMNet - system_of_components.py   ]:     Mapping Branch-to-Trunk (i.e., self.branch_to_trunk Object): ", self.branch_to_trunk) 

        try:
            self.transfered_model      = InputData.transfered_model
        except:
            self.transfered_model      = None
            
        try:
            self.n_branch_out          = InputData.n_branch_out
            self.n_trunk_out           = InputData.n_trunk_out
        except:
            self.n_branch_out          = None
            self.n_trunk_out           = None

        try:
            self.dotlayer_bias_flg     = InputData.dotlayer_bias_flg[self.name]
        except:
            self.dotlayer_bias_flg     = None
        try:
            self.dotlayer_mult_flg     = InputData.dotlayer_mult_flg[self.name]
        except:
            self.dotlayer_mult_flg     = None


        try:
            self.gaussnoise_rate           = InputData.gaussnoise_rate[self.name]
        except:
            self.gaussnoise_rate           = None

        try:
            self.softmax_flg               = InputData.softmax_flg[self.name]
        except:
            self.softmax_flg               = False
        if (self.softmax_flg):
            self.softmax_layer             = tf.keras.layers.Softmax(axis=1)

        try:
            self.rectify_flg               = InputData.rectify_flg
        except:
            self.rectify_flg               = False
        if (self.rectify_flg):
            self.rectify_layer             = tf.keras.activations.relu



        print("[ROMNet - system_of_components.py   ]:     Constructing System of Components: " + self.name) 


        # Normalizing Layer
        self.norm_layers_dict = []
        if (InputData.norm_input_flg[self.name]):

            for key, value in InputData.norm_input_flg[self.name].items():
                if (InputData.norm_input_flg[self.name][key]):

                    layer_name                              = self.name + '-' + key + '_Normalization'
                    layer                                   = CustomNormalization(name=layer_name, data_preproc_type=self.data_preproc_type)
                    norm_input                              = np.array(self.norm_input[InputData.input_vars[self.name][key]])
                    layer.adapt(norm_input)
                    if (not key in layers_dict[self.name]):
                        layers_dict[self.name][key]         = {}
                        layer_names_dict[self.name][key]    = {}
                    layers_dict[self.name][key][layer_name] = layer
                    layer_names_dict[self.name][key][layer_name] = layer_name

                    self.norm_layers_dict.append(key)


        # Gaussian Noise Layer
        self.noise_layers_dict = []
        if (self.gaussnoise_rate):

            for key, value in self.gaussnoise_rate.items():
                if (self.gaussnoise_rate[key]):

                    layer_name                              = self.name + '-' + key + '_GaussNoise'
                    layer                                   = tf.keras.layers.GaussianNoise(value)
                    if (not key in layers_dict[self.name]):
                        layers_dict[self.name][key]         = {}
                        ayer_names_dict[self.name][key]     = {}
                    layers_dict[self.name][key][layer_name] = layer
                    layer_names_dict[self.name][key][layer_name] = layer_name

                    self.noise_layers_dict.append(key)


        # Iterating over Components
        self.components     = {}
        self.branch_names   = []
        self.branch1_names  = []
        self.branch2_names  = []
        self.shift_names    = []
        self.stretch_names  = []
        self.rotation_names = []
        self.prenet_names   = []
        self.trunk_names    = []
        for component_name in self.structure[self.name]:

            if  ('branch' in component_name.lower()):
                self.branch_names.append(component_name)
                component_type = 'Branch'
            if  ('branch1' in component_name.lower()):
                self.branch1_names.append(component_name)
                component_type = 'Branch1'
            if  ('branch2' in component_name.lower()):
                self.branch2_names.append(component_name)
                component_type = 'Branch2'
            elif ('shift' in component_name.lower()):
                self.shift_names.append(component_name)
                component_type = 'Shift'
            elif ('stretch' in component_name.lower()):
                self.stretch_names.append(component_name)
                component_type = 'Stretch'
            elif ('rotation' in component_name.lower()):
                self.rotation_names.append(component_name)
                component_type = 'Rotation'
            elif ('prenet' in component_name.lower()):
                self.prenet_names.append(component_name)
                component_type = 'PreNet'
            elif ('trunk' in component_name.lower()):
                self.trunk_names.append(component_name)
                component_type = 'Trunk'
            elif ('FNN' in component_name.lower()):
                component_type = 'FNN'

            if (not component_name in layers_dict[self.name]):
                layers_dict[self.name][component_name]      = {}
                layer_names_dict[self.name][component_name] = {}

            self.components[component_name]            = Component(InputData, self.name, component_name, self.norm_input, layers_dict=layers_dict, layer_names_dict=layer_names_dict)



    # ---------------------------------------------------------------------------------------------------------------------------



    # ---------------------------------------------------------------------------------------------------------------------------
    def call(self, inputs, layers_dict, training):

        inputs_branch1, inputs_branch2, inputs_trunk = inputs


        if ('Branch1' in self.norm_layers_dict):
            inputs_branch1 = layers_dict[self.name]['Branch1'][self.name+'-Branch1_Normalization'](inputs_branch1)
        if ('Branch2' in self.norm_layers_dict):
            inputs_branch2 = layers_dict[self.name]['Branch2'][self.name+'-Branch2_Normalization'](inputs_branch2)
        if ('Trunk'  in self.norm_layers_dict):
            inputs_trunk   = layers_dict[self.name]['Trunk'][self.name+'-Trunk_Normalization'](inputs_trunk)

        if ('Branch1' in self.noise_layers_dict):
            inputs_branch1 = layers_dict[self.name]['Branch1'][self.name+'-Branch1_GaussNoise'](inputs_branch1, training=training)
        if ('Branch2' in self.noise_layers_dict):
            inputs_branch2 = layers_dict[self.name]['Branch2'][self.name+'-Branch2_GaussNoise'](inputs_branch2, training=training)
        if ('Trunk' in self.noise_layers_dict):
            inputs_trunk   = layers_dict[self.name]['Trunk'][self.name+'-Trunk_GaussNoise'](inputs_trunk, training=training)


        # tf.keras.backend.print_tensor('inputs_branch1 = ', inputs_branch1)
        # tf.keras.backend.print_tensor('inputs_branch2 = ', inputs_branch2)
        # tf.keras.backend.print_tensor('inputs_trunk   = ', inputs_trunk)

        pre_input  = [inputs_branch1, inputs_branch2]
        y_pre_dict = {}

        # Checking if Any Shift-Net is Part of the flexDeepONet
        if (self.n_shifts > 0):
            for i_shift in range(self.n_shifts):
                shift_name     = self.shift_names[i_shift]
                y_shift        = self.components[shift_name].call(pre_input[i_shift], layers_dict, None, training=training)
                if (self.n_trunks > 1):
                    y_pre_dict[shift_name] = tf.split(y_shift, num_or_size_splits=[1]*self.n_trunks, axis=1)
                else:
                    y_pre_dict[shift_name] = [y_shift]

        # Checking if Any Stretch-Net is Part of the flexDeepONet
        if (self.n_stretches > 0):
            for i_stretch in range(self.n_stretches):
                stretch_name   = self.stretch_names[i_stretch]
                y_stretch      = self.components[stretch_name].call(pre_input[i_stretch], layers_dict, None, training=training)
                if (self.n_trunks > 1):
                    y_pre_dict[stretch_name] = tf.split(y_stretch, num_or_size_splits=[1]*self.n_trunks, axis=1)
                else:
                    y_pre_dict[stretch_name] = [y_stretch]

        # Checking if Any Rot-Net is Part of the flexDeepONet
        if (self.n_rotations > 0):
            for i_rotation in range(self.n_rotations):
                rotation_name  = self.rotation_names[i_rotation]
                y_rotation     = self.components[rotation_name].call(pre_input[i_rotation], layers_dict, None, training=training)
                if (self.n_trunks > 1):
                    y_pre_dict[rotation_name] = tf.split(y_rotation, num_or_size_splits=[1]*self.n_trunks, axis=1)
                else:
                    y_pre_dict[rotation_name] = [y_rotation]

        # Checking if Any Pre-Net Block is Part of the flexDeepONet
        if (self.n_prenets > 0):
            for i_prenet in range(self.n_prenets):
                prenet_name  = self.prenet_names[i_prenet]
                y_prenet     = self.components[prenet_name].call(pre_input[i_prenet], layers_dict, None, training=training)
                if (self.n_trunks > 1):
                    y_pre_dict[prenet_name] = tf.split(y_prenet, num_or_size_splits=[1]*self.n_trunks, axis=1)
                else:
                    y_pre_dict[prenet_name] = [y_prenet]


        # Create Array of Trunks
        y_trunk_vec = []
        for i_trunk in range(self.n_trunks): 
            trunk_name  = self.trunk_names[i_trunk]
            if (y_pre_dict):
                y_pre_dict_ = y_pre_dict.copy()
                for key in y_pre_dict:
                    y_pre_dict_[key] = y_pre_dict[key][i_trunk]
            else:
                y_pre_dict_ = None
            y_trunk_vec.append(self.components[trunk_name].call(inputs_trunk, layers_dict, y_pre_dict_, training=training))

    
        # Create Array of Branches
        y_branch_vec = []
        output_vec   = []
        for i_branch in range(self.n_branches): 
            i_trunk      = self.branch_to_trunk[i_branch]
            branch1_name = self.branch1_names[i_branch] 
            branch2_name = self.branch2_names[i_branch] 
            y1           = self.components[branch1_name].call(inputs_branch1, layers_dict, None, training=training)
            y2           = self.components[branch2_name].call(inputs_branch2, layers_dict, None, training=training)

            # Perform Dot Pructs Between Trunks and Branches
            #output_dot  = Dot_Add(axes=1)([y, y_trunk_vec[i_trunk]])     
            if (self.n_branch_out == None) or (self.n_branch_out == self.n_trunk_out):
                # Branch Output Layer does not contain either Centering nor Scaling
                output_mult         = tf.keras.layers.multiply([y1, y2, y_trunk_vec[i_trunk]])
                output_             = tf.math.reduce_sum(output_mult, 1, keepdims=True)
            elif (self.n_branch_out == self.n_trunk_out+2):
                # Branch Output Layer contains Centering and Scaling
                alpha_vec1, c1, d1  = tf.split(y1, num_or_size_splits=[self.n_trunk_out, 1, 1], axis=1)
                alpha_vec2, c2, d2  = tf.split(y2, num_or_size_splits=[self.n_trunk_out, 1, 1], axis=1)
                output_mult         = tf.keras.layers.multiply([alpha_vec1, alpha_vec2, y_trunk_vec[i_trunk]])
                output_dot          = tf.math.reduce_sum(output_mult, 1, keepdims=True)
                output_mult         = tf.keras.layers.multiply([output_dot,  d1, d2])            
                output_             = tf.keras.layers.add([output_mult, c1, c2])   
            elif (self.n_branch_out == self.n_trunk_out+1):
                # Branch Output Layer contains Centering
                alpha_vec1, c1      = tf.split(y1, num_or_size_splits=[self.n_trunk_out, 1], axis=1)
                alpha_vec2, c2      = tf.split(y2, num_or_size_splits=[self.n_trunk_out, 1], axis=1)
                output_mult         = tf.keras.layers.multiply([alpha_vec1, alpha_vec2, y_trunk_vec[i_trunk]])
                output_dot          = tf.math.reduce_sum(output_mult, 1, keepdims=True)
                output_             = tf.keras.layers.add([output_dot, c1, c2])   
            else:
                # Branch Output Layer incompatible with Trunk Output Layer
                raise NameError("[ROMNet - call_mionet.py ]: Branch Output Layer incompatible with Trunk Output Layer! Please, Change No of Neurons!")


            output_vec.append( output_ )
            

        # Concatenate the Outputs of Multiple Dot-Layers
        if (self.n_branches > 1):
            output_concat            = tf.keras.layers.Concatenate(axis=1)(output_vec)
        else:
            output_concat            = output_vec[0]

        if (self.dotlayer_mult_flg):
            # Add Biases to Concatenated Dot-Layers
            output_concat            = layers_dict[self.name]['MultLayer'](output_concat)
        
        if (self.dotlayer_bias_flg):
            # Add Biases to Concatenated Dot-Layers
            output_concat            = layers_dict[self.name]['BiasLayer'](output_concat)


        if (self.internal_pca_flg):
            # Anti-Transform from PCA Space to Original State Space
            output_concat            = layers_dict[self.name]['PCAInvLayer'](output_concat)


        if (self.softmax_flg is True):
            # Apply SoftMax for Forcing Sum(y_i)=1

            output_T, output_concat  = tf.split(output_concat, [1,self.n_outputs-1], axis=1)
            output_concat            = self.softmax_layer(output_concat)
            output_concat            = tf.keras.layers.Concatenate(axis=1)([output_T, output_concat])


        if (self.rectify_flg is True):
            # Apply ReLu for Forcing y_i>0
            
            output_concat            = self.rectify_layer(output_concat)


        return output_concat

    # ---------------------------------------------------------------------------------------------------------------------------

#=======================================================================================================================================
