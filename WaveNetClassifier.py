import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Reshape, Activation, Add, Multiply, Conv1D, AveragePooling1D
from tensorflow.keras.models import Model 
from tensorflow.keras import backend as K
from tensorflow.keras import metrics
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import History, ModelCheckpoint
from tensorflow_addons.metrics import FBetaScore


class WaveNetClassifier():

    def __init__(self, input_shape, output_shape, kernel_size = 2, dilation_depth = 9, n_filters = 40):
        """
          Parameters:
          input_shape: (tuple) tuple of input shape. (e.g. If input is raw waveform with 25000 time samples, (25000,) is the input_shape)
          output_shape: (tuple)tuple of output shape. (e.g. If we want classify the signal into 5 classes, (5,) is the output_shape)
          kernel_size: (integer) kernel size of convolution operations in residual blocks
          dilation_depth: (integer) type total depth of residual blocks
          n_filters: (integer) # of filters of convolution operations in residual blocks
        """     

        # save input info
        self.input_shape = input_shape
        self.output_shape = output_shape
        
        # save hyperparameters of WaveNet
        self.kernel_size = kernel_size
        self.dilation_depth = dilation_depth
        self.n_filters = n_filters
 
        self.model = self.construct_model()

    
    def residual_block(self, x, i):
        """
        Define residual block
        Parameters:
        x: input of residual_block
        i: index of residual_block
        """

        tanh_out = Conv1D(self.n_filters, 
                          self.kernel_size, 
                          dilation_rate = self.kernel_size**i, 
                          padding='causal', 
                          name='dilated_conv_%d_tanh' % (self.kernel_size ** i), 
                          activation='tanh'
                          )(x)

        sigm_out = Conv1D(self.n_filters, 
                          self.kernel_size, 
                          dilation_rate = self.kernel_size**i, 
                          padding='causal', 
                          name='dilated_conv_%d_sigm' % (self.kernel_size ** i), 
                          activation='sigmoid'
                          )(x)

        z = Multiply(name='gated_activation_%d' % (i))([tanh_out, sigm_out])
        skip = Conv1D(self.n_filters, 1, name='skip_%d'%(i))(z)
        res = Add(name='residual_block_%d' % (i))([skip, x])

        return res, skip
  
    def construct_model(self):    
        """
        Construct WaveNet classifier Network
        """

        # input layer
        x = Input(shape=self.input_shape, name='original_input')
        x_reshaped = Reshape(self.input_shape + (1,), name='reshaped_input')(x)

        skip_connections = []

        # generate feature map of each frequency
        out = Conv1D(self.n_filters, 2, dilation_rate=1, padding='causal', name='dilated_conv_1')(x_reshaped)

        for i in range(1, self.dilation_depth + 1):
            out, skip = self.residual_block(out,i)
            skip_connections.append(skip)

        # classification part
        out = Add(name='skip_connections')(skip_connections)
        out = Activation('relu')(out)
        out = Conv1D(self.n_filters, 80, strides = 1, padding='same', name='conv_5ms', activation = 'relu')(out)
        out = AveragePooling1D(80, padding='same', name='downsample_to_200Hz')(out)

        out = Conv1D(self.n_filters, 100, padding='same', activation='relu', name='conv_500ms')(out)
        out = Conv1D(self.output_shape[0], 100, padding='same', activation='relu', name='conv_500ms_target_shape')(out)
        out = AveragePooling1D(100, padding='same',name = 'downsample_to_2Hz')(out)

        out = Conv1D(self.output_shape[0], (int) (self.input_shape[0] / 8000), padding='same', name='final_conv')(out)
        out = AveragePooling1D((int) (self.input_shape[0] / 8000), name='final_pooling')(out)

        out = Reshape(self.output_shape)(out)
        out = Activation('softmax')(out)

        model = Model(x, out)  
        model.summary()

        return model
  
    def fit_wn(self, X, Y, validation_split = None,  validation_data = None, epochs = 100, batch_size = 32, beta=2., optimizer='adam', save=False, save_dir='./',sample_weight=None):
        """
        Fit WaveNet classifier
        Parameters:
        X, Y(tuple): Data and Label for training (numpy array)
        validation_split(float): portion of the validation data (0 ~ `)
        validation_data(tuple): numpy arrays for validation (x_val, y_val)
        epochs(int): number of epochs for trainig
        batch_size(int): size of batch for mini-batch training
        beta(float): beta value for F-beta score
        optimizer(string): name of optimizer
        save(bool): save training information or not
        save_dir(string): if save is True, directory for save training information
        sample_weight(tuple): training weight for each sample
        """
        
        # set default losses if not defined
        loss = 'categorical_crossentropy'
        metrics = ['accuracy',FBetaScore(num_classes=len(Y[0]),beta=beta,average=None)]
            
        # set callback functions
        if save:
            saved = save_dir + "wave_clas-{epoch:02d}.h5"
            hist = save_dir + 'wavenet_classifier_training_history.csv'
            if (validation_data is None and validation_split is None):
                checkpointer = ModelCheckpoint(filepath=saved, monitor='loss', verbose=1, save_best_only=False)
            else:
                checkpointer = ModelCheckpoint(filepath=saved, monitor='val_accuracy', verbose=1, save_best_only=False)
            history = History()
            callbacks = [history, checkpointer]
        else:
            callbacks = None
          
        # compile the model
        self.model.compile(optimizer, loss, metrics)

        # fit the model
        self.history = self.model.fit(X, Y, shuffle = True, batch_size=batch_size, epochs = epochs, validation_split = validation_split, 
                validation_data = validation_data, callbacks=callbacks, sample_weight=sample_weight)
        if save:
            df = pd.DataFrame.from_dict(history.history)
            df.to_csv(hist, encoding='utf-8', index=False)

        return self.history


    def predict(self, x):
        return self.model.predict(x)
