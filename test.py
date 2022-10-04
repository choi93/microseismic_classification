import numpy as np
import tensorflow as tf
import os
from WaveNetClassifier import WaveNetClassifier

# GPU setting
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"
os.environ [ "CUDA_VISIBLE_DEVICES" ] = '0'

# load test data set
data_dir='./data'
x_test=np.load(data_dir+'/d_test.npy')
y_test=np.load(data_dir+'/l_test.npy')

# load the trained WaveNet model
wnc=tf.keras.models.load_model('./wave_clas-27.h5')

# make confusion matrix
test_predict=wnc.predict(x_test)
test_pred_int=test_predict.argmax(axis=1)
test_label=y_test.argmax(axis=1)

cm_test=tf.math.confusion_matrix(test_label,test_pred_int)
print(cm_test)
