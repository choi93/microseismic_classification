import numpy as np
import tensorflow as tf
import os
from WaveNetClassifier import WaveNetClassifier

# GPU setting
os.environ [ "TF_FORCE_GPU_ALLOW_GROWTH" ] = "true"
os.environ [ "CUDA_VISIBLE_DEVICES" ] = '0'

data_dir='./data'
x_train=np.load(data_dir+'/d_train.npy')
y_train=np.load(data_dir+'/l_train.npy')
x_val=np.load(data_dir+'/d_val.npy')
y_val=np.load(data_dir+'/l_val.npy')

# set sample weight
ns=y_train.sum(axis=0)
ndata=len(x_train)

weight=(1/ns)*(ndata)/5.
weight[4]=weight[4]*2

y_int=y_train.argmax(axis=1)
sample_weight=np.zeros(ndata)
for i in range(ndata):
    sample_weight[i]=weight[y_int[i]]


print("data prepared, ready to train!")

# construct the WaveNet model
wnc=WaveNetClassifier((25000,),(5,),kernel_size=2,dilation_depth=9,n_filters=40)

# training the WaveNet model
history=wnc.fit_wn(x_train,y_train,validation_data=(x_val,y_val),epochs=10,batch_size=30,beta=2.,optimizer='adam',save=True,save_dir='./check/',sample_weight=sample_weight)
