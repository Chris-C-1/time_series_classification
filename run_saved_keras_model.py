
import numpy as np

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

val_datafile='final_validation.csv'
#val_datafile='test.csv'

data_test = np.loadtxt(val_datafile,delimiter=',')
print('Loaded test data file %s with %d rows'% (val_datafile,len(data_test)))

X_test = data_test[:,1:]
y_test = data_test[:,0]
Ntest = X_test.shape[0]
D = X_test.shape[1]
base = np.min(y_test)  #Check if categories are 0-based
if base != 0:
    print('Changing categories to be zero based by subtracting %d from all' % base)
    y_test -= base
else:
    print('Categories seem zero based')

X_test = np.expand_dims(X_test, axis=2)

load_model_name='keras_saved_model'
print('\nLoading Keras model "%s"\n' % load_model_name)
model = load_model(load_model_name)
#from graphviz import *
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

predictions=model.predict(X_test)
score = model.evaluate(X_test, y_test, verbose=0)
print('\nAccuracy on %s data is %.1f%%\n' % (val_datafile,100.0*score[1]))

res_pred=[]
cnt=0
for pred,truth in zip(predictions,y_test):
    if pred[0] > pred[1]:
        pred_val='NormDist'
    else:
        pred_val='EqDist'

    if  pred[0] > 0.35 and pred[1]>0.35:
        pred_val=pred_val+' (but unsure)'

    if truth<0.0001:
        truth_val='NormDist'
    else:
        truth_val='EqDist'
    cnt+=1
    res_pred.append((cnt,pred_val,truth_val,pred[0],pred[1]))

incorr_cnt=0

print('Example predictions:')
for s,p,t,v1,v2 in res_pred[:30]:
    print('Series %d, predicted: %s, truth: %s' % (s,p,t))

for s,p,t,v1,v2 in res_pred:
    if t not in p:
#        print('Series %d, predicted: %s, truth: %s (%.2f,%.2f)' % (s,p,t,v1,v2))
        incorr_cnt+=1

print('\n%.1f%% incorrect (%d of %d)\n' % (100.0*incorr_cnt/len(res_pred),incorr_cnt,len(res_pred)))

