import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt

from keras.utils import np_utils

"""Load the data"""

training_data='train.csv'
testing_data='test.csv'
save_network_as='./saved/ts_class_network'


data_train = np.loadtxt(training_data,delimiter=',')
print('Loaded training data %s with %d rows'% (training_data,len(data_train)))
data_test_val = np.loadtxt(testing_data,delimiter=',')
print('Loaded testing data %s with %d rows'% (testing_data,len(data_test_val)))
data_test,data_val = np.split(data_test_val,2)
print('Splitted testing data into %d for testing and %d for validation' % (len(data_test),len(data_val)))
# Usually, the first column contains the target labels
X_train = data_train[:,1:]
X_val = data_val[:,1:]
X_test = data_test[:,1:]
N = X_train.shape[0]
Ntest = X_test.shape[0]
D = X_train.shape[1]
y_train = data_train[:,0]
y_val = data_val[:,0]
y_test = data_test[:,0]
print('We have training data with %s entries, each with %s data points'%(N,D))
base = np.min(y_train)  #Check if data is 0-based
if base != 0:
    print('Changing categories to be zero based by subtracting %d from all' % base)
    y_train -=base
    y_val -= base
    y_test -= base
else:
    print('Categories seem zero based')

X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)
X_val = np.expand_dims(X_val, axis=2)

model = Sequential()
model.add(Convolution1D(12,6, input_shape=(D,1),use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Convolution1D(8, 3,use_bias=True))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dropout(0.50))
model.add(Dense(12, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,validation_data=(X_val,y_val),
          batch_size=32, epochs=200, verbose=0)

save_model_name='keras_saved_model'
model.save(save_model_name)
print('Saving Keras model as "%s"' % save_model_name)

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

score = model.evaluate(X_test, y_test, verbose=1)

print('\nAccuracy on test data is %.1f%%\n' % (100.0*score[1]))

#print(score)

