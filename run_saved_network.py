
'Written by me'

import numpy as np
import tensorflow as tf

path_to_saved_nw='./saved/'
saved_network='ts_class_network'

testing_data='test.csv'

data_test = np.loadtxt(testing_data,delimiter=',')
print('Loaded test data file %s with %d rows'% (testing_data,len(data_test)))

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

with tf.Session() as sess:
    # Load graph and variable values
    new_saver = tf.train.import_meta_graph(path_to_saved_nw+saved_network+'.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint(path_to_saved_nw))
    graph = tf.get_default_graph()

    # Get input placeholders from graph
    x = graph.get_tensor_by_name('Input_data:0')
    y_ = graph.get_tensor_by_name('Ground_truth:0')
    keep_prob = graph.get_tensor_by_name('keep_prob:0')
    bn_train = graph.get_tensor_by_name('bn_train:0')

    # Get accuracy operation from graph - note that this too is fetched as a tensor
    accuracy = graph.get_tensor_by_name('Evaluating_accuracy/accacc:0')
    prediction = graph.get_tensor_by_name('Evaluating_accuracy/prediction:0')

    # Evaluate accuracy on testing data.
    result = sess.run(accuracy, feed_dict={ x: X_test, y_: y_test,keep_prob: 1.0,bn_train : False})
    print('\nThe accuracy on the test data is %.3f\n' %(result))

    # Get predictions for alla rows
    pred_res=sess.run(prediction, feed_dict={ x: X_test, y_: y_test,keep_prob: 1.0,bn_train : False})

    print('Predictions for the first 10 lines are:')
    for pred,corr_res,data in list(zip(pred_res,y_test,X_test))[:10]:
        data=['%.1f'%x for x in data]
        # Adding base to restore original categories
        print('Pred: %d Truth: %d Data: %s'% (pred+base,int(corr_res+base),' '.join(data)))


    print('\nDone!')


