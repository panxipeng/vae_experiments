from keras.layers import Input, Dense
from keras.models import Model
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


encoding_dim = 64

encoder = load_model('Compiled/variationalAutoencoders_stacked_noLayerwise_encoder.h5')    # Please modify this to your wanted encoder.

# Get all of the coded images from the Trainning Sets
x_train = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_data.dat', delimiter=',')
x_train = x_train.astype('float32') / 255.
x_classi_positive = encoder.predict(x_train)
y_classi_positive = np.ones(13400)

x_train_neg = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_neg_data.dat', delimiter=',')
x_train_neg= x_train_neg.astype('float32') / 255.
x_classi_negative = encoder.predict(x_train_neg)
y_classi_negative = np.zeros(13400)

# Get all of the coded images from the Test Sets
x_test = np.genfromtxt('CRCHistoDataSets/Detection/test_nucleis_data.dat', delimiter=',')
x_test = x_test.astype('float32') / 255.
test_classi_positive = encoder.predict(x_test)

x_test_n = np.genfromtxt('CRCHistoDataSets/Detection/test_nucleis_neg_data.dat', delimiter=',')
x_test_n= x_test_n.astype('float32') / 255.
test_classi_negative = encoder.predict(x_test_n)

# Build new coded labeled trainning sets
classi_sets_x = np.concatenate((x_classi_positive,x_classi_negative), axis=0)
classi_sets_y = np.concatenate((y_classi_positive,y_classi_negative), axis=0)

# Start to build a two layers classifier
input_img = Input(shape=(64, ))
classifier_l1 = Dense(32, activation='relu')(input_img)
classifier_l2 = Dense(8, activation='relu')(classifier_l1)
classifier_out = Dense(1, activation='sigmoid')(classifier_l2)
classifier = Model(input=input_img, output=classifier_out)

classifier.compile(optimizer='Adadelta', loss='binary_crossentropy')

classifier_history = classifier.fit(classi_sets_x, classi_sets_y,
                nb_epoch=50,
                batch_size=25,
                shuffle=True,
                validation_data=(classi_sets_x, classi_sets_y))

classifier.save('Compiled/Classifiers/variationalAutoencoders_stacked_noLayerwise.h5') # Please modify this to your wanted encoder.

# Save history
ae_history = np.array(classifier_history.history['loss'])
np.savetxt('history_classifier_vAs_noLW.csv', ae_history, delimiter=',')

# Training Sets Precision
result_neg = classifier.predict(x_classi_negative)
result_pos = classifier.predict(x_classi_positive)
np.savetxt('Training_rawResult_vAs_noLW.csv', np.concatenate((result_pos,result_neg), axis=0), delimiter=',')
result_neg[result_neg<0.2] = 0
result_pos[result_pos>0.8] = 1
print('Trainning Sets Precision:')
print(np.count_nonzero(result_pos == 1) / 13400)
print(np.count_nonzero(result_neg == 0) / 13400)
np.savetxt('Training_Precision_vAs_noLW.csv', np.array([np.count_nonzero(result_pos == 1), np.count_nonzero(result_neg == 0)]), delimiter=',')

# Test Sets Precision
result_neg = classifier.predict(test_classi_negative)
result_pos = classifier.predict(test_classi_positive)
np.savetxt('Test_rawResult_vAs_noLW.csv', np.concatenate((result_pos,result_neg), axis=0), delimiter=',')
result_neg[result_neg<0.2] = 0
result_pos[result_pos>0.8] = 1
print('Test Sets Precision:')
print(np.count_nonzero(result_pos == 1) / 13400)
print(np.count_nonzero(result_neg == 0) / 13400)
np.savetxt('Test_Precision_vAs_noLW.csv', np.array([np.count_nonzero(result_pos == 1), np.count_nonzero(result_neg == 0)]), delimiter=',')

