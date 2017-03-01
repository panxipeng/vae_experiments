from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt

x_train = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_data.dat', delimiter=',')
x_train2 = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_neg_data.dat', delimiter=',')
x_train = x_train.astype('float32') / 255.
x_train2 = x_train2.astype('float32') / 255.
x_train = np.concatenate((x_train,x_train2), axis=0)
x_test = x_train2.copy()


encoding_dim = 64

input_img = Input(shape=(867,))
input_l1 = Input(shape=(512,))
input_l2 = Input(shape=(256,))
input_l3 = Input(shape=(128,))

# # Layerwise Trainning
# encoded_l1 = Dense(512, activation='relu')(input_img)
# encoder_l1 = Model(input=input_img, output=Dense(867, activation='sigmoid')(encoded_l1))
# encoder_l1.compile(optimizer='Adadelta', loss='binary_crossentropy')
# encoder_l1.fit(x_train, x_train, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(x_train, x_train))
# encoder_l1_intermediate = Model(input=input_img, output=encoder_l1.layers[1].output)
#
# encoded_l2 = Dense(256, activation='relu')(input_l1)
# encoder_l2 = Model(input=input_l1, output=Dense(512, activation='sigmoid')(encoded_l2))
# encoder_l2.compile(optimizer='Adadelta', loss='binary_crossentropy')
# x_train_l2 = encoder_l1_intermediate.predict(x_train)
# encoder_l2.fit(x_train_l2, x_train_l2, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(x_train_l2, x_train_l2))
# encoder_l2_intermediate = Model(input=input_l1, output=encoder_l2.layers[1].output)
#
# encoded_l3 = Dense(128, activation='relu')(input_l2)
# encoder_l3 = Model(input=input_l2, output=Dense(256, activation='sigmoid')(encoded_l3))
# encoder_l3.compile(optimizer='Adadelta', loss='binary_crossentropy')
# x_train_l3 = encoder_l2_intermediate.predict(x_train_l2)
# encoder_l3.fit(x_train_l3, x_train_l3, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(x_train_l3, x_train_l3))
# encoder_l3_intermediate = Model(input=input_l2, output=encoder_l3.layers[1].output)
#
# encoded_l4 = Dense(64, activation='relu')(input_l3)
# encoder_l4 = Model(input=input_l3, output=Dense(128, activation='sigmoid')(encoded_l4))
# encoder_l4.compile(optimizer='Adadelta', loss='binary_crossentropy')
# x_train_l4 = encoder_l3_intermediate.predict(x_train_l3)
# encoder_l4.fit(x_train_l4, x_train_l4, nb_epoch=50, batch_size=256, shuffle=True, validation_data=(x_train_l4, x_train_l4))
# encoder_l4_intermediate = Model(input=input_l3, output=encoder_l4.layers[1].output)
#
# # Fine-tuning, reuse Weights
# encoded_l1 = Dense(512, activation='relu', weights=encoder_l1.layers[1].get_weights())(input_img)
# encoded_l2 = Dense(256, activation='relu', weights=encoder_l2.layers[1].get_weights())(encoded_l1)
# encoded_l3 = Dense(128, activation='relu', weights=encoder_l3.layers[1].get_weights())(encoded_l2)
# encoded = Dense(64, activation='relu', weights=encoder_l4.layers[1].get_weights())(encoded_l3)
#
# decoded = Dense(128, activation='relu')(encoded)
# decoded_l1 = Dense(256, activation='relu')(decoded)
# decoded_l2 = Dense(512, activation='relu')(decoded_l1)
# decoded_l3 = Dense(867, activation='sigmoid')(decoded_l2)

# For someone who don't want Laywise training

encoded_l1 = Dense(512, activation='relu')(input_img)
encoded_l2 = Dense(256, activation='relu')(encoded_l1)
encoded_l3 = Dense(128, activation='relu')(encoded_l2)
encoded = Dense(64, activation='relu')(encoded_l3)

decoded = Dense(128, activation='relu')(encoded)
decoded_l1 = Dense(256, activation='relu')(decoded)
decoded_l2 = Dense(512, activation='relu')(decoded_l1)
decoded_l3 = Dense(867, activation='sigmoid')(decoded_l2)

autoencoder = Model(input=input_img, output=decoded_l3)

encoder = Model(input=input_img, output=encoded)

autoencoder.compile(optimizer='Adadelta', loss='binary_crossentropy')

ae_history = autoencoder_history = autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=50,
                shuffle=True,
                validation_data=(x_train, x_train))

encoder.save('Compiled/stackedAutoencoders_encoder.h5')
autoencoder.save('Compiled/stackedAutoencoders_autoencoder.h5')

# Save history
ae_history = np.array(ae_history.history['loss'])
np.savetxt('history_stA.csv', ae_history, delimiter=',')

# Decode
decoded_imgs = autoencoder.predict(x_train)
decoded_imgs_t = autoencoder.predict(x_test)

# Plot samples and its representation
n = 10
plt.figure(figsize=(20, 8))
for i in range(n):

    ax = plt.subplot(4, n, i + 1)
    plt.imshow(x_train[i].reshape(17,17,3, order = "F"))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(17,17,3, order = "F"))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i + 1 + 2 * n)
    plt.imshow(x_test[i + 13].reshape(17, 17, 3, order="F"))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(4, n, i + 1 + 3 * n)
    plt.imshow(decoded_imgs_t[i + 13].reshape(17, 17, 3, order="F"))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()