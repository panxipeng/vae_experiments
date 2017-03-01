from keras.layers import Input, Dense
from keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras import regularizers

encoding_dim = 64

input_img = Input(shape=(867,))

encoded = Dense(encoding_dim, activation='relu')(input_img)

decoded = Dense(867, activation='sigmoid')(encoded)

autoencoder = Model(input=input_img, output=decoded)

x_train = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_data.dat', delimiter=',')
x_train2 = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_neg_data.dat', delimiter=',')
x_train = x_train.astype('float32') / 255.
x_train2 = x_train2.astype('float32') / 255.
x_train = np.concatenate((x_train,x_train2), axis=0)
x_test = x_train2.copy()

encoder = Model(input=input_img, output=encoded)

encoded_input = Input(shape=(encoding_dim,))

decoder_layer = autoencoder.layers[-1]

decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')


ae_history = autoencoder.fit(x_train, x_train,
                nb_epoch=100,
                batch_size=50,
                shuffle=True,
                validation_data=(x_train, x_train))

# Save Models
encoder.save('Compiled/simpleAutoencoders_encoder.h5')
decoder.save('Compiled/simpleAutoencoders_decoder.h5')
autoencoder.save('Compiled/simpleAutoencoders_autoencoder.h5')

# Save history
ae_history = np.array(ae_history.history['loss'])
np.savetxt('history_sA.csv', ae_history, delimiter=',')

# encode and decode some digits
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