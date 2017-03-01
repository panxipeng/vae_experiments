import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda
from keras.models import Model
from keras import backend as K
from keras import objectives

batch_size = 100
Image_Dim = 867
latent_dim = 64
intermediate_dim = 256
nb_epoch = 100

vae_input = Input(shape=(867,))
encoded_l1 = Dense(intermediate_dim, activation='relu')(vae_input)
z_mean = Dense(latent_dim)(encoded_l1)
z_log_var = Dense(latent_dim)(encoded_l1)

x_train = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_data.dat', delimiter=',')
x_train2 = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_neg_data.dat', delimiter=',')
x_train = x_train.astype('float32') / 255.
x_train2 = x_train2.astype('float32') / 255.
x_train = np.concatenate((x_train,x_train2), axis=0)
x_test = x_train2.copy()

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(64, ), mean=0.,
                              std=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling, output_shape=(64,))([z_mean, z_log_var])

decoded_l1 = Dense(intermediate_dim, activation='relu')(z)
vae_output = Dense(Image_Dim, activation='sigmoid')(decoded_l1)

def vae_loss(vae_input, vae_output):
    xent_loss = Image_Dim * objectives.binary_crossentropy(vae_input, vae_output)
    kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

vae = Model(vae_input, vae_output)
vae.compile(optimizer='Adadelta', loss=vae_loss)

vae_history = vae.fit(x_train, x_train,
        shuffle=True,
        nb_epoch=100,
        batch_size=50,
        validation_data=(x_train, x_train))

encoder = Model(input=vae_input, output=z_mean)

encoder.save('Compiled/variationalAutoencoders_encoder.h5')
vae.save('Compiled/variationalAutoencoders_autoencoder.h5')

# Save history
ae_history = np.array(vae_history.history['loss'])
np.savetxt('history_vA.csv', ae_history, delimiter=',')

decoded_imgs = vae.predict(x_train)
decoded_imgs_t = vae.predict(x_test)

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
