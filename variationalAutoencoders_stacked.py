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
intermediate_dim = 512
nb_epoch = 100

x_train = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_data.dat', delimiter=',')
x_train2 = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_neg_data.dat', delimiter=',')
x_train = x_train.astype('float32') / 255.
x_train2 = x_train2.astype('float32') / 255.
x_train = np.concatenate((x_train,x_train2), axis=0)
x_test = x_train2.copy()

vae_input = Input(shape=(867, ))
input_l1 = Input(shape=(512, ))
input_l2 = Input(shape=(256, ))
input_l3 = Input(shape=(128, ))

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(64, ), mean=0.,
                              std=1.0)
    return z_mean + K.exp(z_log_var / 2) * epsilon

# Layerwise Trainning
encoded_l1 = Dense(512, activation='relu')(vae_input)
z_mean_l1 = Dense(latent_dim)(encoded_l1)
z_log_var_l1 = Dense(latent_dim)(encoded_l1)
z_l1 = Lambda(sampling, output_shape=(latent_dim,))([z_mean_l1, z_log_var_l1])
de_l1 = Dense(512, activation='relu')(z_l1)
encoder_l1 = Model(input=vae_input, output=Dense(867, activation='sigmoid')(de_l1))
encoder_l1.compile(optimizer='Adadelta', loss='binary_crossentropy')
encoder_l1.fit(x_train, x_train, nb_epoch=50, shuffle=True, batch_size=100, validation_data=(x_train, x_train))
encoder_l1_intermediate = Model(input=vae_input, output=encoder_l1.layers[1].output)

encoded_l2 = Dense(256, activation='relu')(input_l1)
z_mean_l2 = Dense(latent_dim)(encoded_l2)
z_log_var_l2 = Dense(latent_dim)(encoded_l2)
z_l2 = Lambda(sampling, output_shape=(latent_dim,))([z_mean_l2, z_log_var_l2])
de_l2 = Dense(256, activation='relu')(z_l2)
encoder_l2 = Model(input=input_l1, output=Dense(512, activation='sigmoid')(de_l2))
encoder_l2.compile(optimizer='Adadelta', loss='binary_crossentropy')
x_train_l2 = encoder_l1_intermediate.predict(x_train)
encoder_l2.fit(x_train_l2, x_train_l2, nb_epoch=50, batch_size=100, shuffle=True, validation_data=(x_train_l2, x_train_l2))
encoder_l2_intermediate = Model(input=input_l1, output=encoder_l2.layers[1].output)

encoded_l3 = Dense(128, activation='relu')(input_l2)
z_mean_l3 = Dense(latent_dim)(encoded_l3)
z_log_var_l3 = Dense(latent_dim)(encoded_l3)
z_l3 = Lambda(sampling, output_shape=(latent_dim,))([z_mean_l3, z_log_var_l3])
de_l3 = Dense(128, activation='relu')(z_l3)
encoder_l3 = Model(input=input_l2, output=Dense(256, activation='sigmoid')(de_l3))
encoder_l3.compile(optimizer='Adadelta', loss='binary_crossentropy')
x_train_l3 = encoder_l2_intermediate.predict(x_train_l2)
encoder_l3.fit(x_train_l3, x_train_l3, nb_epoch=50, batch_size=100, shuffle=True, validation_data=(x_train_l3, x_train_l3))
encoder_l3_intermediate = Model(input=input_l2, output=encoder_l3.layers[1].output)

z_mean_container = Model(input=input_l2, output=Dense(256, activation='sigmoid')(z_mean_l3))
z_log_var_container = Model(input=input_l2, output=Dense(256, activation='sigmoid')(z_log_var_l3))

encoded_l1 = Dense(512, activation='relu', weights=encoder_l1.layers[1].get_weights())(vae_input)
encoded_l2 = Dense(256, activation='relu', weights=encoder_l2.layers[1].get_weights())(encoded_l1)
encoded_l3 = Dense(128, activation='relu', weights=encoder_l3.layers[1].get_weights())(encoded_l2)
z_mean = Dense(latent_dim, weights=z_mean_container.layers[2].get_weights())(encoded_l3)
z_log_var = Dense(latent_dim, weights=z_log_var_container.layers[2].get_weights())(encoded_l3)

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

decoded_l1 = Dense(128, activation='relu')(z)
decoded_l2 = Dense(256, activation='relu')(decoded_l1)
decoded_l3 = Dense(512, activation='relu')(decoded_l2)
vae_output = Dense(Image_Dim, activation='sigmoid')(decoded_l3)

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

encoder.save('Compiled/variationalAutoencoders_stacked_encoder.h5')
vae.save('Compiled/variationalAutoencoders_stacked_autoencoder.h5')

# Save history
ae_history = np.array(vae_history.history['loss'])
np.savetxt('history_vAs.csv', ae_history, delimiter=',')

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
