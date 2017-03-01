# Nucleis detection on real test images. Here we adopt [img2 img4 img48], [img55, img81 img89] for detection.

import os
import ctypes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model

threshold = 0.6 # Empirical Parameter
encoding_dim = 64
instances = ['img2', 'img4', 'img48', 'img55', 'img81', 'img89']    # Just modify this list, if you wish to detect other images
encoder_marshalling = ['simpleAutoencoders_encoder.h5', 'stackedAutoencoders_encoder.h5', 'variationalAutoencoders_encoder.h5',
                       'variationalAutoencoders_stacked_encoder.h5', 'variationalAutoencoders_noKL_encoder.h5', 'variationalAutoencoders_stacked_noLayerwise_encoder.h5']
abbreviation = ['sA', 'stA', 'vA', 'vAs', 'vA_noKL', 'vAs_noLW']
classifier_marshalling = ['simpleAutoencoders.h5', 'stackedAutoencoders.h5', 'variationalAutoencoders.h5',
                       'variationalAutoencoders_stacked.h5', 'variationalAutoencoders_noKL.h5', 'variationalAutoencoders_stacked_noLayerwise.h5']

# plt.ion()
indicator_container = np.zeros([500, 500])
indicator_container = indicator_container.astype('float32')

for (AI_no, AI) in enumerate(encoder_marshalling):

    encoder_path = os.path.join('Compiled', AI)
    classifier_path = os.path.join('Compiled', 'Classifiers', classifier_marshalling[AI_no])

    encoder = load_model(encoder_path)
    classifier = load_model(classifier_path)

    for imgx in instances:

        xy = np.zeros([1, 2])
        current_path = os.path.join('CRCHistoDataSets', 'Detection', imgx)
        img_path = os.path.join(current_path, imgx + '.bmp')
        label_path = os.path.join(current_path, imgx + '_detection.csv')
        img = mpimg.imread(img_path)
        label = np.genfromtxt(label_path, delimiter=',')

        fig = plt.figure(figsize = (10, 10))
        plt.imshow(img)

        x_range = np.arange(8, 492, 1)
        y_range = np.arange(8, 492, 1)
        iii = 0
        for x in np.nditer(x_range):
            for y in np.nditer(y_range):
                img_block = img[(y-8):(y+9), (x-8):(x+9),:]
                patch = img_block.reshape(1, 867, order = "F") # Follow the Matlab reshape Fortran order.
                nuclei_indicator = classifier.predict(encoder.predict(patch))
                indicator_container[x, y] = nuclei_indicator
                iii += 1
                print(imgx + '_' + abbreviation[AI_no] + ' Patach: ' + str(iii))
                if nuclei_indicator > threshold:
                    xy = np.vstack((xy, np.array([x, y])))
        np.savetxt(current_path + '\\' + imgx + '_' + abbreviation[AI_no] + '_indicator.csv', indicator_container, delimiter=',')
        xy = xy[1:]
        plt.plot(xy[:,0], xy[:,1], 'ro', markersize= 4, label='Detected')     # Plot detected areas, using red circle
        plt.plot(label[:,0], label[:,1], 'b*', label = 'True') # Plot labeled real nucleis

        leg = plt.legend(loc = 'upper right')
        leg.get_frame().set_alpha(0.5)
        plt.title('Detection: ' + imgx + ' ' + classifier_marshalling[AI_no])
        plt.savefig(current_path + '\\' + imgx + '_' + abbreviation[AI_no] + '_detected.png')
        plt.savefig(current_path + '\\' + imgx + '_' + abbreviation[AI_no] + '_detected.eps')
        plt.savefig(current_path + '\\' + imgx + '_' + abbreviation[AI_no] + '_detected.svg')
        # plt.show()
        fig.clf()
        fig.clear()
        plt.clf()
        plt.cla()
        plt.close('all')
        print(abbreviation[AI_no] + '_', imgx + ' Finished')

print('Finished!')
ctypes.windll.user32.MessageBoxW(0, "Detection Finished", "Message", 1)