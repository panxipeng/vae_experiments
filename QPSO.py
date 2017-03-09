# This is a standard Quantum Particle Swarm Optimization algorithm

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

class QPSO():

    def __init__(self, particleNum, dim, maxIteration, rangeL, rangeR, rangeMax):

        self.decoder = load_model('Compiled/simpleAutoencoders_decoder.h5')
        self.nuclei_data = np.genfromtxt('CRCHistoDataSets/Detection/nucleis_data.dat', delimiter=',')
        self.nuclei_data = self.nuclei_data.astype('float32') / 255.
        self.nuclei = self.nuclei_data[4].copy()
        self.particleNum = particleNum #200
        self.maxIteration = maxIteration #1000
        self.dim = dim #64
        self.rangeL = rangeL #0
        self.rangeR = rangeR #15
        self.rangeMax = rangeMax #15

    def show_original(self):

        plt.imshow(self.nuclei.reshape(17, 17, 3, order="F"))
        plt.title('Original Input Image')
        plt.show()

    def cost(self, arg):

        decoded = self.decoder.predict(arg.reshape(1, 64))
        return np.sqrt(np.sum((decoded - self.nuclei)**2))

    def qpso(self):

        x = (self.rangeR - self.rangeL) * np.random.rand(self.particleNum, self.dim) + self.rangeL
        x = x.astype('float32')

        pbest = x.copy()
        gbest = np.zeros((self.dim), 'float32')
        mbest = np.zeros((self.dim), 'float32')

        f_x = np.zeros((self.particleNum), 'float32')
        f_pbest = np.zeros((self.particleNum), 'float32')

        for i in range(0, self.particleNum, 1):
            f_x[i] = self.cost(x[i])
            f_pbest[i] = f_x[i]

        g = np.argmin(f_pbest)
        gbest = pbest[g]
        f_gbest = f_pbest[g]

        MINIMUM = f_gbest

        for t in range(0, self.maxIteration, 1):

            beta = 0.5 * (self.maxIteration - t) / self.maxIteration + 0.5
            mbest = np.sum(pbest, axis = 0) / self.particleNum

            for i in range(0, self.particleNum, 1):

                fi = np.random.rand(self.dim)
                p = fi * pbest[i] + (1 - fi) * gbest
                u = np.random.rand(self.dim)
                b = beta * np.absolute(mbest - x[i])
                v = -1 * np.log(u)
                y = p + ((-1) ** np.ceil(0.5 + np.random.rand(self.dim))) * b * v
                x[i] = np.sign(y) * np.minimum(np.absolute(y), self.rangeMax)
                x[i] = np.absolute(x[i])
                f_x[i] = self.cost(x[i])

                if f_x[i] < f_pbest[i]:
                    pbest[i] = x[i]
                    f_pbest[i] = f_x[i]
                if f_pbest[i] < f_gbest:
                    gbest = pbest[i]
                    f_gbest = f_pbest[i]

                MINIMUM = f_gbest

            print(str(t) + " " + str(MINIMUM))
            to_show = self.decoder.predict(gbest.reshape(1, 64))
            plt.imshow(to_show.reshape(17, 17, 3, order="F"))
            plt.title('Loss: ' + str(MINIMUM))
            plt.savefig('Animation' + '\img' + str(t) + '.png')
            plt.clf()
            plt.cla()
            plt.close('all')

        print(MINIMUM)
        print(gbest)
        to_show = self.decoder.predict(gbest.reshape(1, 64))
        plt.imshow(to_show.reshape(17, 17, 3, order="F"))
        plt.show()







