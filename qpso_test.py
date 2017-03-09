from QPSO import QPSO

q = QPSO(particleNum=200, dim=64, maxIteration=1000, rangeL=0, rangeR=15, rangeMax=15)

q.show_original()

q.qpso()
