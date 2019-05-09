import matplotlib as plt
import pprint as pp
import numpy as np

import DataSet as DS
import NeuralNetwork as NN


filename = 'xor.txt'
data = DS.DataSet(filename)

n = data.getNumInputElem()

net = NN.NeuralNetwork(n,n,1)
inputs = data.getInputs()
targets = data.getTargets()

'''
pp.pprint(inputs)
pp.pprint(targets)
'''

for lr in np.arange(0.05,0.55,0.05):
    for epc in range(6000):
        error = []
        for x,y in zip(inputs,targets):
            error.append(abs(max(net.train(x,y,lr),key=abs)))
        #print(error)
        if sum(i <= 0.05 for i in error)>=(2*len(error))/3:
            #print("LR:"+str(lr)+", Epoch:"+str(epc))
            break
    print("Training with learning rate = "+str(lr)+" Epochs:"+str(epc+1))
    
_,guess = net.feedFoward([0,1,1,1])
#_,guess = net.feedFoward([1,0,0,0,1,1,1,1,1,0,0])
print(guess)


for lr in np.arange(0.05,0.55,0.05):
    net = NN.NeuralNetwork(n,n,1)
    for epc in range(6000):
        error = []
        for x,y in zip(inputs,targets):
            error.append(abs(max(net.train(x,y,lr),key=abs)))
        #print(error)
        if sum(i <= 0.05 for i in error)>=(2*len(error))/3:
            #print("LR:"+str(lr)+", Epoch:"+str(epc))
            break
    print("Training with learning rate = "+str(lr)+" Epochs:"+str(epc+1))
