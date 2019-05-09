import matplotlib as plt
import pprint as pp
import numpy as np

import DataSet as DS
import NeuralNetwork as NN


filename = 'parity4.txt'
data = DS.DataSet(filename)

n = data.getNumInputElem()

net = NN.NeuralNetwork(n,n,1)
inputs = data.getInputs()
targets = data.getTargets()

'''
pp.pprint(inputs)
pp.pprint(targets)
'''
print("-----Neural Networks with "+str(n)+"Input and hidden units and 1 output unit-----\n")
print("Testing the same Neural Network with increasing values for the learning reate:")
for lr in np.arange(0.05,0.55,0.05):
    for epc in range(1000):
        error = []
        for x,y in zip(inputs,targets):
            error.append(abs(max(net.train(x,y,lr),key=abs)))
        if sum(i <= 0.05 for i in error)>=(2*len(error))/3:
            break
    print("Training with learning rate = "+str(lr)+" Epochs:"+str(epc+1))

entry = [0,1,1,1]
guess = net.predict(entry)
#entry = [1,0,0,0,1,1,1,1,1,0,0]
#guess = net.predict(entry)
print("\nGuessing "+str(entry)+": "+str(guess))
print()

print("Training diferent neural networks for diferent learning rates:")
for version,lr in enumerate(np.arange(0.05,0.55,0.05)):
    net = NN.NeuralNetwork(n,n,1)
    for epc in range(1000):
        error = []
        for x,y in zip(inputs,targets):
            error.append(abs(max(net.train(x,y,lr),key=abs)))
        if sum(i <= 0.05 for i in error)>=(2*len(error))/3:
            break
    print("Network:"+str(version)+"--Training with learning rate = "+str(lr)+" Epochs:"+str(epc+1))

print("-----------------------------------------------------------------------------------")