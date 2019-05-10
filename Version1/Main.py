import matplotlib as plt
import pprint as pp
from prettytable import PrettyTable
import numpy as np
from math import ceil

import DataSet as DS
import NeuralNetwork as NN


'''-----------------------------------------------------------------------------
Open and read a File with data 
-----------------------------------------------------------------------------'''
def openDataSet(filename,v):
    data = DS.DataSet("./Data/"+filename)
    n = data.getNumInputElem()
    inputs = data.getInputs(test=v)
    targets = data.getTargets()
    return n,inputs,targets

def makePrediction(network,guessfilename):
    _,entries,_ = openDataSet(guessfilename,True)
    predictions = []
    for entry in entries:
        prediction = network.predict(entry)
        predictions.append(prediction)
    return entries,predictions
'''-----------------------------------------------------------------------------
Train and test the same Neural Network for different Learning Rates
-----------------------------------------------------------------------------'''
def trainSame(n,inputs,targets,guessfilename,testing):
    print("-----Neural Networks with "+str(n)+"Input and hidden units and 1 output unit-----\n")
    net = NN.NeuralNetwork(n,n,1,testing)

    epochs = int(input("Number of Epochs:"))

    tb = PrettyTable()
    tb.title = "Same Network new Learning Rates"
    tb.field_names = ["Learning Rate","Epochs"]
    for lr in np.arange(0.05,0.55,0.05):
        for epc in range(epochs):
            error = []
            for x,y in zip(inputs,targets):
                error.append(abs(max(net.train(x,y,lr),key=abs)))
            if sum(i <= 0.05 for i in error)==len(error):#>=(2*len(error))/3:
                break
        tb.add_row([str(ceil(lr*100.0)/100.0),str(epc+1)])
    print(tb)
    print()
    
    entries,predictions = makePrediction(net,guessfilename)
    tb = PrettyTable()
    tb.title = "Guessing Output"
    tb.field_names = ["Entry","Prediction"]
    for entry,prediction in zip(entries,predictions):        
        tb.add_row([str(entry),str(prediction)])
    print(tb)

'''-----------------------------------------------------------------------------
Train and test different Neural Networks for different Learning Rates
-----------------------------------------------------------------------------'''
def trainDifferent(n,inputs,targets,guessfilename,testing):
    print("-----Neural Networks with "+str(n)+"Input and hidden units and 1 output unit-----\n")
    net = NN.NeuralNetwork(n,n,1)

    epochs = int(input("Number of Epochs:"))

    tb = PrettyTable()
    tb.title = "Different Networks new Learning Rates"
    tb.field_names = ["Network","Learning Rate","Epochs"]

    tg = PrettyTable()
    tg.title = "Guessing Output for every Network"
    tg.field_names = ["Network Id","Entry","Prediction"]

    for version,lr in enumerate(np.arange(0.05,0.55,0.05)):
        net = NN.NeuralNetwork(n,n,1,testing)
        for epc in range(epochs):
            error = []
            for x,y in zip(inputs,targets):
                error.append(abs(max(net.train(x,y,lr),key=abs)))
            if sum(i <= 0.05 for i in error)>=(2*len(error))/3:
                break
        tb.add_row([str(version),str(ceil(lr*100.0)/100.0),str(epc+1)])

        entries,predictions = makePrediction(net,guessfilename)        
        for entry,prediction in zip(entries,predictions):        
            tg.add_row([str(version),str(entry),str(prediction)])

    print(tb)
    print()
    print(tg)
#-------------------------------------------------------------------------------
filename = "parity11.txt"
guessfilename = "parity11_predictions.txt"
n,inputs,targets = openDataSet(filename,False)
trainSame(n,inputs,targets,guessfilename,True)
