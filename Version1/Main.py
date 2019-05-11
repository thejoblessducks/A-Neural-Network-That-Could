import matplotlib as plt
import os
import pprint as pp
from prettytable import PrettyTable
import numpy as np
from math import ceil

import DataSet as DS
import NeuralNetwork as NN

'''-----------------------------------------------------------------------------
Present options for user
-----------------------------------------------------------------------------'''
def presentProgram():
    print("Files:\n"+str(os.listdir("./Data/"))+"\n")
    filename = input(str("Select input file:"))
    print
    guessfilename = input(str("Select test file:"))
    print
    is_test = input(str("Testing, make random seed (yes/no):"))
    is_test = is_test == "yes"
    print
    runtimes = int(input(str("Times to run program:")))
    print
    func = int(input(str("Which function to choose: trainSame/trainDifferent  (0/1):")))
    print
    return filename,guessfilename,is_test,runtimes,func


'''-----------------------------------------------------------------------------
Open and read a File with data 
-----------------------------------------------------------------------------'''
def openDataSet(filename,v):
    data = DS.DataSet("./Data/"+filename)
    #Number of input units, one for each element in input entry
    n = data.getNumInputElem()
    inputs = data.getInputs(test=v)
    targets = data.getTargets()
    return n,inputs,targets


'''-----------------------------------------------------------------------------
Train and test the same Neural Network for different Learning Rates
-----------------------------------------------------------------------------'''
def trainSame(n,inputs,targets,guessfilename,testing):
    print("-----Neural Networks with "+str(n)+"Input and hidden units and 1 output unit-----\n")
    
    net = NN.NeuralNetwork(n,n,1,testing)
    
    epochs = int(input("Number of Epochs:"))
    
    #Tables Creation
    tb = PrettyTable()
    tb.title = "Same Network new Learning Rates"
    tb.field_names = ["Learning Rate","Epochs"]

    for lr in np.arange(0.05,0.55,0.05):
        for epc in range(epochs):
            error = []
            #Trains the network for every input entry and saves the error for each
            for x,y in zip(inputs,targets):
                error.append(abs(max(net.train(x,y,lr),key=abs)))
            #Break condition
            if sum(i <= 0.05 for i in error)==len(error):#>=(2*len(error))/3:
                break
        #Adds row to table for learning rate and its iterations 
        tb.add_row([str(ceil(lr*100.0)/100.0),str(epc+1)])
    print(tb)
    print()
    
    #Tests the network on unknown data
    entries,predictions = makePrediction(net,guessfilename)

    #Prepares table
    tb = PrettyTable()
    tb.title = "Guessing Output"
    tb.field_names = ["Entry","Prediction"]
    #Makes table
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
        #For every new learning rate, create a new neural network
        net = NN.NeuralNetwork(n,n,1,testing)
        for epc in range(epochs):
            error = []
            for x,y in zip(inputs,targets):
                error.append(abs(max(net.train(x,y,lr),key=abs)))
            if sum(i <= 0.05 for i in error)>=(2*len(error))/3:
                break
        tb.add_row([str(version),str(ceil(lr*100.0)/100.0),str(epc+1)])

        #Tests this iteration network for all the unknown data
        entries,predictions = makePrediction(net,guessfilename)        
        for entry,prediction in zip(entries,predictions):        
            tg.add_row([str(version),str(entry),str(prediction)])

    print(tb)
    print()
    print(tg)


'''-----------------------------------------------------------------------------
Use Neural Network to make predictions for file
-----------------------------------------------------------------------------'''
def makePrediction(network,guessfilename):
    _,entries,_ = openDataSet(guessfilename,True)
    predictions = []
    for entry in entries:
        prediction = network.predict(entry)
        predictions.append(prediction)
    return entries,predictions



#-------------------------------------------------------------------------------
filename,guessfilename,is_test,runtimes,func = presentProgram()
n,inputs,targets = openDataSet(filename,False)
for _ in range(runtimes):
    if not func:
        trainSame(n,inputs,targets,guessfilename,is_test)
    else:
        trainDifferent(n,inputs,targets,guessfilename,is_test)
