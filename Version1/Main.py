import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
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
    show_graph = False
    if not func:
        show_graph = input(str("Show graph, the higher the epochs the slower the graph (yes/no):"))
        show_graph = show_graph == "yes"
    return filename,guessfilename,is_test,runtimes,func,show_graph


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
def trainSame(n,inputs,targets,guessfilename,testing,show_graph):
    print("-----Neural Networks with "+str(n)+"Input and hidden units and 1 output unit-----\n")
    
    net = NN.NeuralNetwork(n,n,1,testing)
    
    epochs = int(input("Number of Epochs:"))
    
    #Tables Creation
    tb = PrettyTable()
    tb.title = "Same Network new Learning Rates"
    tb.field_names = ["Learning Rate","Epochs","Mean Error"]

    #Will only be used if user want to see graph
    learning_rate_errors = dict()

    for lr in np.arange(0.05,0.55,0.05):
        errors = []
        for epc in range(epochs):
            error = []
            #Trains the network for every input entry and saves the error for each
            for x,y in zip(inputs,targets):
                error.append(abs(max(net.train(x,y,lr),key=abs)))
            errors.append(np.mean(error,dtype=float))
            #Break condition
            if sum(i <= 0.05 for i in error)==len(error):#>=(2*len(error))/3:
                break
        #Adds row to table for learning rate and its iterations and the mean value for this learning rate
        l_r = ceil(lr*100.0)/100.0
        tb.add_row([str(l_r),str(epc+1),str(np.mean(errors,dtype=float))])
        #Adds a new key to dictionary, keys=learning rates, values=[error for every epoch]
        learning_rate_errors[lr] = errors
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

    #Graphs
    if show_graph:
        #User wants to see overall graph
        drawError(learning_rate_errors)
    #Show neural network concept graph, does not account for weight influence
    #On a next version it would be nice to implement that feature
    fig = plt.figure(figsize=(12, 12))
    ax = fig.gca()
    ax.axis('off')
    draw_neural_net(ax, .1, .9, .1, .9, [n, n, 1])
    plt.show()


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




'''-----------------------------------------------------------------------------
Function to show Neural Network Cartoon--Not our implementation 
Author: Colin Raffel  https://colinraffel.com/
Code reference: https://gist.github.com/craffel/2d727968c3aaebd10359
-----------------------------------------------------------------------------'''
def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    '''
    Draw a neural network cartoon using matplotilb.
    :usage:
        >>> fig = plt.figure(figsize=(12, 12))
        >>> draw_neural_net(fig.gca(), .1, .9, .1, .9, [4, 7, 2])
    :parameters:
        - ax : matplotlib.axes.AxesSubplot
            The axes on which to plot the cartoon (get e.g. by plt.gca())
        - left : float
            The center of the leftmost node(s) will be placed here
        - right : float
            The center of the rightmost node(s) will be placed here
        - bottom : float
            The center of the bottommost node(s) will be placed here
        - top : float
            The center of the topmost node(s) will be placed here
        - layer_sizes : list of int
            List of layer sizes, including input and output dimensionality
    '''
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    # Nodes
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,color='w', ec='k', zorder=4)
            ax.add_artist(circle)
    # Edges
    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k')
                ax.add_artist(line)

def drawError(dictionary):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    for lr in dictionary:
        errors = dictionary[lr]
        for epc,error in enumerate(errors):
            ax.scatter(lr,epc,error,c='g')
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Epochs")
    ax.set_zlabel("Mean Error")
    plt.show()

#-------------------------------------------------------------------------------
filename,guessfilename,is_test,runtimes,func,show_graph = presentProgram()
n,inputs,targets = openDataSet(filename,False)
for _ in range(runtimes):
    if not func:
        trainSame(n,inputs,targets,guessfilename,is_test,show_graph)
    else:
        trainDifferent(n,inputs,targets,guessfilename,is_test)
