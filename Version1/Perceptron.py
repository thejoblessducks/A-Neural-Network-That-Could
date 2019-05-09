from __future__ import division
import numpy as np


'''-----------------------------------------------------------------------------
Perceptron Class -- Single Neuron
-----------------------------------------------------------------------------'''
class Perceptron():
###############################################################################
# This class reprecents a single neural network neuron (buiding block)        #
# The perceptron is fed a series of inputs, each forms a connection to the    #
#     main processing unit and after some calculations the neuron returns a   #
#     value (the output), this output can either be +1 or -1 depending on     #
#     which class the data belongs to, when the outout is -1, the neuron has  #
#     incorrectly classified the data, as such it requires tweeking:          #
#         gradient descent                                                    #
#                                                                             #
#                   X0------->|                                               #
#                             |----->[Neuron]----->Output(y)                  #
#                   X1------->|                                               #
#                                                                             #
# We need to have a weight(Wi) for every connection of inputs (Xi-->), to     #
#     find the weights we want to find those that are optimal, that provide   #
#     the best results with the least amount of error, in the begining we     #
#     will have to randomly weigh all the inputs                              #
#                                                                             #
# The Perceptron then for all its inputs sums the product between the input   #
#     and its weight: sum = [E(I=0 to inputs) Xi*Wi]  (Step 1)                #
#                                                                             #
# After the Sum step we will apply an Activation Function that conducts the   #
#     output towards a range, we will consider the function to be:            #
#         f(data) = 1/(1+e^(-data))                                           #
#                                                                             #
#                                                                             #
# The algorithm can be:                                                       #
#     1) For every input, multiply that input by its weight                   #
#     2) Sum all of the weighted inputs                                       #
#     3) Compute the output of the perceptron based on that sum passed through#
#     an activation function (sign of the sum)                                #
###############################################################################
    f = lambda self,x: 1/(1+np.exp(-1*x)) #Activation function

    def __init__(self,inputs=None):
        if inputs is not None:
            self.inputs = inputs
            #self.final_class = final_class
            self.weights = []
            for x in inputs:
                self.weights.append(np.random.uniform(-1,1))
        else:
            self.inputs = []
            self.weights = []
    def addConnection(self,conn):
        #Connection from first to hidden and hidden to output
        self.inputs.append(conn)
        self.weights.append(np.random.uniform(-1,1))
    def makeDecision(self, matrix_inputs):
        #Given a matrix of all the inputs in all the 
        weights = self.weights
        all_sum = 0        
        for i,x in enumerate(inputs):
            all_sum += x*weights[i]
            i += 1
        self.value_classification = self.f(all_sum)
    