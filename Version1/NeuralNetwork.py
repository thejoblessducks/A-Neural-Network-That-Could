from __future__ import division
import numpy as np


'''-----------------------------------------------------------------------------
Perceptron Class -- Single Layer Neuron
-----------------------------------------------------------------------------'''
class Perceptron():
    ###############################################################################
    # This class represents a single neural network neuron (building block)       #
    # The perceptron is fed a series of inputs, each forms a connection to the    #
    #     main processing unit and after some calculations the neuron returns a   #
    #     value (the output that output can be compared with its true value       #
    #     allowing us to tweak the weights in order to produce correct results:   #
    #         gradient descent                                                    #
    #                                                                             #
    #                   X0------->|                                               #
    #                             |----->[Neuron]----->Output(y)                  #
    #                   X1------->|                                               #
    #                                                                             #
    # We need to have a weight(Wi) for every connection of inputs (Xi-->), to     #
    #     find the weights we want to find those that are optimal, that provide   #
    #     the best results with the least amount of error, in the begining we     #
    #     will have to randomly weigh all the inputs, the larger the weight the   #
    #     more influential the corresponding input.                               #
    # The Perceptron then for all its inputs sums the product between the input   #
    #     and its weight: sum = [E(I=0 to inputs) Xi*Wi] + bias (Step 1)          #
    # After the Sum step we will apply an Activation Function that conducts the   #
    #     output towards a range, we will consider the function to be:            #
    #         f(data) = 1/(1+e^(-data))                                           #
    #                                                                             #
    # After the Perceptron thinks, it can check if the output is as expected      #
    #   calculating an error, since we can't change the input data we can only    #´
    #   change the weights in order to approach a better result, adjusting the    #
    #   weights is key in the perceptron process, as such the new weight will be: #
    #       Wi = Wi +(Y-output)*Xi                                                #
    # The algorithm can be:                                                       #
    #     1) For every input, multiply that input by its weight                   #
    #     2) Sum all of the weighted inputs                                       #
    #     3) Compute the output of the perceptron based on that sum passed through#
    #     an activation function (sign of the sum)                                #
    #     4) Calculate the error from the output and tweak the weights and bias   #
    #                                                                             #
    # This class isn't used in the project, just serves as a guidance to          #
    #   understand how a neuron in the network works                              #
    ###############################################################################
    f = lambda self,x: 0 if x<=0 else 1 #Activation function
    def __init__(self,inputs,target):
        self.inputs = inputs
        self.target = target
        self.weights = [np.random.uniform(-1,1)for _ in inputs]
        self.bias = np.random.uniform(-1,1)
    def train(self):
        inputs = self.inputs
        weights = self.weights
        target = self.target
        #Process the inputs-activation
        output = self.think(inputs,weights)
        #Calculate the error
        error = target-output
        #Adjust the weights
        weights = [w+error*x for w,x in zip(weights,inputs)]
        self.weights = weights
    def think(self,inputs,weights=None):
        if weights is None:
            weights = self.weights
        #Calculates the product sum, and adds the bias
        all_sum = sum([x*w]for x,w in zip(inputs,weights))+self.bias
        #Activates result
        return self.f(all_sum)
    


'''-----------------------------------------------------------------------------
Neural Network - MLPerceptron (2 layers)
-----------------------------------------------------------------------------'''
class NeuralNetwork():
    ############################################################################
    # In the previous class we introduced a single neuron capable, this neuron #
    #     allows us to classify linearly separable inputs, that is, we can draw#
    #     a plane/line seperating the possible outputs, however if our data in #
    #     non-linearly separable, the perceptron will fail to classify it.     #
    #                                                                          #
    # Regardless of its complexity, linear activation function are only viable #
    #     in one layer deep Perceptrons, this limits the Perceptron for almost #
    #     all real world problems are non-linear. As such we use non-linear    #
    #     activation functions that are continuous and differentiable in       #
    #     our weights range [-1,1]                                             #
    #                                                                          #
    # As such we will have to work with MLP (multi-layer perceptrons), this    #
    #     is accomplished with HIDDEN LAYERS (neuron nodes stacked in between  #
    #     inputs and outputs, allowing neural networks to learn more           #
    #     complicated attributes/features) and BACKPROPAGATION (a procedure to #
    #     repeatedly adjust the weights so as to minimize the difference       #
    #     between the actual output and the desired one) the more neurons your #
    #     network possesses and the more hidden layers,the more precise the out#
    #                                                                          #
    # For this implementation we will focus on a 2 layer Neural Network where  #
    #     the number of input units is proportional to the columns in inputs   #
    # In a future implementation we could enable multiple hidden layers        #
    ############################################################################
    def __init__(self,ninputs,nhidden,nout):
        ##################################################################
        # We can visualize the weights for every layer as a matrix where #
        #     every row represents an input unit (that unit can be an    #
        #     input node if the weights are for the hidden layer or the  #
        #     hidden nodes if the layer is the output layer) and every   #
        #     column represents a node in the forward layer              #
        #                                                                #
        # Since our implementation only has 2 layers (the input layer    #
        #     isn't considered a neural layer) we only need 2 matrixes   #
        #     for the weights (1 for the connections I--->H and 1 for    #
        #     the connections H--->O) and 2 lists with the bias for      #
        #     every layer (for every list, initially we have n random    #
        #     bias, where n is the number of units in the corresponding  #
        #     layer) in backpropagation all this values might be tweaked #
        ##################################################################
        self.nin = ninputs
        self.nhid = nhidden
        self.no = nout
        #Initiate weights randomly [-1,1]
        self.weights_ih = [[np.random.uniform(-1,1) for _ in range(ninputs)]for _ in range(nhidden)]
        self.weights_ho = [[np.random.uniform(-1,1) for _ in range(nhidden)]for _ in range(nout)]
        #initiate bias randomly[0,1]
        self.bias_hidden = [np.random.uniform(-1,1) for _ in range(nhidden)]
        self.bias_out = [np.random.uniform(-1,1) for _ in range(nout)]

    def sigmoid(self,x,deriv=False):
        if not deriv:
            return 1/(1+np.exp(-x))
        return x*(1-x)

    def activation(self,weights, inputs,bias):
        #generating sum
        #Since we are saving the weights in a matrix, the operation is a dot product
        all_sum = np.dot(weights,inputs)
        #Adding bias and activating
        activated = 0
        if type(all_sum)==list:
            all_sum = [val+b for val,b in zip(all_sum,bias)]
            activated = [self.sigmoid(x) for x in all_sum]
        else:
            all_sum += bias
            activated = self.sigmoid(all_sum)
        return all_sum,activated

    def feedForward(self,inputs):
        #Passes input forward in the network
        #generating sum for hidden layer and activating it
        hidden_sum,activated_hidden = self.activation(self.weights_ih,inputs,self.bias_hidden) 
        #generating sum for output layer and activating it
        output_sum,activated_output = self.activation(self.weights_ho,activated_hidden,self.bias_out)
        return hidden_sum,output_sum,activated_hidden,activated_output
    
    def backPropagation(self,inputs,targets,hidden,outputs,learning_rate):
        #Calculates the error for every result to every target
        out_error = [tval-tout for tval,tout in zip(targets,outputs)]
        #Calculates the errors to the hidden layer
        #The errors can be calculated using the transpose of the weights
        hidden_error = np.dot(np.transpose(self.weights_ho),out_error)
        
        #Delta Calculation for H-->O layers
        delta_ho,delta_bias_ho = self.deltaCalculation(learning_rate,outputs,out_error,hidden)
        #Adjust weights by its deltas
        self.weights_ho += delta_ho
        #Adjust bias by its deltas
        self.bias_out += delta_bias_ho

        #Delta Calculation for I-->H Layers
        delta_ih,delta_bias_ih = self.deltaCalculation(learning_rate,hidden,hidden_error,inputs)
        #Adjust weights by its deltas
        self.weights_ih += delta_ih
        #Adjust bias by its deltas
        self.bias_hidden += delta_bias_ih
        return hidden_error,out_error,delta_ih,delta_ho,delta_bias_ih,delta_bias_ho

    def deltaCalculation(self,learning_rate,activated_vector,error_vector,matrix_values):
        #Delta Calculation for 2 consecutive layers
        #The delta is the tweak values for the weights and the bias

        #           [scalar]------[elementwise]--[matrix operation dot]
        #DWeights = learning_rate*(error*gradient)*Matrix
        #DBias = learning:rate*(error*gradient)

        #Calculate gradient- the sigmoid_derivate of the sigmoid values
        gradient = [self.sigmoid(x,deriv=True)for x in activated_vector]
        #Elementwise multiplication between the gradient and the error
        delta = np.dot(error_vector,gradient)
        #Scalar multiplication of delta by the learning rate
        delta = np.dot(learning_rate,delta)
        #Bias calculation
        delta_bias = delta
        #delta . (matrix_values)T --matrix multiplication
        delta = np.dot(delta,np.transpose(matrix_values))
        return delta,delta_bias

    def train(self,inputs,targets,learning_rate):
        #Supervised learning
        #feeds forward the data to get a result from the neural net with sigmoid application
        _,_,hidden,outputs = self.feedForward(inputs) #list of values []
        _,out_error,_,_,_,_ = self.backPropagation(inputs,targets,hidden,outputs,learning_rate)
        return out_error
    
    def predict(self,entry):
        _,_,_,guess = self.feedForward(entry)
        return guess
