import numpy as np

class NeuralNetwork():
    def __init__(self,ninputs,nhidden,nout):
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

    def feedFoward(self,inputs):#Passes input foward in the network
        #generating sum for hidden layer and activating it
        hidden_sum,activated_hidden = self.activation(self.weights_ih,inputs,self.bias_hidden) 
        #generating sum for output layer and activating it
        output_sum,activated_output = self.activation(self.weights_ho,activated_hidden,self.bias_out)
        return hidden_sum,output_sum,activated_hidden,activated_output

    def backPropagation(self,inputs,targets,hidden,outputs,learning_rate):
        #Claculates the error for every result to every target
        out_error = [tval-tout for tval,tout in zip(targets,outputs)]
        #Calculates the errors to the hidden layer
        #The erros can be calculated using the transpose of the weights
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

    #Delta Calculation for 2 consecutive layers
    def deltaCalculation(self,learning_rate,activated_vector,error_vector,matrix_values):
        #Calculate gradient
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
        #feedsfoward the data to get a result from the neural net with sigmoid aplication
        _,_,hidden,outputs = self.feedFoward(inputs) #list of values []
        _,out_error,_,_,_,_ = self.backPropagation(inputs,targets,hidden,outputs,learning_rate)
        return out_error