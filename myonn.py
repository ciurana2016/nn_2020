'''
From> Make your own neural network
'''
import numpy
import scipy.special



class neuralNetwork():

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # Set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Link weight matrices, input to hidden, and hidden to out
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Learning rage
        self.lr = learningrate

        # Sigmoid activation
        self.activation_function = lambda x:scipy.special.expit(x)

        pass

    def train(self, input_list, target_list):
        # TODO > input_list comes in 2 functionsand we repeat code, do a decorator
        # Convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(target_list, ndmin=2).T

        # TODO >  Also we repeat more code..
        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # Error is the (target - actual)
        output_errors = tartets - final_outputs
        # Gidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)

        # Update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), \
            numpy.transpose(hidden_outputs))

        # Update the weights for the links between the input and output hidden
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), \
            numpy.transpose(inputs))


        pass

    # Make a question to the nn
    def query(self, input_list):
        # Convert input list to 2d array
        inputs = numpy.array(input_list, ndmin=2).T

        # Calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # Calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        # Calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # Calculate signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        return final_outputs



if __name__ == '__main__':
    # Creating a neural network
    input_nodes = 3
    hidden_nodes = 3
    output_nodes = 3
    learning_rate = 0.3

    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print(n.query([1.0, 0.5, -1.5]))
