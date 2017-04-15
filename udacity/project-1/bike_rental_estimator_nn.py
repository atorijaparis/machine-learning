#
# PROJECT-1: Neural network to estimate bike sharing business sales
# Alex Torija-Paris: train & run methods
#
# A single file (no jupyter notebook needed) to understand, tweak and solve the project.
#

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#
#
#        DATA PREP STEP
#
#

#
# Read Data from .csv file
#
data_path = 'Bike-Sharing-Dataset/hour.csv'
rides     = pd.read_csv(data_path)
rides.head()

#
# Create Dummy Variables
#
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides   = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()

#
# Scaling target variables
#
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

#
# Split the data into training, testing, and validation sets
#
# Save data for approximately the last 21 days
test_data = data[-21*24:]

# Now remove the test data from the data set
data = data[:-21*24]

# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

# Hold out the last 60 days or so of the remaining data as a validation set
train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]

#
#
#        NEURAL NETWORK IMPLEMENTATION
#
#

def MSE(y, Y):
    return np.mean((y-Y)**2)


class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes  = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes ** -0.5,
                                                        (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes ** -0.5,
                                                         (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation here

        self.activation_function = sigmoid

    def trainOK(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            #### Implement the forward pass here ####
            ### Forward pass ###
            # TODO: Hidden layer - Replace these values with your calculations.
            hidden_inputs  = np.dot(X, self.weights_input_to_hidden)  # signals into hidden layer
            hidden_outputs = self.activation_function(hidden_inputs)  # signals from hidden layer

            # TODO: Output layer - Replace these values with your calculations.
            final_inputs  = np.dot(hidden_outputs, self.weights_hidden_to_output)  # signals into final output layer
            final_outputs = final_inputs #self.activation_function(final_inputs)  # signals from final output layer
            # final_outputs = final_inputs
            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = y - final_outputs  # Output layer error is the difference between desired target and actual output.

            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(self.weights_hidden_to_output, error)

            # TODO: Backpropagated error terms - Replace these values with your calculations.
            output_error_term = error * 1
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

            # Weight step (input to hidden)
            delta_weights_i_h += hidden_error_term * X[:, None]
            # Weight step (hidden to output)
            delta_weights_h_o += output_error_term * hidden_outputs[:, None]

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_hidden_to_output += self.lr * delta_weights_h_o  # update hidden-to-output weights with gradient descent step
        self.weights_input_to_hidden  += self.lr * delta_weights_i_h  # update input-to-hidden weights with gradient descent step


    def train(self, features, targets):
        ''' Train the network on batch of features and targets.

            Arguments
            ---------

            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values

        '''
        n_records = features.shape[0]

        # Initialize weight's deltas to 0
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):

            # Compute input to hidden layer from:
            #  linear combination weight & features
            hidden_inputs = np.dot(X, self.weights_input_to_hidden)

            # Compute output of hidden layer from:
            # sigmoid activation function
            hidden_outputs = self.activation_function(hidden_inputs)

            # Compute input to output(final) layer
            # from: linear combination hidden layer weights & hidden layer output
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

            # Compute output of output(final) layer from:
            # nothing since f(x) = x  just copy input
            final_outputs = final_inputs

            # Compute output(final) error: desired result - actual result
            error = y - final_outputs
            # Compute Hidden layer error: output(final) * weighted by the hidden to output weights
            hidden_error = np.dot(self.weights_hidden_to_output,error)

            # Compute output error term = 1 (derivative of activation function f(x) = x)
            output_error_term = error * 1

            # Compute hidden error term = derivative of sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))
            # sigmoid(x) = hidden_outputs
            hidden_error_term = hidden_error * hidden_outputs * (1 - hidden_outputs)

            # Update delta weights input->hidden
            delta_weights_i_h +=  hidden_error_term * X[:,None]

            # Update delta weights hidden->output
            delta_weights_h_o +=  output_error_term * hidden_outputs[:,None]

        #
        self.weights_input_to_hidden  +=  self.lr * delta_weights_i_h / n_records
        #
        self.weights_hidden_to_output +=  self.lr * delta_weights_h_o / n_records


    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''

        print('features:\n',features) # 1x3
        print('weights_input_to_hidden:\n', self.weights_input_to_hidden) # 3x2
        #
        # HIDDEN LAYER
        #
        hidden_inputs = np.dot(features, self.weights_input_to_hidden) # 1x3 3x2
        print('hidden_inputs:\n', hidden_inputs) # 1x2

        # Compute result of activation function
        hidden_outputs = self.activation_function(hidden_inputs) # 1x2
        print('hidden_outputs:\n', hidden_outputs) # 1x2

        #
        # OUTPUT LAYER
        #
        print('weights_hidden_to_output:\n', self.weights_hidden_to_output) # 2x1
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # 1x2 2x1
        print('final_inputs:\n', final_inputs) # 1x1

        # Activation function for output layer is f(x) = x i.e no change - Just pass final_inputs to output
        final_outputs = final_inputs # 1x1
        print('final_outputs:\n', final_outputs) # 1x1

        return final_outputs


import unittest

inputs     = np.array([[0.5, -0.2, 0.1]])
targets    = np.array([[0.4]])

test_w_i_h = np.array([[0.1, -0.2],
                       [0.4, 0.5],
                       [-0.3, 0.2]])
test_w_h_o = np.array([[0.3],
                       [-0.1]])

class TestMethods(unittest.TestCase):
    ##########
    # Unit tests for data loading
    ##########

    def test_data_path(self):
        print('Test: that file path to dataset has been unaltered')
        self.assertTrue(data_path.lower() == 'bike-sharing-dataset/hour.csv')

    def test_data_loaded(self):
        print('Test: that data frame loaded')
        self.assertTrue(isinstance(rides, pd.DataFrame))

    def test_activation(self):
        print('Test: Activation')
        network = NeuralNetwork(3, 2, 1, 0.5)
        # Test that the activation function is a sigmoid
        self.assertTrue(np.all(network.activation_function(0.5) == 1 / (1 + np.exp(-0.5))))

    def test_train(self):
        print('Test: Training')
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))


suite = unittest.TestLoader().loadTestsFromModule(TestMethods())
unittest.TextTestRunner().run(suite)

'''
    ##########
    # Unit tests for network functionality
    ##########
    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))

    def test_run(self):
        # Test correctness of run method
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden  = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        self.assertTrue(np.allclose(network.run(inputs), 0.09998924))


    def test_train(self):
        # Test that weights are updated correctly on training
        network = NeuralNetwork(3, 2, 1, 0.5)
        network.weights_input_to_hidden = test_w_i_h.copy()
        network.weights_hidden_to_output = test_w_h_o.copy()

        network.train(inputs, targets)
        self.assertTrue(np.allclose(network.weights_hidden_to_output,
                                    np.array([[0.37275328],
                                              [-0.03172939]])))
        self.assertTrue(np.allclose(network.weights_input_to_hidden,
                                    np.array([[0.10562014, -0.20185996],
                                              [0.39775194, 0.50074398],
                                              [-0.29887597, 0.19962801]])))


def trainXYZ(self, features, targets):
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)

        for X, y in zip(features, targets):
            # FORWARD PASS
            print('features:\n', features)  # 1x3
            print('weights_input_to_hidden:\n', self.weights_input_to_hidden)  # 3x2
            #
            # HIDDEN LAYER
            #
            hidden_inputs = np.dot(features, self.weights_input_to_hidden)  # 1x3 3x2
            print('hidden_inputs:\n', hidden_inputs)  # 1x2

            # Compute result of activation function
            hidden_outputs = self.activation_function(hidden_inputs)  # 1x2
            print('hidden_outputs:\n', hidden_outputs)  # 1x2

            #
            # OUTPUT LAYER
            #
            print('weights_hidden_to_output:\n', self.weights_hidden_to_output)  # 2x1
            final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)  # 1x2 2x1
            print('final_inputs:\n', final_inputs)  # 1x1

            # Activation function for output layer is f(x) = x i.e no change - Just pass final_inputs to output
            final_outputs = final_inputs  # 1x1
            print('final_outputs:\n', final_outputs)  # 1x1

            #### Implement the backward pass here ####
            ### Backward pass ###

            # TODO: Output error - Replace this value with your calculations.
            error = targets - final_outputs
            print('error:\n', error) # 1x1
            print('targets:\n', targets) # 1x1
            output_error_term = error * final_outputs * ( 1 - final_outputs)
            print('output_error_term:\n', output_error_term)  # 1x1

            # TODO: Calculate the hidden layer's contribution to the error
            hidden_error = np.dot(output_error_term,self.weights_hidden_to_output.T) # 1x1 2x1.T = 1x1 1x2 = 1x2
            print('hidden_error:\n', hidden_error)  # 1x2
            hidden_grad = hidden_outputs * (1 - hidden_outputs)
            hidden_error_term = hidden_error * hidden_grad
            print('hidden_error_term:\n', hidden_error_term) # 1x2   ### Why 2x2 ###

            # TODO: Backpropagated error terms - Replace these values with your calculations.
            # Weight step (input to hidden)
            print('self.lr:\n', self.lr)
            print('delta_weights_i_h:\n', delta_weights_i_h)
            delta_weights_i_h +=  self.lr * np.dot(hidden_error_term,features.T) # 1 1x2.T 1x3

            # Weight step (hidden to output)
            print('delta_weights_h_o:',delta_weights_h_o.shape,
                  'output_error_term:',output_error_term.shape,
                  "hidden_outputs:",hidden_outputs.shape)
            delta_weights_h_o +=  self.lr * hidden_outputs * output_error_term

        # TODO: Update the weights - Replace these values with your calculations.
        self.weights_input_to_hidden  +=  delta_weights_i_h
        self.weights_hidden_to_output +=  delta_weights_h_o


'''









