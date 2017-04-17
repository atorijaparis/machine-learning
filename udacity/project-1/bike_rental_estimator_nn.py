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

        # input->hidden weights: Apply learning rate & delta weights / n_records
        self.weights_input_to_hidden  +=  self.lr * delta_weights_i_h / n_records

        # hidden->output weights: Apply learning rate & delta weights / n_records
        self.weights_hidden_to_output +=  self.lr * delta_weights_h_o / n_records

    def run(self, features):
        ''' Run a forward pass through the network with input features

            Arguments
            ---------
            features: 1D array of feature values
        '''
        # Compute input to hidden layer from:
        #  linear combination weight & features
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)

        # Compute output of hidden layer from:
        # sigmoid activation function
        hidden_outputs = self.activation_function(hidden_inputs)

        # Compute input to output(final) layer
        # from: linear combination hidden layer weights & hidden layer output
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output)

        # Compute output of output(final) layer from:
        # nothing since f(x) = x  just copy input
        final_outputs = final_inputs

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

import sys

### Set the hyperparameters here ###
iterations    = 20000 #100000
learning_rate = 0.03
hidden_nodes  = 25
output_nodes  = 1

N_i = train_features.shape[1]
network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    batch = np.random.choice(train_features.index, size=128)
    X, y = train_features.ix[batch].values, train_targets.ix[batch]['cnt']

    network.train(X, y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(iterations)) \
                     + "% ... Training loss: " + str(train_loss)[:5] \
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)

#plt.plot(losses['train'], label='Training loss')
#plt.plot(losses['validation'], label='Validation loss')
#plt.legend()
#_ = plt.ylim()

fig, ax = plt.subplots()
ax.plot(losses['train'], label='Training loss')
ax.plot(losses['validation'], label='Validation loss')
ax.grid(True)
_ = plt.ylim()
#ax.set_yscale('linear')
#ax.set_autoscaley_on(True)

# Now add the legend with some customizations.
legend = ax.legend(shadow=True)

# The frame is matplotlib.patches.Rectangle instance surrounding the legend.
frame = legend.get_frame()
frame.set_facecolor('0.90')

# Set the fontsize
for label in legend.get_texts():
    label.set_fontsize('large')

for label in legend.get_lines():
    label.set_linewidth(1.5)  # the legend line width

plt.show()


fig, ax = plt.subplots(figsize=(8,4))
mean, std = scaled_features['cnt']
predictions = network.run(test_features).T*std + mean
ax.plot(predictions[0], label='Prediction')
ax.plot((test_targets['cnt']*std + mean).values, label='Data')
ax.set_xlim(right=len(predictions))
ax.legend()

dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
dates = dates.apply(lambda d: d.strftime('%b %d'))
ax.set_xticks(np.arange(len(dates))[12::24])
_ = ax.set_xticklabels(dates[12::24], rotation=45)
plt.show()





