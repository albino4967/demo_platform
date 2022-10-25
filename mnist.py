from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from tensorflow.keras.datasets import mnist
import argparse
from PIL import Image
import random

tf.compat.v1.enable_eager_execution()

def data_load() :
    num_features = 784  # data features (img shape: 28*28).
    batch_size = 256
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # Convert to float32.
    x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
    # Flatten images to 1-D vector of 784 features (28*28).
    x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
    # Normalize images value from [0, 255] to [0, 1].
    x_train, x_test = x_train / 255., x_test / 255.

    # Use tf.data API to shuffle and batch data.
    train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

    return train_data, x_test, y_test


# Create TF Model.
class NeuralNet(Model):
    # Set layers.
    def __init__(self, n_hidden_1, n_hidden_2, dropout):
        super(NeuralNet, self).__init__()
        # First fully-connected hidden layer.
        self.fc1 = layers.Dense(n_hidden_1, activation=tf.nn.relu)
        # First fully-connected hidden layer.
        self.fc2 = layers.Dense(n_hidden_2, activation=tf.nn.relu)
        self.dropout = layers.Dropout(dropout)
        # Second fully-connecter hidden layer.
        self.out = layers.Dense(10)

    # Set forward pass.
    def call(self, x, is_training=False):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.out(x)
        if not is_training:
            # tf cross entropy expect logits without softmax, so only
            # apply softmax when not training.
            x = tf.nn.softmax(x)
        return x


# Cross-Entropy Loss.
# Note that this will apply 'softmax' to the logits.
def cross_entropy_loss(x, y):
    # Convert labels to int 64 for tf cross-entropy function.
    y = tf.cast(y, tf.int64)
    # Apply softmax to logits and compute cross-entropy.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=x)
    # Average loss across the batch.
    return tf.reduce_mean(loss)


# Accuracy metric.
def accuracy(y_pred, y_true):
    # Predicted class is the index of highest score in prediction vector (i.e. argmax).
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

# Optimization process.
def run_optimization(neural_net, optimizer, x, y):
    # Wrap computation inside a GradientTape for automatic differentiation.
    with tf.GradientTape() as g:
        # Forward pass.
        pred = neural_net(x, is_training=True)
        # Compute loss.
        loss = cross_entropy_loss(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = neural_net.trainable_variables
    # Compute gradients.
    gradients = g.gradient(loss, trainable_variables)
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

def test_plot_image(model) :
    test_num = random.randint(0,9)

    data_path = f'{os.getcwd()}/test_data'
    test_list = os.listdir(data_path)
    predic_list = []
    image_list =[]
    for i, item in enumerate(test_list) :
        show_img = Image.open(f'{data_path}/{item}').convert("RGB")
        image_list.append(show_img)

        img = Image.open(f'{data_path}/{item}').convert("L")
        img = np.array(img)
        img = img.reshape([-1, 784])
        pred = model(img, is_training=False)
        predic_list.append(tf.argmax(pred, 1))


    print(predic_list[test_num][0].numpy())

    plt.imshow(image_list[test_num])
    plt.savefig(f'{predic_list[test_num][0].numpy()}.jpg')


def main(n_hidden_1, n_hidden_2, dropout, learning_rate, training_steps):
    display_step = 100
    # Build neural network model.
    neural_net = NeuralNet(n_hidden_1, n_hidden_2, dropout)
    train_data, x_test, y_test = data_load()
    # Stochastic gradient descent optimizer.
    optimizer = tf.keras.optimizers.SGD(learning_rate)
    # Run training for the given number of steps.
    for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):
        # Run the optimization to update W and b values.
        run_optimization(neural_net, optimizer, batch_x, batch_y)

        if step % display_step == 0:
            pred = neural_net(batch_x, is_training=True)
            loss = cross_entropy_loss(pred, batch_y)
            acc = accuracy(pred, batch_y)
            print("train-step: %i, train-loss: %f, train-accuracy: %f" % (step, loss, acc))

    # Test model on validation set.
    pred = neural_net(x_test, is_training=False)
    print("Test-accuracy: %f" % accuracy(pred, y_test))

    test_plot_image(neural_net)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_hidden_layer_1', type=int, default =128, help='input int')
    parser.add_argument('--num_hidden_layer_2', type=int, default =256, help='input int')
    parser.add_argument('--dropout', type=float, default =1, help='input 0~1 float')
    parser.add_argument('--learning_rate', type=float, default =0.01, help='recommended 0.01')
    parser.add_argument('--epoch', type=int, default =2000, help='recommended 0.9')
    args = parser.parse_args()

    n_hidden_1, n_hidden_2, dropout, learning_rate, training_steps = args.num_hidden_layer_1, args.num_hidden_layer_2, args.dropout, args.learning_rate, args.epoch
    main(n_hidden_1, n_hidden_2, dropout, learning_rate, training_steps)
