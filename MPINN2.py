
import plotly.express as px
import time
import math
import tensorflow as tf ##1.15
import pandas as pd

import csv
import numpy as np
np.random.seed(1235)
tf.set_random_seed(1111)


# Define the filename for boundary

# filename = "boundary_collocation_points_complex_tms_2.csv"  #(boundary x,y,z coordinates) for complex tms 2
filename = "boundary_collocation_points_complex_tms_3.csv"  #(boundary x,y,z coordinates) for complex tms 3

# Initialize an empty list to store the data
data = []

# Read the CSV file
with open(filename, "r", newline="", encoding="utf-8") as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)


    # Read the data row by row and append it to the 'data' list
    for row in csvreader:
        data.append(row)

# Convert 'data' to a NumPy array
Uu = np.array(data, dtype=np.float32)
X_u_train=Uu[:,1:3]
u_train=Uu[:,3:4]
# print(X_u_train)
lbx=np.min(X_u_train)
lby=np.min(X_u_train)

ubx=np.max(X_u_train)
uby=np.max(X_u_train)

# Define the filename for domain collocation points

# filename = "domain_collocation_points_complex_tms_2.csv"  ###Domain collocation points (x,y coordinates) for complex tms 2
filename = "domain_collocation_points_complex_tms_3.csv"  ###Domain collocation points (x,y coordinates) for complex tms 3

# Initialize an empty list to store the data
data = []

# Read the CSV file
with open(filename, "r", newline="", encoding="utf-8") as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)

    # Read the data row by row and append it to the 'data' list
    for row in csvreader:
        data.append(row)

# Convert 'data' to a NumPy array
Uu = np.array(data, dtype=np.float32)
X_f_train=Uu[:,1:3]
Nf=len(X_f_train)

lb=np.array([lbx, lby])
ub=np.array([ubx ,uby])

##prestress values
c1=3.0
c2=3.0
import pandas as pd
global Loss

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub

        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]

        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]

        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.math.reduce_std(tf.square(self.u_tf - self.u_pred)) + \
                    (self.f_pred) + tf.reduce_mean(tf.square(self.u_tf - self.u_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='BFGS',
                                                                options={'maxiter': 1500,
                                                                         'gtol': 1e-15})

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev)*3, dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.nn.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):

        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_tt = tf.gradients(u_t, t)[0]
        f = c1 * u_xx + c2 * u_tt
        f0 = tf.math.reduce_std((f)) + tf.reduce_mean(tf.square(f))
        return f0/2

    def callback(self, loss):
        global Loss
        Loss=loss
        print('Loss:', loss)

    def train(self):

        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star


if __name__ == "__main__":

    h = 20
    layers = [2, 20, 20, 1]
    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)

    start_time = time.time()
    model.train()
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    u_pred, f_pred = model.predict(X_f_train)


# For dowloading the form found shape
import pandas as pd
u_pred, f_pred = model.predict(np.vstack((X_u_train,X_f_train)))

d = np.hstack((np.vstack((X_u_train,X_f_train)), u_pred))

##Plot form found shape
fig = px.scatter_3d(d, x=0, y=1, z=2)
fig.update_traces(marker_size=1)
fig.show()
