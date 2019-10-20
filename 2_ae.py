import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

dataset = "breast_cancer"

dim = 10
f = open(dataset)
lines = f.readlines()

Matrix = []
this_line = []

for line in lines:
    line = line.split(" ")
    this_line = []
    for i in range(1,dim+1):
        data = (line[i].split(":"))[1]
        this_line.append(float(data))
    Matrix.append(this_line)

Matrix = np.array(Matrix)


print("Matrix",Matrix.shape)
n_data = Matrix.shape[0]
n_anomaly = 60
seed = 1
anomaly_data = Matrix[n_data-n_anomaly:,:]

anomaly_data = np.concatenate((anomaly_data[:,5:],anomaly_data[:,:5]),axis = 1)

label = np.array([1]*n_data + [0]*n_anomaly) 
train_and_anomaly = np.concatenate((Matrix,anomaly_data),axis = 0)

random.seed(1)
random.shuffle(train_and_anomaly)
random.shuffle(label)

n_train = 500
train_data = train_and_anomaly[:n_train,:]
train_label = label[:n_train]

test_data = train_and_anomaly[n_train:,:]
test_label = label[n_train:]

X_dim = dim
y_dim = 2
#---------------------------
mb_size = 32
z_dim = 3
#X_dim = mnist.train.images.shape[1]
#y_dim = mnist.train.labels.shape[1]
h_dim = 128
c = 0
lr = 1e-3


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


# =============================== Q(z|X) ======================================

X = tf.placeholder(tf.float32, shape=[None, X_dim//2])
X2 = tf.placeholder(tf.float32, shape=[None, X_dim//2])

z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim//2, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

#Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
#Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    #z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    #return z_mu, z_logvar
    return z_mu
#-------------

Q_W12 = tf.Variable(xavier_init([X_dim//2, h_dim]))
Q_b12 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu2 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu2 = tf.Variable(tf.zeros(shape=[z_dim]))

#Q_W2_sigma2 = tf.Variable(xavier_init([h_dim, z_dim]))
#Q_b2_sigma2 = tf.Variable(tf.zeros(shape=[z_dim]))

def Q2(X):
    h = tf.nn.relu(tf.matmul(X, Q_W12) + Q_b12)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    #z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu


#------------- total encoder-----------
Q_W13 = tf.Variable(xavier_init([h_dim*2, h_dim]))
Q_b13 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu3 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu3 = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma3 = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma3 = tf.Variable(tf.zeros(shape=[z_dim]))

def Q_total(h1,h2):
    h_total = tf.concat([h1,h2], axis = 1)
    h = tf.nn.relu(tf.matmul(h_total, Q_W13) + Q_b13)
    z_mu = tf.matmul(h, Q_W2_mu3) + Q_b2_mu3
    z_logvar = tf.matmul(h, Q_W2_sigma3) + Q_b2_sigma3
    return z_mu, z_logvar

#--------------



def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim//2]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim//2]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

#-------------------

P_W12 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b12 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W22 = tf.Variable(xavier_init([h_dim, X_dim//2]))
P_b22 = tf.Variable(tf.zeros(shape=[X_dim//2]))


def P2(z):
    h = tf.nn.relu(tf.matmul(z, P_W12) + P_b12)
    logits = tf.matmul(h, P_W22) + P_b22
    prob = tf.nn.sigmoid(logits)
    return prob, logits
# =============================== TRAINING ====================================

h1 = Q(X)

h2 = Q2(X2)

#z_mu, z_logvar = Q_total(h1,h2)

#z_sample = sample_z(z_mu, z_logvar)
#z_sample2 = sample_z(z_mu2,z_logvar2)

_, logits = P(h1)

_, logits2 = P2(h2)

# Sampling from random z
#X_samples, _ = P(z)

# E[log P(X|z)]
#recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)

recon_loss = tf.reduce_sum((logits-X)**2,1)

recon_loss2 = tf.reduce_sum((logits2-X2)**2,1)

# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
#kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
mean_recon_loss = tf.reduce_mean(recon_loss+recon_loss2)

score = 0.5*(recon_loss + recon_loss2)
vae_loss = tf.reduce_mean(0.5*(recon_loss + recon_loss2) )

solver = tf.train.AdamOptimizer().minimize(vae_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

#--------------------------
def get_batch(data,it,mb_size):
    n = len(data)
    n_of_batch = int(n//mb_size)
    it = it%n_of_batch
    #print(it)
    #print(mb_size)
    batch = data[it*mb_size:(it+1)*mb_size,:]
    return batch


#--------------------------
for it in range(5000):
    #X_mb, _ = mnist.train.next_batch(mb_size)
    X_mb = get_batch(train_data,it,mb_size)
    #print("X_mb",X_mb.shape,X_mb)
    #X_mb = get_batch(Matrix,it,mb_size)

    _, loss,r_loss= sess.run([solver, vae_loss ,mean_recon_loss], feed_dict={X: X_mb[:,:X_dim//2], X2: X_mb[:,X_dim//2:]})

    if it % 500 == 0:
        print('Iter: {}'.format(it))
        print('Loss: {:.4}'. format(loss))
        
        print('recon_loss: {:.4}'. format(r_loss))
        print()


        '''
        samples = sess.run(X_samples, feed_dict={z: np.random.randn(16, z_dim)})
        
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)
        '''

#------------test----------------
from sklearn.metrics import roc_auc_score

score_,r_loss= sess.run([score ,mean_recon_loss], feed_dict={X: test_data[:,:X_dim//2], X2: test_data[:,X_dim//2:]})

print(loss)
print(test_label)
auc_score = roc_auc_score(test_label, -score_)
print(auc_score)