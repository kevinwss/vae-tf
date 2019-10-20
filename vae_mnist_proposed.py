import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)


dim = 28*28

Matrix = mnist.train.images


print("Matrix",Matrix.shape)
n_data = Matrix.shape[0]
n_anomaly = 550
seed = 1
anomaly_data = Matrix[n_data-n_anomaly:,:]

anomaly_data = np.concatenate((anomaly_data[:,dim//2:],anomaly_data[:,:dim//2]),axis = 1)
#anomaly_data = 2*np.random.rand(n_anomaly,dim) - 1

label = np.array([1]*n_data + [0]*n_anomaly) 
train_and_anomaly = np.concatenate((Matrix,anomaly_data),axis = 0)

random.seed(1)
random.shuffle(train_and_anomaly)
random.seed(1)
random.shuffle(label)

n_train = 50000
train_data = train_and_anomaly[:n_train,:]
train_label = label[:n_train]

test_data = train_and_anomaly[n_train:,:]
test_label = label[n_train:]

X_dim = dim
y_dim = 2
#---------------------------
mb_size = 64
z_dim = 100
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

X = tf.placeholder(tf.float32, shape=[None, X_dim])
z = tf.placeholder(tf.float32, shape=[None, z_dim])

Q_W1 = tf.Variable(xavier_init([X_dim, h_dim]))
Q_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

Q_W2_mu = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_mu = tf.Variable(tf.zeros(shape=[z_dim]))

Q_W2_sigma = tf.Variable(xavier_init([h_dim, z_dim]))
Q_b2_sigma = tf.Variable(tf.zeros(shape=[z_dim]))


def Q(X):
    h = tf.nn.relu(tf.matmul(X, Q_W1) + Q_b1)
    z_mu = tf.matmul(h, Q_W2_mu) + Q_b2_mu
    z_logvar = tf.matmul(h, Q_W2_sigma) + Q_b2_sigma
    return z_mu, z_logvar



#--------------------------
#try global
J = 50 # number of latent
#import tensorflow_probability as tfp
#tfd = tfp.distributions
#gamma = tf.Variable()
def get_sb(x):
    v = np.random.beta(a=1,b=2,size = (mb_size,J))
    pi = np.ones((mb_size,J))
    pi[:,0] = v[:,0]

    for k in range(1,J):
        for j in range(k):
            pi[:,k] = pi[:,k]*(1-v[:,j])
        pi[:,k] = pi[:,k] * v[:,k]
    
    theta = pi
    return theta  #(instance*J)


total_view = 2
def get_zsnd(z,theta): #z (batch,J,z_latent)

    s_nd = tf.multinomial(theta,total_view)
    #print("s_nd",s_nd.shape)
    z_snd = [[],[]]
    #z_snd2 = []
    for view in range(total_view):
        for i in range(mb_size):
            z_snd[view].append(tf.reshape(z[i,s_nd[i][view],:],[1,-1]))
    
    #print("z[i,s_nd[i][view],:]",tf.reshape(z[i,s_nd[i][view],:],[1,-1]))
    z_snd[0] = tf.concat(z_snd[0],0)
    z_snd[1] = tf.concat(z_snd[1],0)
    #z_snd[0] = np.array(z_snd[0])  #(batch,z_dim)
    #z_snd[1] = np.array(z_snd[1])
    print("z_snd[0]",z_snd[0].shape)
    return z_snd


def get_score(z,theta):
    H = 20# number of sampling
    size = theta.shape[0]
    score = [0 for _ in range(size)]
    
    print("size",size)

    for n_of_sample in range(H):#sampling
        s_nd = tf.multinomial(theta,total_view) #(batch,z_dim)
        #print("n_of_sample")
        for i in range(size):
            if s_nd[i][0] == s_nd[i][1]:
                score[i] += 1
    
    score= np.array(score)
    #score = tf.concat(score,0)
    print("score",score.shape)
    return score/H #(batch)
#----
P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim//2]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim//2]))


P_W12 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b12 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W22 = tf.Variable(xavier_init([h_dim, X_dim//2]))
P_b22 = tf.Variable(tf.zeros(shape=[X_dim//2]))
#----

def generate_x(z_snd):
    #----- generate x -----------
    #print("z_snd[0]",z_snd[0].shape)
    h1= tf.nn.relu(tf.matmul(z_snd[0], P_W1) + P_b1)
    logits1 = tf.matmul(h1, P_W2) + P_b2
    

    h2 = tf.nn.relu(tf.matmul(z_snd[1], P_W12) + P_b12)
    logits2 = tf.matmul(h2, P_W22) + P_b22



    return logits1,logits2


#--------------------------
def sample_z(mu, log_var):
    #size = mu.shape[0]
    #latent = mu.shape[1]
    #eps = tf.random_normal(shape=tf.shape(mu))
    #print(mu.shape)
    mu_s = tf.stack([tf.reshape(mu,[mb_size,z_dim]) for _ in range(J)],axis = 2)
    log_var_s = tf.stack([tf.reshape(log_var,[mb_size,z_dim]) for _ in range(J)],axis = 2)
    mu_s = tf.transpose(mu_s, perm=[0, 2, 1])
    log_var_s = tf.transpose(log_var_s, perm=[0, 2, 1])
    #print(mu_s.shape)
    eps = tf.random_normal(shape=tf.shape(mu_s))
    return mu_s + tf.exp(log_var_s / 2) * eps


# =============================== P(X|z) ======================================
'''
P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits

'''
# =============================== TRAINING ====================================
print("0")
z_mu, z_logvar = Q(X)

theta = get_sb(X)
#z_sample = get_z(X)
z_sample = sample_z(z_mu, z_logvar)
z_snd = get_zsnd(z_sample,theta)
logits1,logits2 = generate_x(z_snd)

print("1")

score = get_score(z_sample,theta)
print("2")
#_, logits = P(z_sample)

# Sampling from random z
#X_samples, _ = P(z)

# E[log P(X|z)]
#recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)

recon_loss1 = tf.reduce_sum((logits1-X[:,:X_dim//2])**2,1)

recon_loss2 = tf.reduce_sum((logits2-X[:,X_dim//2:])**2,1)

# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
mean_recon_loss = tf.reduce_mean(recon_loss1)

#score = recon_loss
vae_loss = tf.reduce_mean(recon_loss1 + recon_loss2  + kl_loss)

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

print("train")
#--------------------------
for it in range(1000):
    #X_mb, _ = mnist.train.next_batch(mb_size)
    X_mb = get_batch(train_data,it,mb_size)
    #print("X_mb",X_mb.shape,X_mb)
    #X_mb = get_batch(Matrix,it,mb_size)
    #print("it",it)
    _, loss,r_loss= sess.run([solver, vae_loss ,mean_recon_loss], feed_dict={X: X_mb})

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

score_= sess.run([score], feed_dict={X: test_data})

print(loss)
print(test_label)
auc_score = roc_auc_score(test_label, -score_)
print(auc_score)