import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from tensorflow.examples.tutorials.mnist import input_data
import random

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

mnist = input_data.read_data_sets('../../MNIST_data', one_hot=False)


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


def sample_z(mu, log_var):
    eps = tf.random_normal(shape=tf.shape(mu))
    return mu + tf.exp(log_var / 2) * eps


# =============================== P(X|z) ======================================

P_W1 = tf.Variable(xavier_init([z_dim, h_dim]))
P_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

P_W2 = tf.Variable(xavier_init([h_dim, X_dim]))
P_b2 = tf.Variable(tf.zeros(shape=[X_dim]))


def P(z):
    h = tf.nn.relu(tf.matmul(z, P_W1) + P_b1)
    logits = tf.matmul(h, P_W2) + P_b2
    prob = tf.nn.sigmoid(logits)
    return prob, logits


# =============================== TRAINING ====================================

latent, z_logvar = Q(X)
#z_sample = sample_z(z_mu, z_logvar)
_, logits = P(latent)

# Sampling from random z
X_samples, _ = P(z)

# E[log P(X|z)]
#recon_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=X), 1)

recon_loss = tf.reduce_sum((logits-X)**2,1)

# D_KL(Q(z|X) || P(z)); calculate in closed form as both dist. are Gaussian
#kl_loss = 0.5 * tf.reduce_sum(tf.exp(z_logvar) + z_mu**2 - 1. - z_logvar, 1)
# VAE loss
mean_recon_loss = tf.reduce_mean(recon_loss)

score = recon_loss
vae_loss = tf.reduce_mean(recon_loss)

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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
n_clusters = 10

kmean = KMeans(n_clusters=n_clusters)

#--------

score_,r_loss, latent_= sess.run([score ,mean_recon_loss,latent], feed_dict={X: test_data})
#-----------draw------------
'''
from sklearn.manifold import TSNE
import matplotlib.pyplot as plot
X_embedded = TSNE(n_components=2).fit_transform(latent_)

plt.figure('Scatter fig')
ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
colors = ['b','g','r','orange']

for c in range(2):
    x_list = []
    y_list = []
    for i in range(len(test_data)):
        if train_label[i] == c:
            point = X_embedded[i]
            x_list.append(point[0])
            y_list.append(point[1])
    ax.scatter(x_list, y_list, c=colors[c], s=20, alpha=0.5)

plt.show()
'''
#---------------------
kmean.fit(latent_)
labels = kmean.labels_

print(labels)
record = [0 for _ in range(n_clusters)]
for l in labels:
    record[l] += 1
#print(record)
'''
max1 = max(record)
max1_idx = record.index(max1)
record[max1_idx] = 0
max2 = max(record)
max2_idx = record.index(max2)
'''
#print(max1_idx,max2_idx)

centers = kmean.cluster_centers_
#print(centers)
score_list = []

#centers = [centers[max1_idx],centers[max2_idx]]


for j in range(len(latent_)):
    a = latent_[j]
    sim = []
    for i, c in enumerate(centers):
        sim.append( (np.sum(a*c))/((np.sqrt(np.sum(a**2)))*(np.sqrt(np.sum(a**2)))) )
        #sim.append( np.sum((a-c)**2))
    #print(sim)
    score_list.append(-np.max(sim))
    #print(c)


print(loss)
print(test_label)
auc_score = roc_auc_score(test_label, score_list)
print(auc_score)