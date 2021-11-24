import numpy as np
import scipy.io
from nn import *
from collections import Counter
import matplotlib.pyplot as plt
import pickle

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')

# we don't need labels now!
train_x = train_data['train_data']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
train_size, valid_size = len(train_x), len(valid_x)

max_iters = 100
# pick a batch size, learning rate
batch_size = 36 
learning_rate = 3e-5
hidden_size = 32
lr_rate = 20
batches = get_random_batches(train_x,np.ones((train_x.shape[0],1)),batch_size)
batch_num = len(batches)

params = Counter()

def step(layer):
    # momentum update
    params['m_W'+layer] = 0.9 * params['m_W'+layer] - learning_rate * params['grad_W'+layer]
    params['m_b'+layer] = 0.9 * params['m_b'+layer] - learning_rate * params['grad_b'+layer]
    params['W'+layer] += params['m_W'+layer]
    params['b'+layer] += params['m_b'+layer]

# Q5.1 & Q5.2
# initialize layers here
initialize_weights(1024, hidden_size, params, name='encoder1')
initialize_weights(hidden_size, hidden_size, params, name='encoder2')
initialize_weights(hidden_size, hidden_size, params, name='decoder1')
initialize_weights(hidden_size, 1024, params, name='decoder2')

# should look like your previous training loops
train_loss = []
for itr in range(max_iters):
    total_loss = 0
    for xb,_ in batches:
        # training loop can be exactly the same as q2!
        # your loss is now squared error
        # delta is the d/dx of (x-y)^2
        # to implement momentum
        #   just use 'm_'+name variables
        #   to keep a saved value over timestamps
        #   params is a Counter(), which returns a 0 if an element is missing
        #   so you should be able to write your loop without any special conditions
        encoder1 = forward(xb, params, 'encoder1', relu)
        encoder2 = forward(encoder1, params, 'encoder2', relu)
        decoder1 = forward(encoder2, params, 'decoder1', relu)
        decoder2 = forward(decoder1, params, 'decoder2', sigmoid)

        total_loss += np.sum((xb - decoder2) ** 2)  # squared error

        delta = 2 * (decoder2 - xb)
        delta = backwards(delta, params, 'decoder2', sigmoid_deriv)
        delta = backwards(delta, params, 'decoder1', relu_deriv)
        delta = backwards(delta, params, 'encoder2', relu_deriv)
        backwards(delta, params, 'encoder1', relu_deriv)

        step('encoder1')
        step('encoder2')
        step('decoder1')
        step('decoder2')
    
    total_loss /= train_size
    train_loss.append(total_loss)

    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f}".format(itr,total_loss))
    if itr % lr_rate == lr_rate-1:
        learning_rate *= 0.9
        
plt.plot(range(max_iters), train_loss, label='Training Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q5_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q5.3.1
# visualize some results
params = pickle.load(open('q5_weights.pickle','rb'))

valid_y = np.argmax(valid_y, axis=1)
selected = [0, 7, 23, 26, 35]  # A, H, X, 0, 9
for s in selected:
    orig = valid_x[np.where(valid_y == s)[0][:2], :]
    fig, ax = plt.subplots(2, 2)
    for i, img in enumerate(orig):
        encoder1 = forward(img, params, 'encoder1', relu)
        encoder2 = forward(encoder1, params, 'encoder2', relu)
        decoder1 = forward(encoder2, params, 'decoder1', relu)
        out = forward(decoder1, params, 'decoder2', sigmoid)
        ax[i, 0].imshow(img.reshape(32,32).T, cmap='gray')
        ax[i, 0].set_title('Original')
        ax[i, 0].axis('off')
        ax[i, 1].imshow(out.reshape(32,32).T, cmap='gray')
        ax[i, 1].set_title('Reconstruction')
        ax[i, 1].axis('off')
    plt.show()

# Q5.3.2
from skimage.metrics import peak_signal_noise_ratio
# evaluate PSNR
psnr = 0
for x in valid_x:
    encoder1 = forward(x, params, 'encoder1', relu)
    encoder2 = forward(encoder1, params, 'encoder2', relu)
    decoder1 = forward(encoder2, params, 'decoder1', relu)
    out = forward(decoder1, params, 'decoder2', sigmoid)
    psnr += peak_signal_noise_ratio(x, out)
psnr /= valid_size
print("PSNR: {:.2f}".format(psnr))