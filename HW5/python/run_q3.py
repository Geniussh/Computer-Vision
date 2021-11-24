import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import copy
from nn import *

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
train_size, valid_size, test_size = len(train_x), len(valid_x), len(test_x)

max_iters = 50
# pick a batch size, learning rate
batch_size = 54
learning_rate = 8e-3
hidden_size = 64

batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

params = {}

# initialize layers here
initialize_weights(1024, hidden_size, params, 'layer1')
initialize_weights(hidden_size, 36, params, 'output')

Wlayer1_orig = copy.deepcopy(params['Wlayer1'])

# with default settings, you should get loss < 150 and accuracy > 80%
train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb, params, 'layer1', sigmoid)
        probs = forward(h1, params, 'output', softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss += loss
        total_acc += acc

        # backward
        delta1 = probs - yb
        delta2 = backwards(delta1, params, 'output', linear_deriv)
        backwards(delta2, params, 'layer1', sigmoid_deriv)

        # apply gradient
        params['Woutput'] -= learning_rate * params['grad_W'+'output']
        params['boutput'] -= learning_rate * params['grad_b'+'output']
        params['Wlayer1'] -= learning_rate * params['grad_W'+'layer1']
        params['blayer1'] -= learning_rate * params['grad_b'+'layer1']
    
    total_loss /= train_size
    total_acc /= batch_num
    train_loss.append(total_loss)
    train_acc.append(total_acc)

    # Validation
    valid_h1 = forward(valid_x, params, 'layer1')
    valid_probs = forward(valid_h1, params, 'output', softmax)
    val_loss, val_acc = compute_loss_and_acc(valid_y, valid_probs)
    val_loss /= valid_size
    valid_loss.append(val_loss)
    valid_acc.append(val_acc)

    if itr % 2 == 1:
        print("Itr: {:02d}".format(itr)) 
        print("Train loss: {:.2f} \t acc : {:.2f}".format(total_loss,total_acc))
        print("Valid loss: {:.2f} \t acc : {:.2f} \n".format(val_loss,val_acc))

# Plot
fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.plot(range(max_iters), train_acc, label='Training Accuracy')
ax0.plot(range(max_iters), valid_acc, label='Validation Accuracy')
ax0.set_title('Training Accuracy Curve')
ax0.set_xlabel('Epoch')
ax0.set_ylabel('Accuracy')
ax0.legend()
ax1.plot(range(max_iters), train_loss, label='Training Loss')
ax1.plot(range(max_iters), valid_loss, label='Validation Loss')
ax1.set_title('Training Loss Curve')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Cross-Entropy Loss')
ax1.legend()
plt.show()

import pickle
saved_params = {k:v for k,v in params.items() if '_' not in k}
with open('q3_weights.pickle', 'wb') as handle:
    pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.3
from mpl_toolkits.axes_grid1 import ImageGrid

# visualize weights here
fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8,8), axes_pad=0.1)
for ax, weight in zip(grid, Wlayer1_orig.T):
    ax.imshow(weight.reshape(32,32))
    ax.axis('off')
fig.suptitle('Weights at Initialization')
plt.show()

fig = plt.figure()
grid = ImageGrid(fig, 111, nrows_ncols=(8,8), axes_pad=0.1)
for ax, weight in zip(grid, params['Wlayer1'].T):
    ax.imshow(weight.reshape(32,32))
    ax.axis('off')
fig.suptitle('Weights Learned')
plt.show()

# Q3.4
test_h1 = forward(test_x, params, 'layer1')
test_probs = forward(test_h1, params, 'output', softmax)
_, test_acc = compute_loss_and_acc(test_y, test_probs)
print("Test accuracy: {:.2f}".format(test_acc))

# compute comfusion matrix here
confusion_matrix = np.zeros((test_y.shape[1], test_y.shape[1]))
test_preds = np.argmax(test_probs, axis=1)
for i in range(test_size):
    confusion_matrix[np.argmax(test_y[i])][test_preds[i]] += 1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.xlabel('Prediction')
plt.ylabel('Ground Truth')
plt.show()