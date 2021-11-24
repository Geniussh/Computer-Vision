import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import torchvision
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.autograd import Variable

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']
train_size, valid_size, test_size = len(train_x), len(valid_x), len(test_x)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class NIST36Dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x).float().cuda()
        self.y = torch.from_numpy(np.argmax(y, axis=1)).cuda()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

''' 
=====================
Q6.1.1
=====================
'''
class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(TwoLayerNet, self).__init__()
        self.layer1 = torch.nn.Linear(D_in, H)
        self.layer2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h = self.layer1(x)
        h_sigmoid = torch.sigmoid(h)
        y_pred = self.layer2(h_sigmoid)
        return y_pred

print("\nStart training FC on NIST36\n")
train = NIST36Dataset(train_x, train_y)
valid = NIST36Dataset(valid_x, valid_y)
train_size, valid_size = len(train_x), len(valid_x)
train_loader = DataLoader(train, batch_size=54, shuffle=True)
valid_loader = DataLoader(valid, batch_size=valid_size, shuffle=False)
model = TwoLayerNet(1024, 64, 36)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
max_iters = 50
for itr in range(max_iters):
    model.train()
    total_loss = 0
    total_acc = 0
    for xb, yb in train_loader:
        y_pred = model(xb)
        loss = criterion(y_pred, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * xb.size(0)
        _, preds = torch.max(y_pred.data, 1)
        total_acc += torch.sum(preds == yb.data)

    train_loss.append(total_loss / len(train_loader.dataset))
    train_acc.append(total_acc / len(train_loader.dataset)) 

    # Validation
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in valid_loader:
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
    valid_loss.append(total_loss / len(valid_loader.dataset))
    valid_acc.append(total_acc / len(valid_loader.dataset))

    if itr % 2 == 1:
        print("Itr: {:02d}".format(itr)) 
        print("Train loss: {:.2f} \t acc : {:.2f}".format(train_loss[-1],train_acc[-1]))
        print("Valid loss: {:.2f} \t acc : {:.2f} \n".format(valid_loss[-1],valid_acc[-1]))
    
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
plt.savefig('q6_1_1.png', bbox_inches='tight')
''' 
=====================
End of Q6.1.1
=====================
'''

''' 
=====================
Q6.1.2
=====================
'''
class Unit(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(Unit,self).__init__()
        self.conv = torch.nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size,
                                out_channels=out_channels, padding=padding)
        self.bn = torch.nn.BatchNorm2d(num_features=out_channels)
        self.relu = torch.nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)
        return output

class ConvNet(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=36, isNIST36=True):
        super(ConvNet, self).__init__()

        self.unit1 = Unit(in_channels, 16, 5, 3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3)
        self.unit2 = Unit(16, 32, 2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.unit3 = Unit(32, 16, 2)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)
        
        #Add all the units into the Sequential layer in exact order
        self.net = torch.nn.Sequential(self.unit1, self.pool1, self.unit2, self.pool2, 
                                       self.unit3, self.pool3)

        self.fc = torch.nn.Linear(in_features=64, out_features=out_channels)

    def forward(self, input):
        output = self.net(input)
        output = output.view(input.size(0), -1)
        output = self.fc(output)
        return output

print("\nStart training CNN on NIST36\n")
train_img = np.array([x.reshape(1, 32, 32) for x in train_x])
valid_img = np.array([x.reshape(1, 32, 32) for x in valid_x])

train = NIST36Dataset(train_img, train_y)
valid = NIST36Dataset(valid_img, valid_y)
train_loader = DataLoader(train, batch_size=54, shuffle=True)
valid_loader = DataLoader(valid, batch_size=valid_size, shuffle=False)

model = ConvNet(1, 36)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
max_iters = 50
for itr in range(max_iters):
    model.train()
    total_loss = 0
    total_acc = 0
    for xb, yb in train_loader:
        y_pred = model(xb)
        loss = criterion(y_pred, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * xb.size(0)
        _, preds = torch.max(y_pred.data, 1)
        total_acc += torch.sum(preds == yb.data)

    train_loss.append(total_loss / len(train_loader.dataset))
    train_acc.append(total_acc / len(train_loader.dataset)) 

    # Validation
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in valid_loader:
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
    valid_loss.append(total_loss / len(valid_loader.dataset))
    valid_acc.append(total_acc / len(valid_loader.dataset))

    if itr % 2 == 1:
        print("Itr: {:02d}".format(itr)) 
        print("Train loss: {:.2f} \t acc : {:.2f}".format(train_loss[-1],train_acc[-1]))
        print("Valid loss: {:.2f} \t acc : {:.2f} \n".format(valid_loss[-1],valid_acc[-1]))
    
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
plt.savefig('q6_1_2.png', bbox_inches='tight')
''' 
=====================
End of Q6.1.2
=====================
'''


''' 
=====================
Q6.1.3
=====================
'''
print("\nStart training CNN on CIFAR10\n")
# transform CIFAR10
# reference https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
transform = transforms.Compose(
                    [transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='../data', train=True,
                                        download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=100, shuffle=True)
validset = torchvision.datasets.CIFAR10(root='../data', train=False,
                                        download=True, transform=transform)
validloader = DataLoader(validset, batch_size=len(validset), shuffle=False)

model = ConvNet(3, 10, isNIST36=False)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
max_iters = 20
for itr in range(max_iters):
    model.train()
    total_loss = 0
    total_acc = 0
    for xb, yb in trainloader:
        xb, yb = xb.to(device), yb.to(device)
        y_pred = model(xb)
        loss = criterion(y_pred, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * xb.size(0)
        _, preds = torch.max(y_pred.data, 1)
        total_acc += torch.sum(preds == yb.data)

    train_loss.append(total_loss / len(trainloader.dataset))
    train_acc.append(total_acc / len(trainloader.dataset)) 

    # Validation
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in validloader:
            xb, yb = xb.to(device), yb.to(device)
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
    valid_loss.append(total_loss / len(validloader.dataset))
    valid_acc.append(total_acc / len(validloader.dataset))

    if itr % 2 == 1:
        print("Itr: {:02d}".format(itr)) 
        print("Train loss: {:.2f} \t acc : {:.2f}".format(train_loss[-1],train_acc[-1]))
        print("Valid loss: {:.2f} \t acc : {:.2f} \n".format(valid_loss[-1],valid_acc[-1]))
    
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
plt.savefig('q6_1_3.png', bbox_inches='tight')
''' 
=====================
End of Q6.1.3
=====================
'''


''' 
=====================
Q6.1.4
=====================
'''
# Reference: my own code in the extra credit part for HW1
print("\nStart training CNN on SUN\n")

transform = transforms.Compose([transforms.Resize((224, 224))])
class SunDataset(Dataset):
    def __init__(self, isTrainSet=True):
        if isTrainSet:
            self.img_dir = filtered_train_files
            self.labels = np.loadtxt('../data/SUN/train_labels.txt', np.int64)
        else:
            self.img_dir = filtered_test_files
            self.labels = np.loadtxt('../data/SUN/test_labels.txt', np.int64)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = read_image('../data/SUN/' + self.img_dir[idx])
        img = transform(img)
        label = self.labels[idx]
        return img, label
    

# Filter out malformated images in train/test set
train_files = open('../data/SUN/train_files.txt').read().splitlines()
filtered_train_files = []
for path in train_files:
    try:
        img = read_image('../data/SUN/' + path)
    except:
        continue
    filtered_train_files.append(path)

test_files = open('../data/SUN/test_files.txt').read().splitlines()
filtered_test_files = []
for path in test_files:
    try:
        img = read_image('../data/SUN/' + path)
    except:
        continue
    filtered_test_files.append(path)
    
train_data = SunDataset()
train_size = len(train_data)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
valid_data = SunDataset(isTrainSet=False)
valid_size = len(valid_data)
valid_loader = DataLoader(valid_data, batch_size=valid_size, shuffle=False)

class ConvNetSUN(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=8):
        super(ConvNetSUN, self).__init__()

        self.unit1 = Unit(in_channels, 16, 7, 3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=3)
        self.unit2 = Unit(16, 32, 5)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3)
        self.unit3 = Unit(32, 32, 3)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2)
        self.unit4 = Unit(32, 16, 3)
        self.pool4 = torch.nn.MaxPool2d(kernel_size=2)
        
        #Add all the units into the Sequential layer in exact order
        self.net = torch.nn.Sequential(self.unit1, self.pool1, self.unit2, self.pool2, 
                                       self.unit3, self.pool3, self.unit4, self.pool4)

        self.fc = torch.nn.Linear(in_features=256, out_features=out_channels)

    def forward(self, input):
        output = self.net(input)
        output = output.view(input.size(0), -1)
        output = self.fc(output)
        return output

model = ConvNetSUN(3, 8)
model.cuda()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
max_iters = 20
for itr in range(max_iters):
    model.train()
    total_loss = 0
    total_acc = 0
    for xb, yb in train_loader:
        xb, yb = Variable(xb.float().cuda()), Variable(yb.cuda())
        y_pred = model(xb)
        loss = criterion(y_pred, yb)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item() * xb.size(0)
        _, preds = torch.max(y_pred.data, 1)
        total_acc += torch.sum(preds == yb.data)

    train_loss.append(total_loss / len(train_loader.dataset))
    train_acc.append(total_acc / len(train_loader.dataset)) 

    # Validation
    total_loss = 0
    total_acc = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = Variable(xb.float().cuda()), Variable(yb.cuda())
            y_pred = model(xb)
            loss = criterion(y_pred, yb)
            
            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
    valid_loss.append(total_loss / len(valid_loader.dataset))
    valid_acc.append(total_acc / len(valid_loader.dataset))

    if itr % 2 == 1:
        print("Itr: {:02d}".format(itr)) 
        print("Train loss: {:.2f} \t acc : {:.2f}".format(train_loss[-1],train_acc[-1]))
        print("Valid loss: {:.2f} \t acc : {:.2f} \n".format(valid_loss[-1],valid_acc[-1]))
    
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
plt.savefig('q6_1_4.png', bbox_inches='tight')