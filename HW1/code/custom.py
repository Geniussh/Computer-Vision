import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.autograd import Variable
from torchvision.io import read_image
from torchvision import transforms
from torchvision import transforms
import numpy as np
import cv2

p = transforms.Compose([transforms.Resize((224, 224))])

# Custom Dataset
class SunDataset(Dataset):
    def __init__(self, isTrainSet=True):
        if isTrainSet:
            self.img_dir = filtered_train_files
            self.labels = np.loadtxt('../data/train_labels.txt', np.int64)
        else:
            self.img_dir = filtered_test_files
            self.labels = np.loadtxt('../data/test_labels.txt', np.int64)

    def __len__(self):
        return len(self.img_dir)

    def __getitem__(self, idx):
        img = read_image('../data/' + self.img_dir[idx])
        img = p(img)
        label = self.labels[idx]
        return img, label

class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(Unit,self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=kernel_size,
                                out_channels=out_channels, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.relu = nn.ReLU()

    def forward(self,input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.relu(output)

        return output

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        self.unit1 = Unit(3, 16, 7, 3)
        self.pool1 = nn.MaxPool2d(kernel_size=3)
        self.unit2 = Unit(16, 32, 5)
        self.pool2 = nn.MaxPool2d(kernel_size=3)
        self.unit3 = Unit(32, 32, 3)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.unit4 = Unit(32, 16, 3)
        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        #Add all the units into the Sequential layer in exact order
        self.net = nn.Sequential(self.unit1, self.pool1, self.unit2, self.pool2, 
                                    self.unit3, self.pool3, self.unit4, self.pool4)

        self.fc = nn.Linear(in_features=256, out_features=8)

    def forward(self, input):
        output = self.net(input)
        output = output.view(input.size(0), -1)
        output = self.fc(output)
        return output


# Filter out malformated images in train/test set
train_files = open('../data/train_files.txt').read().splitlines()
filtered_train_files = []
for path in train_files:
    try:
        img = read_image('../data/' + path)
    except:
        continue
    filtered_train_files.append(path)

test_files = open('../data/test_files.txt').read().splitlines()
filtered_test_files = []
for path in test_files:
    try:
        img = read_image('../data/' + path)
    except:
        continue
    filtered_test_files.append(path)

training_data = SunDataset()
train_dataloader = DataLoader(training_data, batch_size=32, shuffle=True)
training_size = len(training_data)
test_data = SunDataset(isTrainSet=False)
test_dataloader = DataLoader(test_data, batch_size=32, shuffle=False)
test_size = len(test_data)

model = ConvNet()
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()

def test():
    model.eval()
    test_acc = 0.0
    for image, label in test_dataloader:
        if torch.cuda.is_available():
            image = Variable(image.float().cuda())
            label = Variable(label.cuda())
        output = model(image)
        _, pred = torch.max(output.data, 1)
        test_acc += torch.sum(pred == label.data)
    
    return test_acc / test_size

def train(num_epochs):
    best_acc = 0.0
    for epoch in range(num_epochs):
        model.train()
        train_acc = 0.0
        for image, label in train_dataloader:
            if torch.cuda.is_available():
                image = Variable(image.float().cuda())
                label = Variable(label.cuda())
            
            optimizer.zero_grad()
            output = model(image)
            _, pred = torch.max(output.data, 1)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            train_acc += torch.sum(pred == label.data)
        test_acc = test()
        train_acc /= training_size
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'SunModel_%s.model' % epoch)
        print("Epoch {}, Train Accuracy: {} , Test Accuracy: {}".format(epoch, train_acc, test_acc))

 
if __name__ == "__main__":
    train(50)