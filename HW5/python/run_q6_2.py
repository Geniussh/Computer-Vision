import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable

use_gpu = True
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
dtype = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor

def dataProcess(isLeNet=False):
    if isLeNet:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # Make it square first
            transforms.Resize(32),  # because LeNet accepts image of size 32x32
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),            
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(224),  # because SqueezeNet accepts image of size 224x224
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),            
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    train = torchvision.datasets.ImageFolder(root='../data/oxford-flowers17/train',
                                            transform=transform)                            
    train_loader = torch.utils.data.DataLoader(train,
                                            batch_size=20, shuffle=True,
                                            num_workers=4)
    valid = torchvision.datasets.ImageFolder(root='../data/oxford-flowers17/val',
                                            transform=transform)                            
    valid_loader = torch.utils.data.DataLoader(valid,
                                            batch_size=len(valid), shuffle=False,
                                            num_workers=4)
    test = torchvision.datasets.ImageFolder(root='../data/oxford-flowers17/test',
                                            transform=transform)                            
    test_loader = torch.utils.data.DataLoader(test,
                                            batch_size=len(test), shuffle=False,
                                            num_workers=4)
    
    return train_loader, valid_loader, test_loader

def plotTrend(filename, max_iters, train_acc, valid_acc, train_loss, valid_loss):
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
    plt.savefig(filename, bbox_inches='tight')

def fineTune():
    train_loader, valid_loader, test_loader = dataProcess()

    model = torchvision.models.squeezenet1_1(pretrained=True)
    model.num_classes = 17
    # The final classifier layer in SqueezeNet consists of 
    # nn.Dropout, final_conv, nn.ReLU, nn.AdaptiveAvgPool2d
    model.classifier[1] = torch.nn.Conv2d(512, model.num_classes, kernel_size=1)
    if use_gpu:
        model.cuda()
    
    # we want to train only the reinitialized last layer for a few epochs.
    # During this phase we do not need to compute gradients with respect to the
    # other weights of the model, so we set the requires_grad flag to False for
    # all model parameters, then set requires_grad=True for the parameters in the
    # last layer only.
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    max_iters = 20
    for itr in range(max_iters):
        model.train()
        total_loss = 0
        total_acc = 0
        for xb, yb in train_loader:
            if use_gpu:
                xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
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
                if use_gpu:
                    xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
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
    plotTrend('q6_2_ft.png', max_iters, train_acc, valid_acc, train_loss, valid_loss)

    # Test Accuracy
    total_acc = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = Variable(xb.type(dtype)), Variable(yb.type(dtype))
            y_pred = model(xb)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
    print('Test Accuracy of Fine Tuning: {:.2f}'.format(total_acc / len(test_loader.dataset)))


def trainLeNet():
    train_loader, valid_loader, test_loader = dataProcess(isLeNet=True)

    class LeNet(torch.nn.Module):
        def __init__(self, in_channels=3, out_channels=17):
            super(LeNet, self).__init__()
            self.featureExtractor = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
                torch.nn.Tanh(),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
                torch.nn.Tanh(),
                torch.nn.AvgPool2d(kernel_size=2, stride=2),
                torch.nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
                torch.nn.Tanh()
            )
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(in_features=120, out_features=84),
                torch.nn.Tanh(),
                torch.nn.Linear(in_features=84, out_features=out_channels)
            )
        
        def forward(self, x):
            feature = self.featureExtractor(x)
            feature_flat = torch.flatten(feature, 1)
            logits = self.classifier(feature_flat)
            probs = torch.nn.functional.softmax(logits, dim=1)
            return logits, probs

    model = LeNet(3, 17)
    model.cuda()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_loss, train_acc, valid_loss, valid_acc = [], [], [], []
    max_iters = 40
    for itr in range(max_iters):
        model.train()
        total_loss = 0
        total_acc = 0
        for xb, yb in train_loader:
            xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
            y_pred, _ = model(xb)
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
                xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
                y_pred, _ = model(xb)
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
    plotTrend('q6_2_ln.png', max_iters, train_acc, valid_acc, train_loss, valid_loss)

    # Test Accuracy
    total_acc = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = Variable(xb.cuda()), Variable(yb.cuda())
            y_pred, _ = model(xb)
            _, preds = torch.max(y_pred.data, 1)
            total_acc += torch.sum(preds == yb.data)
    print('Test Accuracy of LeNet trained from scratch: {:.2f}'.format(total_acc / len(test_loader.dataset)))


# print("\nStart Fine Tuning SqueezeNet 1.1\n")
# fineTune()

print("\nStart Training LeNet from Scratch\n")
trainLeNet()
