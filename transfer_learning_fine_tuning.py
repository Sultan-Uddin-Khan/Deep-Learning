#import
import torch #whole library
import torch.nn as nn #loss function, neural network, convolutional neural network
import torch.optim as optim # gradient descent, adam etc. optimization pattern
import torch.nn.functional as F #relu, tanh the function that has no parameter
from torch.utils.data import DataLoader #for better management of dataset
import torchvision.datasets as datasets # import standard datsets
import torchvision.transforms as transforms #transformation on dataset
import torchvision

#Set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
in_channels=784
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=5



class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self,x):
        return x
#Load pretrained model and modify it
model=torchvision.models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad=False
model.avgpool=Identity()
model.classifier=nn.Sequential(nn.Linear(512,100),nn.ReLU(),nn.Linear(100,10))
model.to(device)


#Load data
train_dataset = datasets.CIFAR10(root='dataset/', train=True, transform=transforms.ToTensor(),download=False)
train_loader= DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)



#Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#Train the network
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        #Get data to CUDA if possible
        data=data.to(device=device)
        targets=targets.to(device=device)

        #forward
        scores=model(data)
        loss=criterion(scores, targets)

        #backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()

#Check accuracy on training and test to see how godd our model is
def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")
    num_correct=0
    num_samples=0
    model.eval()

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device=device)
            y=y.to(device=device)

            scores=model(x)
            _,predictions=scores.max(1) # find the value at index 2 of 64x10
            num_correct+=(predictions == y).sum()
            num_samples+=predictions.size(0) #samples are in the index 0

        print(f'Got {num_correct}/ {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader,model)
#check_accuracy(test_loader,model)





