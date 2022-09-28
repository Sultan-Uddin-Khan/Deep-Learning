#import
import torch #whole library
import torch.nn as nn #loss function, neural network, convolutional neural network
import torch.optim as optim # gradient descent, adam etc. optimization pattern
import torch.nn.functional as F #relu, tanh the function that has no parameter
from torch.utils.data import DataLoader #for better management of dataset
import torchvision.datasets as datasets # import standard datsets
import torchvision.transforms as transforms #transformation on dataset

#Create fully connected network


class NN(nn.Module):
    def __init__(self, input_size, num_classes): #28x28=784 nodes (MNIST class image size)
        super(NN, self).__init__() #super calls the initialization method of parent class which is nn.module here
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = NN(784, 10)
x = torch.randn(64, 784)
print(model(x).shape) #torch.Size([64, 10]) is the output which is mini batch size

#Set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
input_size=784
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=1

#Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(),download=True)
train_loader= DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(),download=True)
test_loader= DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#initialize the network
model = NN(input_size=input_size,num_classes=num_classes).to(device)

#Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#Train the network
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        #Get data to CUDA if possible
        data=data.to(device=device)
        targets=targets.to(device=device)

        #Get to shape
        data=data.reshape(data.shape[0], -1)
        print(data.shape)

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
            x=x.reshape(x.shape[0], -1)

            scores=model(x)
            _,predictions=scores.max(1) # find the value at index 2 of 64x10
            num_correct+=(predictions == y).sum()
            num_samples+=predictions.size(0) #samples are in the index 0

        print(f'Got {num_correct}/ {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)





