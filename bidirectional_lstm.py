#import
import torch #whole library
import torch.nn as nn #loss function, neural network, convolutional neural network
import torch.optim as optim # gradient descent, adam etc. optimization pattern
import torch.nn.functional as F #relu, tanh the function that has no parameter
from torch.utils.data import DataLoader #for better management of dataset
import torchvision.datasets as datasets # import standard datsets
import torchvision.transforms as transforms #transformation on dataset


#Set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
input_size=28
sequence_length=28
num_layers=2
hidden_size=256
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=2

#Create a bidirectional LSTM
class BRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BRNN, self).__init__()
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,bidirectional=True)
        self.fc=nn.Linear(hidden_size*2, num_classes)

    def forward(self,x):
        h0=torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        #Forward prop
        out, _=self.lstm(x,(h0,c0))
        out=self.fc(out[:,-1,:])
        return out

#Load data
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=transforms.ToTensor(),download=True)
train_loader= DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=transforms.ToTensor(),download=True)
test_loader= DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)


#initialize the network
model = BRNN(input_size, hidden_size, num_layers, num_classes).to(device)

#Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

#Train the network
for epoch in range(num_epochs):
    for batch_idx, (data,targets) in enumerate(train_loader):
        #Get data to CUDA if possible
        data=data.to(device=device).squeeze(1)
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
            x=x.to(device=device).squeeze(1)
            y=y.to(device=device)

            scores=model(x)
            _,predictions=scores.max(1) # find the value at index 2 of 64x10
            num_correct+=(predictions == y).sum()
            num_samples+=predictions.size(0) #samples are in the index 0

        print(f'Got {num_correct}/ {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')

    model.train()

check_accuracy(train_loader,model)
check_accuracy(test_loader,model)





