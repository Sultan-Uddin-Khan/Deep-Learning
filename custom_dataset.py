#import
import torch #whole library
import torch.nn as nn #loss function, neural network, convolutional neural network,Batchnorm
import torch.optim as optim # gradient descent, adam etc. optimization pattern
import torchvision
import torchvision.transforms as transforms #transformation on dataset
from torch.utils.data import DataLoader #for better management of dataset
from customDataset import CatsAndDogsDataset

#Set device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
in_channels=3
num_classes=10
learning_rate=0.001
batch_size=32
num_epochs=1

#Load data
dataset=CatsAndDogsDataset(csv_file='cats_dogs.csv',root_dir='cats_dogs_resized', transform=transforms.ToTensor())
train_set, test_set=torch.utils.data.random_split(dataset, [20000, 5000])
train_loader= DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
test_loader= DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

#Model
model=torchvision.models.googlenet(pretrained=True)
model.to(device)

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
check_accuracy(test_loader,model)





