from dataset import BinaryFashionMNIST
from network import BasicQNN

from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
import torch

# training setup
class_1 = 0
class_2 = 1
n_components = 4
batch_size = 32
shuffle=True
n_layers=4
learning_rate = 1e-3
epochs = 100

# initialize the train and test dataloaders
train_dataset = BinaryFashionMNIST(class_1=class_1,class_2=class_2,train=True,n_components=n_components)
test_dataset = BinaryFashionMNIST(class_1=class_1,class_2=class_2,train=False,n_components=n_components)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=batch_size, shuffle=shuffle)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size)

# initialize the model
model = BasicQNN(n_qubits=n_components,n_layers=n_layers)


optimizer = Adam(model.parameters(), lr=learning_rate)
criterion = BCEWithLogitsLoss()

#training loop
train_losses = []
test_accuracies = []
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_x, batch_y in train_dataloader:
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        loss.backward()
        optimizer.step()
        total_loss+=loss.item()

    avg_loss = total_loss / len(train_dataloader)
    train_losses.append(avg_loss)

    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            outputs = model(inputs)
            predicted = (outputs > 0).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)
    
