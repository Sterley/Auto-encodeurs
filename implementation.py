import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import math


# Autoencoder architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

# Loading MNIST dataset
train_dataset = datasets.MNIST(root='data', train=True, download=True, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
# # Create an instance of the autoencoder
autoencoder = Autoencoder()
# Define the loss function and the optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(), lr=0.001)


x_values = []
y_values = []
# Training the autoencoder
num_epochs = 10
for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        optimizer.zero_grad()
        output = autoencoder(img)
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()
    x_values.append(epoch)
    y_values.append(loss.item())
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Show the loss function
plt.plot(x_values, y_values)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss function')
plt.grid(True)
plt.show()

def add_noise(img):
    noise = np.random.normal(loc=0.0, scale=0.2, size=img.shape)
    img += torch.Tensor(noise)
    return img

# Testing the autoencoder
with torch.no_grad():
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)
    for data in test_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        # Add noise if specified
        img = add_noise(img)
        output = autoencoder(img)
        fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))
        for i in range(5):
            axs[0, i].imshow(img[i].view(28, 28), cmap='gray')
            axs[1, i].imshow(output[i].view(28, 28), cmap='gray')
        plt.show()
        break


# Visualize the weights of the first layer of the decoder
weights = autoencoder.decoder[0].weight.data
num_images = weights.shape[0]
# Calculate the number of rows and columns
cols = min(10, num_images)
rows = math.ceil(num_images / cols)
# Create a figure
fig, axs = plt.subplots(nrows=rows, ncols=cols, figsize=(20, 2 * rows))
# Show the weights
for i in range(rows):
    for j in range(cols):
        idx = i * cols + j
        if idx < num_images:
            weight_image = weights[idx].view(8, 8)
            axs[i, j].imshow(weight_image, cmap='gray')
        axs[i, j].axis('off')
plt.show()

cp = 0
sum = 0
with torch.no_grad():
    test_dataset = datasets.MNIST(root='data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=False)
    for data in test_loader:
        img, _ = data
        imgC = img.view(img.size(0), -1)
        imgN = add_noise(imgC)
        output = autoencoder(imgN)
        debruiting_error = torch.mean((imgC - output) ** 2)
        sum += debruiting_error.item()
        cp += 1
print("MSE debruiting error : ", sum/cp)