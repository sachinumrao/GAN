import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils import data as data
import torchvision.datasets as datasets
from torchvision import transforms

# Define transformation on data
train_transforms = transforms.Compose([transforms.ToTensor()])

# Load MNIST train data
mnist_train = datasets.MNIST(root = './data',
                            train=True,
                            download=True,
                            transform=train_transforms)

# parameters
batch_size = 128
latent_size = 128

# Dataloader for training data
train_dataloader = data.DataLoader(mnist_train,
                                batch_size=batch_size,
                                shuffle=True)


# Noise sampling function
def sample_noise():
    return torch.rand(batch_size, latent_size)

# Define Generator network architecture
class GeneratorNetwork(nn.Module):

    def __init__(self, input_size, output_size):

        super(GeneratorNetwork, self).__init__()

        self.model = nn.Sequential(
                                nn.Linear(input_size, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 1024),
                                nn.ReLU(inplace=True),
                                nn.Linear(1024, 800),
                                nn.ReLU(inplace=True),
                                nn.Linear(800, output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out


# Define Discriminator network architecture
class DiscriminatorNetwork(nn.Module):

    def __init__(self, input_size, output_size):
        super(DiscriminatorNetwork, self).__init__()

        self.model = nn.Sequential(
                                nn.Linear(input_size, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, 512),
                                nn.ReLU(inplace=True),
                                nn.Linear(512, 256),
                                nn.ReLU(inplace=True),
                                nn.Linear(256, output_size),
                                nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x)
        return out

def plot_images(array):
    # transform generator output to image dimensions
    array = array.cpu().detach().numpy()
    array = array.reshape(-1,28,28)
    batch_size = array.shape[0]

    fig = plt.figure(figsize=(8,8))
    cols = min(4, batch_size)
    rows = batch_size//cols

    for i in range(1, cols*rows + 1):
        fig.add_subplot(rows, cols, i)
        plt.imshow(array[i-1, :, :], cmap='binary')

    plt.show()

# parameters
discriminator_train_steps = 50
generator_train_steps = 50
input_size = 784
output_size = 1
logging_steps = 50
epochs = 300

# Instantiate discriminator network
discriminator = DiscriminatorNetwork(input_size, output_size)
discm_loss_crit = nn.BCELoss()
discm_optim = optim.SGD(discriminator.parameters(), lr=0.005, momentum=0.9)

# Instantiate generator network
generator = GeneratorNetwork(latent_size, input_size)
gener_loss_crit = nn.BCELoss()
gener_optim = optim.SGD(generator.parameters(), lr=0.005, momentum=0.9)

# Trainign loop
for epoch in range(epochs):
    print(f'Epoch : {epoch}')

    for j in range(discriminator_train_steps):
        # Train discriminator network
        for i, train_data in enumerate(train_dataloader, 0):
            discriminator.zero_grad()
            break

        # Convert 2d matirx to vector
        train_imgs, labels = train_data
        
        n = train_imgs.shape[0]
        train_imgs = train_imgs.reshape(n, 784)

        # Train discriminator on real data
        discm_real_out = discriminator(train_imgs)
        discm_real_loss = discm_loss_crit(discm_real_out, 
                                    Variable(torch.ones(n, 1)))

        # calculate gradients
        discm_real_loss.backward()

        # Training discriminator on generator data
        # Sample noise from function
        noise = sample_noise()

        # Use noise to generate image by generator
        gener_data = generator(noise).detach()

        # Pass generated data to dicriminator
        discm_gener_out = discriminator(gener_data)

        discm_gener_loss = discm_loss_crit(discm_gener_out,
                                        Variable(torch.zeros(batch_size, 1)))
        # calculate gradients
        discm_gener_loss.backward()

        if (j+1)%logging_steps == 0:
            dism_gener_loss = discm_gener_loss.item()
            dism_real_loss = discm_real_loss.item()
            print(f'Epoch : {epoch+1} Discm Step : {j+1} Real Loss: {dism_real_loss:.5f} Gener Loss: {dism_gener_loss:.5f}')

        # apply gradients
        discm_optim.step()

    # Train generator network
    for j in range(generator_train_steps):
        generator.zero_grad()

        # Generate data to feed into generator network
        gener_input_data = sample_noise()

        gener_out = generator(gener_input_data)
        discm_out = discriminator(gener_out)
        gener_loss = gener_loss_crit(discm_out,
                                    Variable(torch.ones(batch_size, 1)))

        # calculate gradients
        gener_loss.backward()

        
        gener_loss_val = gener_loss.item()
        print(f'Epoch : {epoch+1} Gener Step : {j+1} Gener Loss: {gener_loss_val:.5f}')

        # apply gradient step
        gener_optim.step()

        

# Test GAN
plot_images(generator(sample_noise()))










                            