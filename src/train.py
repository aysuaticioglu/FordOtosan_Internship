from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
from constant import *
import os
import glob
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = BATCH_SIZE
epochs = 20
cuda = True
input_shape = (HEIGHT, WIDTH)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]


# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
model = FoInternNet(input_size=input_shape, n_classes=n_classes)


# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()
else:
    model = model.to('cpu')

def calculate_accuracy(outputs, labels):
    with torch.no_grad():
        predicted = outputs.round() 
        correct = (predicted == labels).sum().item()  
        total = labels.size(0)  
        accuracy = correct / total
        return accuracy

# TRAINING THE NEURAL NETWORK
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    running_accuracy = 0
    
    pair_IM = list(zip(train_input_path_list, train_label_path_list))
    np.random.shuffle(pair_IM)
    unzipped_object = zip(*pair_IM)
    zipped_list = list(unzipped_object)
    train_input_path_list = list(zipped_list[0])
    train_label_path_list = list(zipped_list[1])
    
    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size * ind:batch_size * (ind + 1)]
        batch_label_path_list = train_label_path_list[batch_size * ind:batch_size * (ind + 1)]
        
        batch_input = tensorize_image(batch_input_path_list, input_shape)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes)

        if cuda: 
            batch_input = batch_input.cuda()
            batch_label = batch_label.cuda()
        else:
            batch_input = batch_input.to('cpu')
            batch_label = batch_label.to('cpu')

        optimizer.zero_grad()
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()    
        
        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_label)
    
    train_losses.append(running_loss / steps_per_epoch)
    train_accuracies.append(running_accuracy / steps_per_epoch)
    
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
            batch_input = tensorize_image([valid_input_path], input_shape)
            batch_label = tensorize_mask([valid_label_path], input_shape, n_classes)
            outputs = model(batch_input)
            loss = criterion(outputs, batch_label)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, batch_label)
        val_losses.append(val_loss / len(valid_input_path_list))
        val_accuracies.append(val_accuracy / len(valid_input_path_list))
    
    print('Epoch {}: Train Loss: {:.4f}, Train Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch+1, train_losses[-1], train_accuracies[-1], val_losses[-1], val_accuracies[-1]))

# Plotting
plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss over Epochs')
plt.legend()
plt.show()

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy over Epochs')
plt.legend()
plt.show()
# Print final epoch values
final_epoch = epochs - 1
print("\nFinal Epoch Results:")
print(f"Epoch {final_epoch + 1} - "
      f"Train Loss: {train_losses[final_epoch]:.4f}, Train Accuracy: {train_accuracies[final_epoch]:.4f}, "
      f"Validation Loss: {val_losses[final_epoch]:.4f}, Validation Accuracy: {val_accuracies[final_epoch]:.4f}")
