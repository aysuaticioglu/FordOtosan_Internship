from matplotlib import pyplot as plt
from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import cv2

######### PARAMETERS ##########
# Define hyperparameters and settings for the training process
valid_size = 0.3  # Validation set size as a fraction of the whole dataset
test_size = 0.2  # Test set size as a fraction of the whole dataset
batch_size = 4  # Batch size for training
epochs = 20  # Number of training epochs
cuda = True  # Whether to use CUDA for GPU acceleration
input_shape = (224, 224)  # Input image size
n_classes = 2  # Number of classes (e.g., binary classification)
###############################

######### DIRECTORIES #########
# Define the directories where images and masks are stored
MASK_DIR = 'data/masks'  # Directory for original masks
IMAGE_DIR = 'data/images'  # Directory for original images
AUG_IMAGE = 'data/augmentation'  # Directory for augmented images
AUG_MASK = 'data/augmentation_mask'  # Directory for augmented masks
###############################

# PREPARE IMAGE AND MASK LISTS
# Create lists of image and mask file paths
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()
mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# PREPARE IMAGE AND MASK LISTS
# Create lists of augmented image and mask file paths
aug_path_list = glob.glob(os.path.join(AUG_IMAGE, '*'))
aug_path_list.sort()
aug_mask_path_list = glob.glob(os.path.join(AUG_MASK, '*'))
aug_mask_path_list.sort()

# DATA CHECK
# Check if the number of image and mask files match
image_mask_check(image_path_list, mask_path_list)

# SHUFFLE INDICES
# Shuffle the indices to randomize the dataset
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
# Calculate the indices for test and validation sets
test_ind = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
# Divide the dataset into test, validation, and training sets
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

aug_size = int(len(aug_mask_path_list) / 2)
train_input_path_list = aug_path_list[:aug_size] + train_input_path_list + aug_path_list[aug_size:]
train_label_path_list = aug_mask_path_list[:aug_size] + train_label_path_list + aug_mask_path_list[aug_size:]

# CALL MODEL
# Create an instance of the FoInternNet model
model = FoInternNet(input_size=input_shape, n_classes=2)

# DEFINE LOSS FUNCTION AND OPTIMIZER
# Specify the loss function and optimizer for training
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
# If CUDA is enabled, move the model to the GPU
if cuda:
    model = model.cuda()

val_losses = []
train_losses = []

# TRAINING THE NEURAL NETWORK
# Start the training loop
for epoch in tqdm.tqdm(range(epochs)):
    running_loss = 0

    pair_IM = list(zip(train_input_path_list, train_label_path_list))
    np.random.shuffle(pair_IM)
    unzipped_object = zip(*pair_IM)
    zipped_list = list(unzipped_object)
    train_input_path_list = list(zipped_list[0])
    train_label_path_list = list(zipped_list[1])

    for ind in range(steps_per_epoch):
        batch_input_path_list = train_input_path_list[batch_size * ind:batch_size * (ind + 1)]
        batch_label_path_list = train_label_path_list[batch_size * ind:batch_size * (ind + 1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

        optimizer.zero_grad()
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if ind == steps_per_epoch - 1:
            train_losses.append(running_loss)
            print('training loss on epoch {}: {}'.format(epoch, running_loss))
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                outputs = model(batch_input)
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
            val_losses.append(val_loss)
            print('validation loss on epoch {}: {}'.format(epoch, val_loss))

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print("Model Saved!")

def draw_graph(val_losses, train_losses, epochs):
    norm_validation = [float(i) / sum(val_losses) for i in val_losses]
    norm_train = [float(i) / sum(train_losses) for i in train_losses]
    epoch_numbers = list(range(1, epochs + 1, 1)
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers, norm_validation, color="red")
    plt.title('Train losses')
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers, norm_train, color="blue")
    plt.title('Validation losses')
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers, norm_validation, 'r-', color="red")
    plt.plot(epoch_numbers, norm_train, 'r-', color="blue")
    plt.legend(['w=1', 'w=2'])
    plt.title('Train and Validation Losses')
    plt.savefig('loss_graph.png')
    plt.show()

draw_graph(val_losses, train_losses, epochs)

def predict(test_input_path_list):
    # Define the folder you want to create
    predict_mask_dir = "predict_mask"

    # Create the folder if it doesn't exist
    if not os.path.exists(predict_mask_dir):
        os.makedirs(predict_mask_dir)

    for i in tqdm.tqdm(range(len(test_input_path_list)):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
        outs = model(test_input)
        out = torch.argmax(outs, axis=1)
        out_cpu = out.cpu()
        outputs_list = out_cpu.detach().numpy()
        mask = np.squeeze(outputs_list, axis=0)

        img = cv2.imread(batch_test[0])
        mg = cv2.resize(img, (224, 224))
        mask_ind = mask == 1
        cpy_img = mg.copy()
        mg[mask == 1, :] = (255, 0, 125)
        opac_image = (mg / 2 + cpy_img / 2).astype(np.uint8)
        predict_name = os.path.basename(batch_test[0])
        predict_path = os.path.join(predict_mask_dir, predict_name.replace('.jpg', '.png'))
        cv2.imwrite(predict_path, opac_image.astype(np.uint8))
predict(test_input_path_list)
