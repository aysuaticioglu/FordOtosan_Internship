# Import Required Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from constant import *
from model import FoInternNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check

# Define Hyperparameters
valid_size = 0.3
test_size = 0.1
batch_size = 4
epochs = 20
cuda = True  # Whether to use GPU
input_shape = (224, 224)
n_classes = 2
output_png_path = 'graphs/loss_graph.png'

# Define Directories
IMAGE_DIR = 'data/images'
MASK_DIR = 'data/masks'

# Prepare Image and Mask Lists
# Glob all image and mask files in the directories
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

# Check if the number of image and mask files match
image_mask_check(image_path_list, mask_path_list)

# Shuffle Indices for Randomization
indices = np.random.permutation(len(image_path_list))

# Define Test and Valid Indices
test_ind = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# Slice Test Dataset
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# Slice Validation Dataset
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# Slice Train Dataset
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]

# Define Steps per Epoch
steps_per_epoch = len(train_input_path_list) // batch_size

# Instantiate the Model
model = FoInternNet(input_size=input_shape, n_classes=n_classes)

# Define Loss Function and Optimizer
criterion = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for binary classification
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Move Model to CUDA (if applicable)
if cuda:
    model = model.cuda()

# Function to Calculate Accuracy
def calculate_accuracy(outputs, labels):
    with torch.no_grad():
        predicted = torch.sigmoid(outputs) > 0.5  
        correct = (predicted == labels.bool()).sum().item()
        total = labels.size(0) * labels.size(2) * labels.size(3)
        accuracy = correct / total
        return accuracy
    
# Training Loop
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    running_accuracy = 0
    
    # Shuffle Training Data for Each Epoch
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

        optimizer.zero_grad()
        outputs = model(batch_input)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()    

        running_loss += loss.item()
        running_accuracy += calculate_accuracy(outputs, batch_label)

    train_losses.append(running_loss / steps_per_epoch)
    train_accuracies.append(running_accuracy / steps_per_epoch)

    # Validation Loop
    val_loss = 0
    val_accuracy = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
            batch_input = tensorize_image([valid_input_path], input_shape)
            batch_label = tensorize_mask([valid_label_path], input_shape, n_classes)
            batch_input = batch_input.cuda() if cuda else batch_input
            batch_label = batch_label.cuda() if cuda else batch_label
            outputs = model(batch_input)
            loss = criterion(outputs, batch_label)
            val_loss += loss.item()
            val_accuracy += calculate_accuracy(outputs, batch_label)
        val_losses.append(val_loss / len(valid_input_path_list))
        val_accuracies.append(val_accuracy / len(valid_input_path_list))

    # Print Epoch-wise Losses and Accuracies
    print('Epoch {}: Train Loss: {:.4f}, Train Accuracy: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}'.format(epoch+1, train_losses[-1], train_accuracies[-1], val_losses[-1], val_accuracies[-1]))

# Print Final Epoch Results
final_epoch = epochs - 1
print("\nFinal Epoch Results:")
print(f"Epoch {final_epoch + 1} - "
      f"Train Loss: {train_losses[final_epoch]:.4f}, Train Accuracy: {train_accuracies[final_epoch]:.4f}, "
      f"Validation Loss: {val_losses[final_epoch]:.4f}, Validation Accuracy: {val_accuracies[final_epoch]:.4f}")

# Save Loss and Accuracy Graphs to a PNG
if not os.path.exists('graphs'):
    os.makedirs('graphs')

def draw_graph_to_png(val_losses, train_losses, val_accuracies, train_accuracies, epochs, output_path):
    # Normalize losses and accuracies for comparison
    norm_validation = [float(i)/sum(val_losses) for i in val_losses]
    norm_train = [float(i)/sum(train_losses) for i in train_losses]
    norm_val_accuracies = [float(i)/sum(val_accuracies) for i in val_accuracies]
    norm_train_accuracies = [float(i)/sum(train_accuracies) for i in train_accuracies]
    
    epoch_numbers = list(range(1, epochs+1, 1))
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.plot(epoch_numbers, norm_validation, color="red")
    plt.plot(epoch_numbers, norm_val_accuracies, color="orange")
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.title('Normalized Validation Losses and Accuracies')
    
    plt.subplot(2, 2, 2)
    plt.plot(epoch_numbers, norm_train, color="blue")
    plt.plot(epoch_numbers, norm_train_accuracies, color="green")
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    plt.title('Normalized Train Losses and Accuracies')
    
    plt.subplot(2, 1, 2)
    plt.plot(epoch_numbers, norm_validation, 'r-')
    plt.plot(epoch_numbers, norm_train, 'b-')
    plt.plot(epoch_numbers, norm_val_accuracies, 'r--')
    plt.plot(epoch_numbers, norm_train_accuracies, 'b--')
    plt.legend(['Validation Loss', 'Train Loss', 'Validation Acc', 'Train Acc'])
    plt.title('Train and Validation Losses and Accuracies')
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(1))
    
    plt.savefig(output_path)
    plt.close()

draw_graph_to_png(val_losses, train_losses, val_accuracies, train_accuracies, epochs, output_png_path)

# Visualize and Save Predictions
def visualize_and_save_predictions(test_input_path_list, test_label_path_list, test_masked_path_list, n_samples=5):
    if not os.path.exists('predictions'):
        os.makedirs('predictions')

    for i in range(n_samples):
        test_input = tensorize_image([test_input_path_list[i]], input_shape, cuda)
        test_label = tensorize_mask([test_label_path_list[i]], input_shape, n_classes, cuda)
        with torch.no_grad():
            outs = model(test_input)
        
        predicted_mask = torch.sigmoid(outs).cpu().numpy()
        predicted_mask = predicted_mask[0, 1, :, :]
        
        img = cv2.imread(test_input_path_list[i])
        img_resized = cv2.resize(img, input_shape)
        
        mask_resized = cv2.resize(predicted_mask, img_resized.shape[:2])
        mask_ind = mask_resized > 0.5
        
        ground_truth_mask = cv2.imread(test_label_path_list[i], cv2.IMREAD_GRAYSCALE)
        ground_truth_mask = cv2.resize(ground_truth_mask, img_resized.shape[:2])
        ground_truth_mask = (ground_truth_mask > 128).astype(np.uint8) * 255  # Convert to binary mask
        
        masked_input_img = cv2.imread(test_masked_path_list[i])
        masked_input_img_resized = cv2.resize(masked_input_img, input_shape)
        
        # Color the masked area
        masked_image = np.copy(img_resized)
        masked_image[mask_ind] = [0, 255, 0]  # Highlight masked area in green
        
        # Color the ground truth mask area
        ground_truth_colored = cv2.cvtColor(ground_truth_mask, cv2.COLOR_GRAY2BGR)
        ground_truth_colored[mask_ind] = [0, 255, 0]  # Highlight masked area in green
        
        # Concatenate images horizontally
        concat_images = np.concatenate((img_resized, masked_image, ground_truth_colored, masked_input_img_resized), axis=1)
        
        masked_image_name = os.path.basename(test_masked_path_list[i])
        prediction_name = f"prediction_{masked_image_name}"
        prediction_path = os.path.join('predictions', prediction_name)
        
        # Create a separate matrix for each image
        result_image = np.copy(concat_images)
        
        # Set font and color for text
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White color
        font_thickness = 2
        
        # Set the height for placing text
        text_height = 20
        
        # Add text to the input image
        cv2.putText(result_image, 'Input Image', (5, text_height), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        # Add text to the predicted image
        cv2.putText(result_image, 'Predicted Image', (input_shape[1] + 10, text_height), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        # Add text to the ground truth image
        cv2.putText(result_image, 'Ground Truth Image', (2 * input_shape[1] + 10, text_height), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        # Add text to the labeled image
        cv2.putText(result_image, 'Labeled Image', (3 * input_shape[1] + 10, text_height), font, font_scale, font_color, font_thickness, cv2.LINE_AA)
        
        cv2.imwrite(prediction_path, result_image)

# List of masked images for testing predictions
test_masked_path_list = glob.glob(os.path.join('data/masked_images', '*'))
test_masked_path_list.sort()

n_samples = 5  # Number of samples to visualize
visualize_and_save_predictions(test_input_path_list, test_label_path_list, test_masked_path_list, n_samples)
