# FordOtosan_Internship
<h1>Freespace Segmentation with Fully Convolutional Neural Network (FCNN)</h1>


<h3>What is Machine Learning?</h3>
It is a branch of artifical intelligence where computer systems learn and perform tasks by analyzing data and using their experiences.Machine learning algorithms and statistical models are used to analysze data,discover patterns,relationships and extract meaningful information.

<img alt="Adsız" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/1.png">



<h3>What is Unsupervised vs Supervised learning difference?</h3>
The main difference is whether the training data has labels or not. In supervised learning, the data includes inputs and corresponding labels. In unsupervised learning, the data is unlabeled, and the focus is on discovering structures and patterns within the data.

<img alt="Adsız" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/2.png">

<h3>What is Deep Learning?</h3>
 It is a subfield of machine learning that uses mathematical models called artificial neural networks to perform complex tasks.Deep learning methods automatically learn and interpret complex structures in data by leveraging datasets.

<h3>What is Neural Network (NN)?</h3>
 It's a mathematical model that imitates the structure of the human brain. It uses interconnected artificial neurons to handle input data and generate outputs.Neural networks are widely used in machine learning.
<br>
<img alt="Adsız" width="300" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/3.png">

<h3>What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.</h3>
It is a special kind of neural network made for working with structured data,particularly images.CNNs are widely used in deep learning for tasks like sorting images into categories,recognizing objeects and dividing images into parts.CNNs have two benefits over regular neural networks:they can automatically learn important features from the data and they share parameters between different parts of the network which make them more efficient.

<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/4.jpeg">



<h3>What is segmentation task in NN? Is it supervised or unsupervised?</h3>
The segmentation task involves identifying and delineating different objects or regions within data,typically images.It usually requires labeled data, making it a supervised task.

<h3>What is classification task in NN? Is it supervised or unsupervised?</h3>
The classification task assigns input data to specific categories or classes.It is also a supervised task that requires labeled data with inputs and corresponding target classes.

<h3>Compare segmentation and classification in NN.</h3>
Segmentation focuses on identifying regions or objects within data, while classification assigns data to predefined classes or categories. Segmentation is more detailed and deals with complex data structures, while classification is a more general task with simpler output.
<br>
<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/5.png">


<h3>What is data and dataset difference?</h3>
Data refers to any information or observations, while a dataset is a collection of organized data specifically used for training or evaluating machine learning models.
  
<h3>What is the difference between supervised and unsupervised learning in terms of dataset?</h3>
In supervised learning, the dataset contains labeled data with inputs and corresponding labels. In unsupervised learning, the dataset consists of unlabeled data, and the algorithms aim to find patterns or structures within the data without predefined labels.

<h1>Data Preprocessing</h1>
<h2>Extracting Masks</h2>


<h3>What is color space ?</h3>
Color space refers to specific way of representing colors in a mathematical or visual model.It defines the range of colors that can be displayed or captured and provides a standardized system for encoding and interpreting colors.

<h3>What RGB stands for ?</h3>
RGB stands for Red,Green and Blue.It is an additive color model in which colors are created by combining different intensities of red,green and blue primary colors.RGB is widely used in digital imaging and displays.


<h3>In Python, can we transform from one color space to another?</h3>
Yes,in Python,we can transform images from one color space to another using various libraries and functions.Libraries like OpenCv an PIL(Python Imaging Library) provide functions and methods to convert between diffrent color spaces,such as RGB,HSV,CMYK and more.


<h3>What is the popular library for image processing?</h3>

One popular library for image processing in Python is OpenCV (Open Source Computer Vision Library). OpenCV provides a wide range of functions and algorithms for image manipulation, transformation, feature extraction, object detection, and more. It is widely used in computer vision and image processing applications.

<h2>Json to Mask</h2>


JSON files contain the exterior and interior point locations of the freespace (drivable area) class. With this information, we can create a mask image.
Here's the JSON representation:


<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/6.png">

The masks are created using these exterior and interior points. Masks are black and white images representing pixels. Pixels belonging to the "Freespace" class are represented as white (e.g., pixel value 255), while other pixels are represented as black (e.g., pixel value 0). This is done using the information from the JSON files.

Masks were drawn using the Python cv2 library and the fillPoly function.
```python
mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)
```
<h4>Mask example;</h4>


<img alt="Adsız" width="300" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/7.png">
<img alt="Adsız" width="300" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/8.png">
<img alt="Adsız" width="300" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/9.png">

Here's the section with the codes;
<a href="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/src/json2mask.py">json2mask.py</a>

<h2>Mask on Image</h2>

In this section, the masked areas are highlighted in green or another color. The masked regions correspond to pixels belonging to the "Freespace" class, which are obtained from JSON files. After overlaying these masked areas on the original images, specific regions of the images will be emphasized with the chosen color, such as green.

The use of 50% opacity during this process helps to achieve a smoother and more transparent effect. This allows the masked areas to blend better with the original images. As a result, the masked regions are visualized in a way that harmonizes with the original images.

Overall, this process aims to visually inspect the accuracy and appropriateness of the masks, by superimposing them on the original images and highlighting the relevant areas with the chosen color and opacity.

Codes to add to the image by coloring the pixels determined as empty space and with 50% opacity:
```python
  cpy_image=image.copy();
    image[mask==1,:]=(0,255,0)
    opac_image=(image/2+cpy_image/2).astype(np.uint8)
```

<h4>Mask on image example;</h4>
<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/10.png">

Here's the section with the codes;
<a href="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/src/mask_on_image.py">mask_on_image.py</a>


<h2>Converting into Tensor</h2>
<h4>Explain Computational Graph.</h4>
	A Computational Graph is a graphical representation method that shows mathematical operations and their intermediate results. It is an essential concept used to understand the training and backpropagation processes of deep learning and machine learning models.
In a Computational Graph, all the mathematical operations and intermediate values from the input data to the output data are represented as nodes and the relationships between them are represented as directed edges.
This graph illustrates how data flows during the forward pass and how gradients are computed during backpropagation. It provides insights into the computations performed by the model during the training process and helps visualize how parameters are updated.
Computational Graphs are valuable tools to gain a deeper understanding of the mathematical structure of deep learning models and make them more interpretable and traceable. They play a crucial role in improving the model's performance and understanding and correcting errors.


For example, consider the relatively simple expression: 

<code>f(x, y, z) = (x + y) * z </code>

This is how we would represent that function as as computational graph:

<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/11.png">

There are three input nodes, labeled X, Y, and Z. The two other nodes are function nodes. In a computational graph we generally compose many simple functions into a more complex function. We can do composition in mathematical notation as well, but I hope you’ll agree the following isn’t as clean as the graph above:


<code>f(x, y, z) = h(g(x, y) , z)
g(i, j) = i + j
h(p, q) = p*q
</code>

In both of these notations we can compute the answers to each function separately, provided we do so in the correct order. Before we know the answer to f(x, y, z) first we need the answer to g(x, y) and then h(g(x, y), z). With the mathematical notation we resolve these dependencies by computing the deepest parenthetical first; in computational graphs we have to wait until all the edges pointing into a node have a value before computing the output value for that node. Let’s look at the example for computing f(1, 2, 3).

<code>f(1, 2, 3) = h(g(1, 2), 3)
g(1, 2) = 1 + 2 = 3
f(1, 2, 3) = h(3, 3)
h(3, 3) = 3*3 = 9
f(1, 2, 3) = 9</code>

And in the graph, we use the output from each node as the weight of the corresponding edge:

<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/12.png">




The weight of the outbound edge for the plus (+) node has to be computed before the the multiply (*) node can compute its outbound edge weight.
Either way, graph or function notation, we get the same answer because these are just two ways of expressing the same thing.
In this simple example it might be hard to see the advantage of using a computational graph over function notation. After all, there isn’t anything terribly hard to understand about the function f(x, y, z) = (x + y) * z. The advantages become more apparent when we reach the scale of neural networks.



<h4>What is Tensor?</h4>
A Tensor is a mathematical concept used in scientific and computational calculations. It can be thought of as a generalization of a vector and a matrix.

In simple terms, a Tensor represents data in multi-dimensional arrays or matrices. A scalar is the simplest form of a Tensor and contains only one number. A vector is a Tensor that consists of a sequence of n numbers. A matrix can be thought of as a table with rows and columns.

However, a Tensor can have three or more dimensions. For example, a three-dimensional Tensor is used to represent a colored image (height x width x color channels). Video data can be represented using four-dimensional Tensors (sequence of frames x height x width x color channels).

Tensors are widely used in areas like deep learning and artificial intelligence. In these fields, input and output data in neural network models, as well as weights and gradients, are represented as Tensors, and various mathematical operations are performed on these Tensors.

In conclusion, Tensors are essential for representing data in different dimensions and structures and performing mathematical operations. They are widely used in various fields, from scientific calculations to artificial intelligence applications.
￼

<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/13.png">


<h4>What is one hot encoding?</h4>
One Hot Encoding is a method used in machine learning and deep learning models to represent categorical data in a numerical format. In this method, each category is represented as an array (vector), where only the element corresponding to the category is "1" and the other elements are "0". This way, categorical data is converted into a numerical format.
For example, if we have three different color types: “red”, “green” and “blue, we can apply One Hot Encoding as follows:

* One Hot Encoding for “red”: [1, 0, 0]
* One Hot Encoding for “green”: [0, 1, 0]
* One Hot Encoding for “blue: [0, 0, 1]

  
This transformation allows machine learning models to process and classify categorical data more effectively. It also emphasizes that there is no inherent order or ranking among the categories.
￼

<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/14.png">


<h4>What is CUDA programming? Answer without detail.</h4>
CUDA is a technology developed by NVIDIA. This technology allows graphics cards, which are typically used to accelerate graphics tasks, to also perform tasks involving mathematical calculations rapidly. In other words, computers can think and process tasks faster.


<h2>Preprocessing</h2>
In this stage, data preprocessing steps were conducted to prepare for training the Image Segmentation model.
Preparing and Normalizing Images
To feed images into the model, images were first normalized and resized to a specific dimension.


```python
import cv2
import numpy as np

def preprocess_image(image_path, output_shape):
    # Read the image
    img = cv2.imread(image_path)
    
    # Normalize the image to the range [0, 255]
    norm_img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    
    # Resize the image to the specified dimension
    img_resized = cv2.resize(norm_img, output_shape)
    
    return img_resized

```
<h3>Preparing Masks and One-Hot Encoding</h3>
To provide masks to the model, masks were first resized to a specific dimension. Subsequently, masks were prepared for the model using one-hot encoding.

```python
def one_hot_encoder(mask, n_classes):
    # Create a separate channel for each class
    one_hot_mask = np.zeros((*mask.shape, n_classes), dtype=np.int)
    
    # Apply one-hot encoding by iterating over each pixel of the mask
    for i, unique_value in enumerate(np.unique(mask)):
        one_hot_mask[:, :, i][mask == unique_value] = 1
    
    return one_hot_mask
```

<h3>Converting to Tensor Format</h3>
For the PyTorch model, data needed to be converted into tensor format.

```python
import torch

def tensorize_image(image_array, cuda=False):
    # Convert the image to tensor format (C x H x W)
    torch_image = torch.from_numpy(image_array).permute(2, 0, 1).float()
    
    # GPU usage (optional)
    if cuda:
        torch_image = torch_image.cuda()
    
    return torch_image

def tensorize_mask(mask_array, cuda=False):
    # Convert the mask to tensor format (C x H x W)
    torch_mask = torch.from_numpy(mask_array).permute(2, 0, 1).float()
    
    # GPU usage (optional)
    if cuda:
        torch_mask = torch_mask.cuda()
    
    return torch_mask
```  
Images and masks were prepared and converted into tensor format to be fed into the PyTorch model. As a result, the data was ready to be loaded into the model for the segmentation process.   

Here's the section with the codes;
<a href="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/src/mask_on_image.py">preprocess.py</a>



<h2>Design Segmentation Model</h2>
<h4>What is the difference between CNN and Fully CNN (FCNN) ?</h4>

<img alt="Adsız" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/15.png">


CNN (Convolutional Neural Network) and FCNN (Fully Convolutional Neural Network) are both types of neural networks commonly used for image analysis and other tasks. However, there are differences between the two in terms of architecture and applications.

<h5>Convolutional Neural Network (CNN):</h5>

*A CNN consists of multiple layers, including convolutional layers, pooling layers, and fully connected layers.

*The convolutional layers are responsible for detecting features in the input image by applying convolution operations with learnable filters.
Pooling layers reduce the spatial dimensions of the features, helping to decrease the computational load and make the network more robust to variations in input.

*CNNs are commonly used for image classification, object detection, and other tasks where spatial hierarchies of features are important.

*The final layers of a CNN are typically fully connected layers that make predictions based on the extracted features.

<h5>Fully Convolutional Neural Network (FCNN):</h5>

*An FCNN is a type of CNN where all layers, including the final prediction layers, are convolutional layers.

*FCNNs are designed for tasks that involve pixel-wise prediction or semantic segmentation, where the goal is to assign a class label to each pixel in an image.

*FCNNs take an input image and produce an output feature map with the same spatial dimensions as the input image.

*The output feature map can then be upsampled using techniques like transposed convolutions to generate a dense prediction map.

*FCNNs are used in tasks such as semantic segmentation, image-to-image translation, and other tasks where maintaining spatial information is crucial.

In summary, the key difference between CNN and FCNN lies in their architecture and applications. CNNs are more general and suitable for tasks like image classification and object detection, while FCNNs are specialized for tasks involving pixel-wise predictions and semantic segmentation, where preserving spatial information is essential.



<h4>What are the different layers on CNN ?</h4>



A Convolutional Neural Network (CNN) typically consists of various types of layers that work together to process input data, extract features, and make predictions. Here are the main types of layers found in a CNN:

<b>Input Layer:</b> This layer takes raw input data, often an image. The dimensions of the input data are determined by the size and resolution of the images in the dataset.

<b>Convolutional Layer:</b> This is the fundamental layer of a CNN. It applies learnable filters (kernels) to the input image and computes dot products between the filter and local patches of the input. This process helps detect features like edges, corners, and textures.

<b>Activation Layer (ReLU):</b> Following each convolutional layer, an activation layer is typically applied, often using the Rectified Linear Unit (ReLU) activation. ReLU sets negative values to zero while leaving positive values unchanged.

<b>Pooling Layer (Pooling or Subsampling):</b> Pooling layers perform downsampling of the input's spatial dimensions. Common pooling methods include Max Pooling (selecting the maximum value from a group of values) and Average Pooling (computing the average value).

<b>Fully Connected Layer (FC Layer):</b> Also known as a dense layer, this layer is typically used at the end of a CNN. It connects each neuron from the previous layer to every neuron in the subsequent layer. It's used for making predictions.

<b>Flatten Layer:</b> This layer is used to transform the 2D output of previous convolutional and pooling layers into a 1D vector. This vector can then be fed into a fully connected layer.

<b>Batch Normalization Layer:</b> Batch Normalization is used to normalize the activations of the previous layer. It helps stabilize and speed up training by reducing internal covariate shifts.

<b>Dropout Layer:</b> Dropout involves temporarily "dropping out" random neurons during training. This encourages the network to learn more robust features by preventing overfitting.

These layers are typically stacked sequentially to form the architecture of a CNN. The specific arrangement and number of layers can vary based on the complexity of the task and design preferences. CNNs have demonstrated exceptional performance in visual tasks such as image classification, object detection, and image segmentation.


<img alt="Adsız" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/16.png">


<h4>What is activation function ? Why is softmax usually used in the last layer?</h4>

An activation function is a mathematical operation applied to the output of each neuron in a neural network. These functions help enhance the learning and representation capabilities of incoming data, allowing the neural network to understand more complex relationships.

Softmax, on the other hand, is an activation function typically used in the last layer of a neural network, especially for classification problems with multiple classes. It represents the contribution of each class to the output as a probability distribution. This helps us identify which class is more likely to give the correct result. The main reason for using Softmax is to make the outputs meaningful and interpretable, aiding in selecting the most probable class.

<h3>Model</h3>


<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/17.png">


In this project, the U-Net model was specifically employed. U-Net offers an ideal solution for semantic segmentation tasks and plays a central role in this project. Semantic segmentation is a crucial task that involves assigning each pixel in an image to a specific class. U-Net demonstrates excellent performance in such tasks and serves as a key component in this project.

One of the reasons for choosing U-Net in this project is its ability to achieve high success with limited training data. Furthermore, its lack of specific size requirements for input images makes it an ideal choice for use in this project.

Thanks to the U-Net model within the project, successful pixel-based classification of images has been accomplished.


<h4>U-Net for Image Segmentation</h4>
This repository contains an implementation of the U-Net architecture for image segmentation tasks. U-Net is a popular convolutional neural network (CNN) architecture that excels in tasks like semantic segmentation. In this project, we use the U-Net model to perform image segmentation, which involves classifying each pixel in an image into specific categories.

<h4>U-Net Architecture</h4>

The U-Net model is characterized by its encoder-decoder architecture, which consists of the following components:

<h4>Encoder Layers</h4>

The encoder layers are responsible for extracting essential features from the input image. This is achieved through a series of convolutional layers followed by ReLU activation functions. The code snippet below demonstrates the encoder layers in action:

```python
# Encoder Layers
x = self.dconv_down1(x)
x = self.maxpool(x)

x = self.dconv_down2(x)
x = self.maxpool(x)

x = self.dconv_down3(x)
x = self.maxpool(x)

x = self.dconv_down4(x)
```
<h4>Decoder Layers</h4>
The decoder layers aim to upsample the feature maps to the original image size and generate the segmentation output. The code snippet below illustrates how the decoder layers are implemented:

```python
# Decoder Layers
x = self.upsample(x)
x = torch.cat([x, conv3], dim=1)
x = self.dconv_up3(x)

x = self.upsample(x)
x = torch.cat([x, conv2], dim=1)
x = self.dconv_up2(x)

x = self.upsample(x)
x = torch.cat([x, conv1], dim=1)
x = self.dconv_up1(x)
```

<h4>Activation Functions</h4>
ReLU activation functions are applied to introduce non-linearity in the model and enhance feature representation. The following code snippet demonstrates the application of ReLU activation:

```python
x = F.relu(x)  # ReLU activation
```
<h4>Final Layer and Softmax</h4>
The final layer is a 1x1 convolutional layer that outputs the segmentation predictions. A softmax activation is applied to obtain class probabilities for each pixel:

```python
x = self.conv_last(x)
x = nn.Softmax(dim=1)(x)  # Apply softmax
```
Here's the section with the codes;
<a href="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/src/model.py">model.py</a>



<h2>Train</h2>  


<h3>What is parameter in NN ?</h3>  

A parameter in a neural network is a number that guides how the network works. For instance, weights and biases of the neurons are parameters that shape how input turns into output. These numbers get adjusted while the network learns to make its predictions more accurate.


<h3>What is hyper-parameter in NN ?</h3>  

A hyperparameter in a neural network is like a setting that guides how the network learns and how well it works. For instance, you can pick values for things like the number of hidden layers, the learning rate, and the activation function before you start training the network. These choices influence how quickly the network learns and how accurate its predictions become. Unlike the numbers the network learns during training (like weights and biases), hyperparameters are chosen by you to get the best results for a specific job.


<h3>We mention the dataset and we separate it into 2: training & test. In addition to them, there is a validation dataset. What is it for?</h3>  

A validation dataset is like a practice field for a machine learning model. It's a part of the dataset that's not used for training or testing. Instead, it helps you fine-tune how the model learns. For example, you can adjust important choices like hidden layers, learning speed, or activation style using this data. The goal is to make the model as smart as possible.
Comparing the model's performance on this validation data with its training performance shows if it's really learning or maybe just memorizing things. This helps to avoid mistakes like overthinking (overfitting) or not learning enough (underfitting). If the model's accuracy on the validation data starts getting worse, you might want to change how it learns.

<h3>What is epoch?</h3>  

An epoch is a term in machine learning that represents a complete cycle through the entire training dataset during the training process. It's like going through all the pages of a book one time. Each epoch helps the model learn and adjust its parameters based on the data it sees. The number of epochs is something you can decide before training, like choosing how many times to read the book to understand it better. Too few epochs might not capture all the patterns, while too many epochs can lead to memorizing the data instead of learning from it.

<h3>What is batch?</h3>  
A batch in machine learning is a small group of examples from the training dataset that are processed together. It's like learning from a few pages of a book at a time instead of reading the whole book in one go. Using batches helps speed up training and makes computations more efficient. Each batch contributes a little bit to adjusting the model's parameters. Think of it as learning in small chunks rather than trying to understand everything all at once.

<h3>What is iteration? Explain with an example: "If we have x images as data and batch size is y. Then an epoch should run z iterations."</h3>  

An iteration in machine learning is a single step where the model processes a batch of data to update its parameters. It's like reading a few pages of a book before making some notes.
Let's break down the example: "If we have x images as data and batch size is y. Then an epoch should run z iterations."  

If you have a total of x images in your training dataset and you're using batches of size y, then an epoch – which covers the whole dataset – should run z iterations. The number of iterations (z) is calculated by dividing the total number of images (x) by the batch size (y):  
<code>z = x / y</code>  
For instance, if you have 1000 images and you're using batches of 100, then an epoch should run 10 iterations because 1000 / 100 = 10. Each iteration would process a batch of 100 images, updating the model's parameters gradually as it goes through the entire dataset.


<h3>What Is the Cost Function?</h3>  

The cost function is a way to measure how wrong a machine learning model's predictions are compared to the actual outcomes. It calculates the difference between predicted and true values. The model's goal is to minimize this difference while training, so it gets better at making accurate predictions.

<h3>The process of minimizing (or maximizing) any mathematical expression is called optimization. What is/are the purpose(s) of an optimizer in NN?</h3>  

The purpose of an optimizer in a neural network is to guide the process of making the model better at its task. Optimization involves adjusting the model's internal parameters so that its predictions match the desired outcomes more accurately.

<h3>What is Batch Gradient Descent & Stochastic Gradient Descent? Compare them.</h3>  
<h5>Batch Gradient Descent (BGD):</h5>  
Batch Gradient Descent is a optimization technique used to update the model's parameters by considering the entire training dataset in each iteration. In BGD, the model computes the gradient of the cost function with respect to all training examples and then updates the parameters accordingly. This leads to more accurate gradient estimates, as it takes into account the global structure of the data. However, BGD can be computationally expensive, especially for large datasets, because it requires processing the entire dataset in each iteration.

<h5>Stochastic Gradient Descent (SGD):</h5>  
Stochastic Gradient Descent is a variation of gradient descent where the model updates its parameters using only one randomly selected training example in each iteration. This means that the parameter updates are more frequent and can lead to faster convergence. However, because each update is based on a single example, the gradient estimates can be noisy and have more variance. As a result, the optimization process can be more erratic, with a tendency to jump around the optimal solution.   

<b>Comparison:</b>

<b>Efficiency:</b>  
BGD processes the entire dataset, making fewer updates but with more accurate gradients.  
SGD processes one example at a time, leading to frequent updates but with noisy gradients.    
		
<b>Convergence:</b>  
BGD typically converges to the optimal solution with smoother, more gradual updates.  
SGD can converge faster due to frequent updates, but the path to convergence might be more erratic.  

<b>Noise:</b>  
BGD has smoother gradients because it considers the entire dataset, reducing the impact of noisy individual examples.  
SGD has noisy gradients due to its single-example updates, which can cause fluctuations in the optimization process.  

<b>Computational Complexity:</b>  
BGD can be computationally intensive, especially for large datasets, as it requires processing the entire dataset in each iteration.  
SGD is computationally more efficient since it only processes one example at a time, making it suitable for large datasets. 

<b>Generalization:</b>
BGD might generalize better as it considers the entire dataset and smooths out noise.
SGD can sometimes generalize less effectively due to the noisy updates, which might cause overfitting.
		
<b>Batch Size:</b>
Mini-batch Gradient Descent is a compromise between BGD and SGD. It processes a small subset (batch) of the dataset, offering a trade-off between smooth updates and computational efficiency.
  
In summary, Batch Gradient Descent processes the whole dataset, providing accurate gradients but can be slow. Stochastic Gradient Descent processes one example at a time, converging faster but with noisy gradients. The choice between them often depends on the dataset size, computational resources, and desired convergence behavior.


<h3>What is Backpropogation ? What is used for ?</h3>  

<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/18.png">

Backpropagation, a term derived from "backward propagation of errors," constitutes a core algorithm in training artificial neural networks. Its role is pivotal in guiding neural networks toward making accurate predictions by iteratively refining their internal parameters, namely weights and biases. The key mechanism behind backpropagation is the calculation of gradients for these parameters with respect to a chosen loss function.  

<h5>Purpose and Significance:</h5>  
Backpropagation bears immense significance as the keystone of neural network training. It equips the network with the capability to adapt its internal parameters based on observed discrepancies, thereby fostering continuous improvement in prediction accuracy. The very essence of machine learning hinges upon the ability to learn from data, and backpropagation offers the mechanism for neural networks to decipher complex patterns, generalize from training data to novel instances, and ultimately fulfill intricate tasks through the refinement of their parameters.

<h2>Training the Model</h2>
In this stage, a neural network model has been defined and trained for free space segmentation. Below, the fundamental steps and functions of the code are summarized:

Hyperparameter Definition: Important parameters were determined to manage the training process. These values control the duration of the training loop, learning rate, model size, and other essential settings.
```python
valid_size = 0.3
test_size = 0.1
batch_size = 4
epochs = 20
cuda = True
input_shape = (224, 224)
n_classes = 2
output_png_path = 'graphs/loss_graph.png'
```  
Preparing Data Lists: Paths to image and mask files were specified. At this stage, the dataset was divided into training, validation, and test data. Shuffling the data enables the model to adapt to different datasets.
```python
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()

image_mask_check(image_path_list, mask_path_list)

indices = np.random.permutation(len(image_path_list))

test_ind = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]
steps_per_epoch = len(train_input_path_list) // batch_size
```  

Model Creation and Optimization: A neural network model was defined, and its learning capability was enhanced through optimization processes. The model's weights and parameters were established.
```python
model = FoInternNet(input_size=input_shape, n_classes=n_classes)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

CUDA Usage: If available, GPU acceleration was utilized. This speeds up computations and improves performance, especially with larger datasets.
```python
if cuda:
    model = model.cuda()
```
Accuracy Calculation Function: A function was defined to assess the model's performance. This function calculates how closely the model's predictions align with the actual labels.
```python
def calculate_accuracy(outputs, labels):
    # ...
    return accuracy
```
Training Loop: The model was fed with training data to carry out the learning process. In each training loop, errors between model outputs and actual labels were corrected to update the model. 
```python
for epoch in tqdm.tqdm(range(epochs)):
    model.train()
    running_loss = 0.0
    running_accuracy = 0

    # ...
```

Validation Loop: During the training process, the model's generalization ability was tested using validation data. This step ensures that the model doesn't overfit during the learning process.
```python
val_loss = 0
val_accuracy = 0
model.eval()
with torch.no_grad():
    for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
        # ...
```
Graphing Results: The performance of both the training and validation processes was visualized using graphs. These graphs allow tracking the progress of the model's training and overall performance.
```python
def draw_graph_to_png(val_losses, train_losses, val_accuracies, train_accuracies, epochs, output_path):
    # ...

draw_graph_to_png(val_losses, train_losses, val_accuracies, train_accuracies, epochs, output_png_path)
```
Visualization and Saving Predictions: The trained model was used on test data to generate masked predictions of images. This stage visually demonstrates the impact of the model on real-world data.
```python
test_masked_path_list = glob.glob(os.path.join('data/masked_images', '*'))
test_masked_path_list.sort()

n_samples = 5
visualize_and_save_predictions(test_input_path_list, test_label_path_list, test_masked_path_list, n_samples)
```

At this stage's conclusion, the trained model was used to make predictions on test data, resulting in "masked" images. These images showcase the areas the model identified as free space based on real-world data. This project addresses the problem of free space segmentation, demonstrating the foundational steps and how the neural network model is constructed and trained.
<h2>Data Augmentation</h2>  

This piece of code helps you diversify your training data set using data amending methods. Data amputation is a common technique used to diversify the dataset in which your model is trained and help the model to generalize better.
```python
for image, mask in tqdm.tqdm(zip(train_input_path_list, train_label_path_list)):
    if os.path.isfile(image) and os.path.isfile(mask):
        img = Image.open(image)
        msk = cv2.imread(mask)
        color_aug = T.ColorJitter(brightness=0.4, contrast=0.4, hue=0.06)
        img_aug = color_aug(img)
        new_image_path = image[:-4] + "-1.png"
        new_mask_path = mask[:-4] + "-1.png"
        img_aug = np.array(img_aug)
        cv2.imwrite(new_image_path, img_aug)
        cv2.imwrite(new_mask_path, msk)
```
The data augmentation process resulted in the addition of data to the training set.
```python
aug_size = int(len(aug_mask_path_list) / 2)
train_input_path_list = aug_path_list[:aug_size] + train_input_path_list + aug_path_list[aug_size:]
train_label_path_list = aug_mask_path_list[:aug_size] + train_label_path_list + aug_mask_path_list[aug_size:]
```
<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/19.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/20.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/21.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/22.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/23.png"><img alt="Adsız" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/24.png">


<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/25.png">



<img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/26.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/27.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/28.png"><img alt="Adsız" src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/29.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/30.png"><img alt="Adsız"  src="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/img/31.png">






Here's the section with the codes;
<a href="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/src/train.py">train.py</a>



