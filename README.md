# FordOtosan_Internship
<h1>Freespace Segmentation with Fully Convolutional Neural Network (FCNN)</h1>


<h3>What is Machine Learning?</h3>
It is a branch of artifical intelligence where computer systems learn and perform tasks by analyzing data and using their experiences.Machine learning algorithms and statistical models are used to analysze data,discover patterns,relationships and extract meaningful information.

![image](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/4080a23f-f029-46c6-8c6a-0ccb02e4c6f5)


<h3>What is Unsupervised vs Supervised learning difference?</h3>
The main difference is whether the training data has labels or not. In supervised learning, the data includes inputs and corresponding labels. In unsupervised learning, the data is unlabeled, and the focus is on discovering structures and patterns within the data.

![image](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/65ef8ba3-c77d-4bde-bb19-01a3047f243a)

<h3>What is Deep Learning?</h3>
 It is a subfield of machine learning that uses mathematical models called artificial neural networks to perform complex tasks.Deep learning methods automatically learn and interpret complex structures in data by leveraging datasets.

<h3>What is Neural Network (NN)?</h3>
 It's a mathematical model that imitates the structure of the human brain. It uses interconnected artificial neurons to handle input data and generate outputs.Neural networks are widely used in machine learning.
<br>
<img width="300" alt="Adsız" src="https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/f653aa46-0546-4a43-bd73-dee9f266c0fc">


<h3>What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.</h3>
It is a special kind of neural network made for working with structured data,particularly images.CNNs are widely used in deep learning for tasks like sorting images into categories,recognizing objeects and dividing images into parts.CNNs have two benefits over regular neural networks:they can automatically learn important features from the data and they share parameters between different parts of the network which make them more efficient.

![CNN](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/06901603-f28e-4aba-b456-1f46eb027555)



<h3>What is segmentation task in NN? Is it supervised or unsupervised?</h3>
The segmentation task involves identifying and delineating different objects or regions within data,typically images.It usually requires labeled data, making it a supervised task.

<h3>What is classification task in NN? Is it supervised or unsupervised?</h3>
The classification task assigns input data to specific categories or classes.It is also a supervised task that requires labeled data with inputs and corresponding target classes.

<h3>Compare segmentation and classification in NN.</h3>
Segmentation focuses on identifying regions or objects within data, while classification assigns data to predefined classes or categories. Segmentation is more detailed and deals with complex data structures, while classification is a more general task with simpler output.
<br>
<img width="401" alt="Adsız 4" src="https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/ffa6e785-de41-46ae-88ac-58d7e7991dbc">


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


<img width="434" alt="jsonfile" src="https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/2ec33c1d-7966-46b0-aaf1-18f36feafabe">

The masks are created using these exterior and interior points. Masks are black and white images representing pixels. Pixels belonging to the "Freespace" class are represented as white (e.g., pixel value 255), while other pixels are represented as black (e.g., pixel value 0). This is done using the information from the JSON files.

Masks were drawn using the Python cv2 library and the fillPoly function.
```python
mask = cv2.fillPoly(mask, np.array([obj['points']['exterior']]), color=1)
```
<h4>Mask example;</h4>


<img width="450" alt="Adsız 6" 
src=https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/9438aa82-4c0f-45a4-bc63-81c72140180c)/>
<img width="450" alt="Adsız 5" 
src=https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/311162e4-c96b-4b8a-818a-8158d6baa981)/>
<img width="900" alt="mask" src="https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/c558cd78-65a2-4ceb-8190-1e67b5eb1283">

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
<img width="779" alt="Adsız 2" src="https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/e1597a9e-e09b-4113-b1b9-0e5c55a16633">

Here's the section with the codes;
<a href="https://github.com/aysuaticioglu/FordOtosan_Internship/blob/main/src/mask_on_image.py">mask_on_image.py</a>


<h2>Converting into Tensor</h2>
<h4>Explain Computational Graph.</h4>
	A Computational Graph is a graphical representation method that shows mathematical operations and their intermediate results. It is an essential concept used to understand the training and backpropagation processes of deep learning and machine learning models.
In a Computational Graph, all the mathematical operations and intermediate values from the input data to the output data are represented as nodes and the relationships between them are represented as directed edges.
This graph illustrates how data flows during the forward pass and how gradients are computed during backpropagation. It provides insights into the computations performed by the model during the training process and helps visualize how parameters are updated.
Computational Graphs are valuable tools to gain a deeper understanding of the mathematical structure of deep learning models and make them more interpretable and traceable. They play a crucial role in improving the model's performance and understanding and correcting errors.


<h4>What is Tensor?</h4>
A Tensor is a mathematical concept used in scientific and computational calculations. It can be thought of as a generalization of a vector and a matrix.

In simple terms, a Tensor represents data in multi-dimensional arrays or matrices. A scalar is the simplest form of a Tensor and contains only one number. A vector is a Tensor that consists of a sequence of n numbers. A matrix can be thought of as a table with rows and columns.

However, a Tensor can have three or more dimensions. For example, a three-dimensional Tensor is used to represent a colored image (height x width x color channels). Video data can be represented using four-dimensional Tensors (sequence of frames x height x width x color channels).

Tensors are widely used in areas like deep learning and artificial intelligence. In these fields, input and output data in neural network models, as well as weights and gradients, are represented as Tensors, and various mathematical operations are performed on these Tensors.

In conclusion, Tensors are essential for representing data in different dimensions and structures and performing mathematical operations. They are widely used in various fields, from scientific calculations to artificial intelligence applications.
￼

![image](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/cd29249b-e405-446b-ae83-043a7f723711)





<h4>What is one hot encoding?</h4>
One Hot Encoding is a method used in machine learning and deep learning models to represent categorical data in a numerical format. In this method, each category is represented as an array (vector), where only the element corresponding to the category is "1" and the other elements are "0". This way, categorical data is converted into a numerical format.
For example, if we have three different color types: “red”, “green” and “blue, we can apply One Hot Encoding as follows:

* One Hot Encoding for “red”: [1, 0, 0]
* One Hot Encoding for “green”: [0, 1, 0]
* One Hot Encoding for “blue: [0, 0, 1]

  
This transformation allows machine learning models to process and classify categorical data more effectively. It also emphasizes that there is no inherent order or ranking among the categories.
￼

![url](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/b3442e5e-63d8-4e33-925e-df24f44ffdab)


<h4>What is CUDA programming? Answer without detail.</h4>
CUDA is a technology developed by NVIDIA. This technology allows graphics cards, which are typically used to accelerate graphics tasks, to also perform tasks involving mathematical calculations rapidly. In other words, computers can think and process tasks faster.

<h2>Design Segmentation Model</h2>
<h4>What is the difference between CNN and Fully CNN (FCNN) ?</h4>

![image](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/0ed85221-b63d-4fff-bfd4-b3a697dbab6b)


CNN (Convolutional Neural Network) and FCNN (Fully Convolutional Neural Network) are both types of neural networks commonly used for image analysis and other tasks. However, there are differences between the two in terms of architecture and applications.

<h5>Convolutional Neural Network (CNN):</h5>

A CNN consists of multiple layers, including convolutional layers, pooling layers, and fully connected layers.

The convolutional layers are responsible for detecting features in the input image by applying convolution operations with learnable filters.
Pooling layers reduce the spatial dimensions of the features, helping to decrease the computational load and make the network more robust to variations in input.

CNNs are commonly used for image classification, object detection, and other tasks where spatial hierarchies of features are important.

The final layers of a CNN are typically fully connected layers that make predictions based on the extracted features.

<h5>Fully Convolutional Neural Network (FCNN):</h5>

An FCNN is a type of CNN where all layers, including the final prediction layers, are convolutional layers.

FCNNs are designed for tasks that involve pixel-wise prediction or semantic segmentation, where the goal is to assign a class label to each pixel in an image.

FCNNs take an input image and produce an output feature map with the same spatial dimensions as the input image.

The output feature map can then be upsampled using techniques like transposed convolutions to generate a dense prediction map.

FCNNs are used in tasks such as semantic segmentation, image-to-image translation, and other tasks where maintaining spatial information is crucial.

In summary, the key difference between CNN and FCNN lies in their architecture and applications. CNNs are more general and suitable for tasks like image classification and object detection, while FCNNs are specialized for tasks involving pixel-wise predictions and semantic segmentation, where preserving spatial information is essential.



<h4>What are the different layers on CNN ?</h4>



A Convolutional Neural Network (CNN) typically consists of various types of layers that work together to process input data, extract features, and make predictions. Here are the main types of layers found in a CNN:

Input Layer: This layer takes raw input data, often an image. The dimensions of the input data are determined by the size and resolution of the images in the dataset.
Convolutional Layer: This is the fundamental layer of a CNN. It applies learnable filters (kernels) to the input image and computes dot products between the filter and local patches of the input. This process helps detect features like edges, corners, and textures.

Activation Layer (ReLU): Following each convolutional layer, an activation layer is typically applied, often using the Rectified Linear Unit (ReLU) activation. ReLU sets negative values to zero while leaving positive values unchanged.
Pooling Layer (Pooling or Subsampling): Pooling layers perform downsampling of the input's spatial dimensions. Common pooling methods include Max Pooling (selecting the maximum value from a group of values) and Average Pooling (computing the average value).

Fully Connected Layer (FC Layer): Also known as a dense layer, this layer is typically used at the end of a CNN. It connects each neuron from the previous layer to every neuron in the subsequent layer. It's used for making predictions.

Flatten Layer: This layer is used to transform the 2D output of previous convolutional and pooling layers into a 1D vector. This vector can then be fed into a fully connected layer.

Batch Normalization Layer: Batch Normalization is used to normalize the activations of the previous layer. It helps stabilize and speed up training by reducing internal covariate shifts.

Dropout Layer: Dropout involves temporarily "dropping out" random neurons during training. This encourages the network to learn more robust features by preventing overfitting.

These layers are typically stacked sequentially to form the architecture of a CNN. The specific arrangement and number of layers can vary based on the complexity of the task and design preferences. CNNs have demonstrated exceptional performance in visual tasks such as image classification, object detection, and image segmentation.


![image](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/dc6d9cc6-c54f-4028-95b2-edba3d78312c)


<h4>What is activation function ? Why is softmax usually used in the last layer?</h4>

An activation function is a mathematical operation applied to the output of each neuron in a neural network. These functions help enhance the learning and representation capabilities of incoming data, allowing the neural network to understand more complex relationships.

Softmax, on the other hand, is an activation function typically used in the last layer of a neural network, especially for classification problems with multiple classes. It represents the contribution of each class to the output as a probability distribution. This helps us identify which class is more likely to give the correct result. The main reason for using Softmax is to make the outputs meaningful and interpretable, aiding in selecting the most probable class.









