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
The segmentation task involves identifying and delineating different objects or regions within data,typically images.It usally requires labeled data, making it a supervised task.

<h3>What is classification task in NN? Is it supervised or unsupervised?</h3>
The classification task assigns input data to specific categories or classes.It is also a supervised task that requires labeled data with inputs and corresponding target classes.

<h3>Compare segmentation and classification in NN.</h3>
Segmentation focuses on identifying regions or objects within data, while classification assigns data to predefined classes or categories. Segmentation is more detailed and deals with complex data structures, while classification is a more general task with simpler output.

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
RGB stands for Red,Green and Blue.It is an additive color model in which colors are created by combining different intensities of red,green and blue prmary colors.RGB is widely useed in digital imaging and displays.


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

