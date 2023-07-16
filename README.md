# FordOtosan_Internship
Freespace Segmentation with Fully Convolutional Neural Network (FCNN)
![machine-learning1-1](https://github.com/aysuaticioglu/FordOtosan_Internship/assets/75265305/0e7d761b-5c6b-4e54-884f-dd4f2d64fbf0)
* What is Machine Learning?
  It is a branch of artifical intelligence where computer systems learn and perform tasks by analyzing data and using their experiences.Machine learning algorithms and statistical models are used to analysze data,discover patterns,relationships and extract meaningful information.
* What is Unsupervised vs Supervised learning difference?
* What is Deep Learning?
  It is a subfield of machine learning that uses mathematical models called artificial neural networks to perform complex tasks.Deep learning methods automatically learn and interpret complex structures in data by leveraging datasets.
* What is Neural Network (NN)?
* What is Convolution Neural Network (CNN)? Please give 2 advantages over NN.
* What is segmentation task in NN? Is it supervised or unsupervised?
* What is classification task in NN? Is it supervised or unsupervised?
* Compare segmentation and classification in NN.
* What is data and dataset difference?
* What is the difference between supervised and unsupervised learning in terms of dataset?
Data Preprocessing
Extracting Masks
* What is color space ?
* What RGB stands for ?
* In Python, can we transform from one color space to another?
* What is the popular library for image processing?
In this part of the project, we want you to convert every JSON file into mask images:
1. Move json files into data/jsons folder.
2. Open the src folder and run json2mask.py There is a video explaining the code: Please click for English Lecture. Please click for Turkish Lecture. For those who want to follow the steps one by one:
    1. You need to move your files into data/jsons folder.
    2. You need to create a list which contains every file name in jsons folder.
    3. In a for loop, you need to read every json file and convert them into json dictionaries.
    4. You need to get width and height of image.
    5. You need to create an empty mask which will be filled with freespace polygons.
    6. You need to get objects in the dictionary, and in a for loop, you need to check the objects 'classTitle' are 'Freespace' or not.
    7. If it is a Freespace object, then you need to extract 'points' then 'exterior' of points which is a point list that contains every edge of polygon you clicked while labeling.
    8. You need to fill the mask with the array.
    9. You need to write mask image into data/masks folder.
3. cd src/
4. python3 json2mask.py
5. 
6. To check mask files, run mask_on_image.py There is a video explaining the code: Please click for Turkish Lecture. python3 mask_on_image.py
7. 
8. What do you think these masks will be used for? (Feature ? Label ?)
Converting into Tensor
Before go on, please search and answer following questions:
* Explain Computational Graph.
* What is Tensor?
* What is one hot encoding?
* What is CUDA programming? Answer without detail.
The images and masks refer to "features" and "labels" for Segmentation. To feed them into the Segmentation model, which will be written in PyTorch, we need to format them appropriately. In this part, we will solve this issue. In the preprocess.py There is a video explaining the code: Please click for Turkish Lecture. There are two helper functions:
1. To convert images to tensor, we need tensorize_mask(.). For this complete torchlike_data(.)
2. To convert masks to tensor, we need tensorize_mask(.). For this, complete one_hot_encoder(.)
At the end of the task, your data will be ready to train the model designed. We can use these functions!
The above operations are mandatory for our task. In addition to these, other preprocessing techniques can be performed.
Design Segmentation Model
Before go on, please search and answer following questions:
* What is the difference between CNN and Fully CNN (FCNN) ?
* What are the different layers on CNN ?
* What is activation function ? Why is softmax usually used in the last layer?
There is a script to design our model: model.py. In this script, we could program our model. This will require research. Below are the videos we think can help you:
* Create tensors in PyTorch + Matrix Multiplication
* Forward Pass
* Backpropogation
* Calculating Gradients
* Build NN & model.py
To visualize your model at the end, you can use this website.
Train
Before go on, please search and answer following questions:
* What is parameter in NN ?
* What is hyper-parameter in NN ?
* We mention the dataset and we separate it into 2: training & test. In addition to them, there is a validation dataset. What is it for?
* What is epoch?
* What is batch?
* What is iteration? Explain with an example: "If we have x images as data and batch size is y. Then an epoch should run z iterations."
* What Is the Cost Function?
* The process of minimizing (or maximizing) any mathematical expression is called optimization. What is/are the purpose(s) of an optimizer in NN?
* What is Batch Gradient Descent & Stochastic Gradient Descent? Compare them.
* What is Backpropogation ? What is used for ?
We prepare a train.py script that combines all our work & techniques mention in the questions. Play with hyper-parameters and examine their effects! Enjoy 🙂

