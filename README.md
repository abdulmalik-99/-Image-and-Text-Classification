# Image and Text Classification Projects


##  Project 1: Arabic Poetry Classifier

### 1.1 Introduction
Arabic poetry is the oldest and most prominent form of Arabic literature today. Ancient Arabic poetry is probably the primary source for describing the social, political, and intellectual life in the Arab world. Modern poetry has gone through major changes and shifts, both in the form and in the topics. that led me to build an Arabic Poetry Classifier model.
In this project, a classification model is built to classify Arabic poetry based on the poet's origin.


### 1.2 Dataset 
The dataset contains around 11K sample of poems that extend from the 6th century to the present day. This dataset consist of 9 features and 11604 instance. In addition,it included 11594 poems of 591 poets.The total number of words was 1741848(before pre-processing)

### 1.3 Model 
To clean the dataset, i applied techniques to remove irrelevant and unnecessary information from the raw text data and convert the text into a normalized format. After cleaning the data, i implemented NLP processes such as tokenization, stop word removal, and stemming. One significant challenge i faced was imbalanced classes, which could affect model performance. i explored two approaches to deal with this problem.

First, i used the Random Forest algorithm, which is less sensitive to imbalanced data. i then applied the SMOTE (Synthetic Minority Over-sampling Technique), a commonly used oversampling method in machine learning.

Second, i divided the dataset into sub-data and deployed it into sub-models, which is a methodology for handling imbalanced data.

<details open>
<summary>Install</summary>

```
pip install -r requirements.txt
```

</details>

<details open>
<summary>Usage</summary>

I have built a web for testing Arabic Poetry Classifier...
**Change the port number if it does not work. ** 
```
python Text_Web.py
```
</details>


##  Project 2: Facial Expression Recognization

## 2.1 Introduction

Facial expression recognition is a fascinating field of computer vision that involves training machines to identify human emotions based on facial cues. This technology has applications in various domains, such as security, marketing, healthcare, and entertainment. The project involves using deep learning techniques to develop a facial expression recognition model that can accurately identify emotions such as happiness, sadness, anger, surprise, fear, and disgust.

### 2.2 Dataset 
This dataset consists of over 35,000 grayscale images of faces that are labeled with one of seven emotions: anger, disgust, fear, happiness, sadness, surprise, and neutral. The images are in various resolutions and were collected from the internet.

### 2.3 Model 
To preprocess the images, i applied multiple techniques such as rescale, shear range, and zoom range to scale the pixel values, distort the shape of the images, and zoom in or out of the images, and then i built a Convolutional Neural Network (CNN) model because it is a powerful and effective deep learning architecture for image classification.

One significant challenge i faced was imbalanced classes, which could affect the model's performance. To address this, i built three different models and selected the most accurate one.

Model 1: Trained the model without additional techniques such as transfer learning.

Model 2: Trained the model with a pre-trained model on VGG19 architecture.

Model 3: Used data augmentation techniques to handle imbalanced data. Data augmentation is one of the most common methodologies to reduce imbalanced data issues.


<details open>
<summary>Usage</summary>

I have built a web for testing Arabic Poetry Classifier...
** Change the port number if it does not work. ** 
```
python Image_Web.py
```
</details>

** Note ** you can get the models weights through this link https://drive.google.com/drive/folders/1SDZwp34qpolcAc15_AX6vwI8YFDx8F1V?usp=share_link 


