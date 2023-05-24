# ASSIGNMENT 4 - Self assigned project
Training a classifier on coral reef dataset 

## DESCRIPTION 
The goal with this project is to train a classifier on a dataset of images of 923 corals. The purpose is to train a model to distinguish between healthy and bleached coral. The bleaching of corals is caused by rising water temperatures, and while bleached corals are not dead they are under more stress and are subject to mortality (National Oceanic and Atmospheric Administration, 2020). Bleached corals can recover over time, which is why it is important to monitor their development and health. Machine learning models can help with that task.
The dataset used for this project and details was acquired from Kaggle: https://www.kaggle.com/datasets/vencerlanz09/healthy-and-bleached-corals-image-classification?select=bleached_corals 

## METHODS
Firstly, the data is cleaned by removing any corrupted images. To explore and learn more about the dataset, a bar plot is created to visualize the two categories, which shows an uneven number of healthy and bleached coral images. 
The pretrained neural network VGG16 is initialized, and the data is split into train and test sets using data generator with data augmentation. The data is then fed to the pretrained model using ```model.fit``` and a classification report is generated. Additionally a loss and accuracy plot is created and saved in the out folder. 

## HOW TO INSTALL AND RUN THE PROJECT
Initialize this repository:
1. First you need to clone this repository 
2. Your working repository should look like this:
    self-assigned-project-visual
    ├── data
    | └── bleached_corals
    | └── healthy_corals
    ├── out

    ├── src
    | └── corals_classfication.py
    README.md
    requirements.txt
    setup.sh

Since I have already executed the script, the ´´´out´´´ folder will contain the final results: classification_report.txt, loss_accuracy_curve_corals.png, visualizing_dataframe.png

Install packages:
3. Navigate from the root of your working directory to ´´´self-assigned-project-visual´´´
4. Run the setup file, which will install all the requirements by writing ´´´bash setup.sh´´´ in the terminal

Run the script:
5. Navigate to the folder ´´´src´´´ of this projects repository by writing ´´´cd src´´´ in the terminal, assuming your current directory is **self-assigned-project-visual**
6. Run the script by writing ´´´python corals_classification.py´´´ in the terminal

After running the script, the results are saved in the ´´´out´´´ folder.

## Discussion of results
**Dataset by category**
The number of bleached coral images is slightly higher than the number of healthy coral images.

**Classification report**
The classification report shows that the classifier is performing well at identifying bleached corals, however it is unable to identify healthy corals. Initially I though that the there was a problem with reading the classes, however the imageDataGenerator is able to identify 2 classes and support also shows 87 instances of “healthy_corals”. The issue might be in my approach and the choice of classifier. In visualizing_dataframe.png it also becomes evident that there is a slight difference in the number of images for bleached and healthy corals, which might cause imbalance in class distribution.

**Loss and accuracy curve**
Ideally the loss curve would decrease and the gap between train and validation loss would close. However, this is not the case for this classifier. The gap between both curves remains, which might indicate that the training dataset is too small relative to the validation dataset (Brownlee, 2019).

## Literature
Brownlee, J. (2019). A Gentle Introduction to Learning Curves for Diagnosing Machine Learning Model Performance. [online] Machine Learning Mastery. Available at: https://machinelearningmastery.com/learning-curves-for-diagnosing-machine-learning-model-performance/.

National Oceanic and Atmospheric Administration (2020). What is coral bleaching? [online] oceanservice.noaa.gov. Available at: https://oceanservice.noaa.gov/facts/coral_bleach.html 