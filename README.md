# 5-Device-Image-Classification

The purpose of this challenge is to develop a multiclass image classifier that returns five probabilities
of Device A or Device B or Device C or Device D or none of these devices in the images provided.

### Requirement
Following are the requirements of the challenge -
• Develop multi-class image classifier model to identify the devices in images.
• Train model on the provided images.
• Use the developed model to predict on new images and output the probabilities to ‘output.csv’ file.
• Convert the model to onnx format
• Document pre-processing and output details.

### Solution
• VGG-16 model has been used.
• The output layer of model has been modified to output 5 class probabilities – Device A, Device B,
Device C, Device D and no device.
The path to the training data, images for prediction and saved models are provided in ‘config.csv’
file. Any changes to the paths can be updated in this file.
• Training data has been augmented using the following approaches –
    • Training images provided in the forum
    • Extracting images from video files provided
    • Using keras function ImageDataGenerator
       (https://keras.io/preprocessing/image/#imagedatagenerator-class)
• The trained model is saved as ‘model.h5’ and ‘model.onnx’ in ‘saved_model’ folder.
• In order to predict the probability of the devices in new images, these images are to be placed in
the ‘predict’ folder and ‘predict.py’ file is to be run. The output file ‘output.csv’ is generated

### Technology Requirements
The solution is developed in Python 3.6. Following are the dependent packages -
1> Numpy 1.17.1
2> Keras 2.2.4 [Tensorflow backend]
3> scikit-learn 0.21.2
4> pandas 0.24.2
5> onnx 1.5.0
6> keras2onnx 1.5.1
The dependent packages are provided in ‘requirements.txt’
