import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.preprocessing.image import load_img
from keras import optimizers
from keras import regularizers
from keras import layers
from keras import models
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.preprocessing.image import img_to_array
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
import keras2onnx
import onnx
import os
import keras
import glob
import json


CONFIG_JSON_PATH='./config.json'

# Dynamically allocate GPU memory
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.log_device_placement = True
sess = tf.Session(config=config)
set_session(sess)

print('[ INFO ] Compile model...')

## Our model structure
def build_model():
    ##  add VGG16 network to our Classification model
    conv_base=VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))
    conv_base.trainable = True
    set_trainable = False
    ##  fine-tuning
    for layer in conv_base.layers:
        if layer.name == 'block5_conv1':
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                    loss='binary_crossentropy',
                    metrics=['acc']
                )

    return model

## Read training picture
def read_train_picture(DeviceA_path, DeviceB_path, DeviceC_path, DeviceD_path, no_device_path):
    labels=[]
    data=[]
    for img_path in glob.glob(os.path.join(DeviceA_path,'*')):
        image=load_img(img_path,target_size=(150,150))
        image=img_to_array(image)
        data.append(image)
        labels.append((0,))
    for img_path in glob.glob(os.path.join(DeviceB_path,'*')):
        image=load_img(img_path,target_size=(150,150))
        image=img_to_array(image)
        data.append(image)
        labels.append((1,))
    for img_path in glob.glob(os.path.join(DeviceC_path,'*')):
        image=load_img(img_path,target_size=(150,150))
        image=img_to_array(image)
        data.append(image)
        labels.append((2,))
    for img_path in glob.glob(os.path.join(DeviceD_path,'*')):
        image=load_img(img_path,target_size=(150,150))
        image=img_to_array(image)
        data.append(image)
        labels.append((3,))
    for img_path in glob.glob(os.path.join(no_device_path,'*')):
        image=load_img(img_path,target_size=(150,150))
        image=img_to_array(image)
        data.append(image)
        labels.append((4,))
    labels=MultiLabelBinarizer().fit_transform(labels)

    data=np.array(data,dtype='float32')
    data=data/255.0
    labels=np.array(labels,dtype='float32')
    return data,labels


## Get training and prediction paths from config.json
def load_json(config_json_path):
    with open(config_json_path,'r') as json_file:
        datas=json.load(json_file)
    return datas



if __name__=='__main__':

    print('[ INFO ] Load models and data...')
    ## Get training and prediction paths from config.json
    paths=load_json(CONFIG_JSON_PATH)
    DeviceA_path=paths['DeviceA_path']
    DeviceB_path=paths['DeviceB_path']
    DeviceC_path=paths['DeviceC_path']
    DeviceD_path=paths['DeviceD_path']
    no_device_path=paths['no_device_path']
    predict_path=paths['predict_path']
    save_model_path=paths['save_model_path']

    data,labels=read_train_picture(DeviceA_path, DeviceB_path, DeviceC_path, DeviceD_path, no_device_path)

    ## Use image enhancements for training and test sets
    datagen= ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    model=build_model()

    ## Read training picture
    (train_x,test_x,train_y,test_y)=train_test_split(data,labels,test_size=0.2,random_state=2019)

    print('[ INFO ] Training model...')
    model.fit_generator(
        datagen.flow(train_x,train_y,batch_size=10),
        validation_data=(test_x,test_y),
        steps_per_epoch=170,
        epochs=16,
        )

    print('[ INFO ]  Save our model...')

    onnx_model=keras2onnx.convert_keras(model,model.name)

    model.save(os.path.join(save_model_path,'model.h5'))
    onnx.save_model(onnx_model,os.path.join(save_model_path,'model.onnx'))


    print('[ INFO ]  Model saved')
    images=[]
    picture_name=[]

    print('[ INFO ] Load the image to be recognized...')
    paths=glob.glob(os.path.join(predict_path,'*'))
    for pic_path in paths:
        img=load_img(pic_path,target_size=(150,150))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=img.astype('float32')
        img/=225.0
        images.append(img)
        picture_name.append(os.path.basename(pic_path))
    if len(images) >0:
        images=np.concatenate([i for i in images])

        print('[ INFO ]  Recognizing...')
        ans=model.predict(images)
        df=pd.DataFrame({'filename':picture_name})
        df['prob_device_A']=ans[:,0]
        df['prob_device_B']=ans[:,1]
        df['prob_device_C']=ans[:,2]
        df['prob_device_D']=ans[:,3]
        df['prob_no_device']=ans[:,4]
        df.to_csv('output.csv',index=False)
    else:
        print('[ INFO ]  There is no image in the predict file')
    print('[ INFO ] Finished...')
