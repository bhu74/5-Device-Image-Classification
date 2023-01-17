import glob
from keras.preprocessing.image import img_to_array,load_img
from keras.models import load_model
import os
import json
import numpy as np
import pandas as pd
CONFIG_JSON_PATH='./config.json'

def load_json(config_json_path):
    with open(config_json_path,'r') as json_file:
        datas=json.load(json_file)
    return datas



def predict():
    path_info=load_json(CONFIG_JSON_PATH)

    save_csv_path=path_info['save_csv_path']
    predict_path=path_info['predict_path']
    save_model_path=path_info['save_model_path']
    
    images=[]
    picture_name=[]
    paths=glob.glob(os.path.join(predict_path,'*'))
    for pic_path in paths:
        img=load_img(pic_path,target_size=(150,150))
        img=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        img=img.astype('float32')
        img/=225.0
        images.append(img)
        picture_name.append(os.path.basename(pic_path))

    if len(images)==0:
        print('[ INFO ]  There is no image in the predict file')
        return 0
    images=np.concatenate([i for i in images])
    model=load_model(os.path.join(save_model_path,'model.h5'))
    ans=model.predict(images)

    df=pd.DataFrame({'filename':picture_name})
    df['prob_device_A']=ans[:,0]
    df['prob_device_B']=ans[:,1]
    df['prob_device_C']=ans[:,2]
    df['prob_device_D']=ans[:,3]
    df['prob_no_device']=ans[:,4]
    
    save_csv_path=os.path.join(save_csv_path,'output.csv')
    df.to_csv(save_csv_path,index=False)
    print('[ INFO  ]  Finished...')

if __name__=='__main__':
    predict()



