
# Usage Tutorial

 1. ### The data set is stored as follows
```
my_solution/  
    predict/  
        pic1.jpg  
        pic2.jpg  
        ...
    train_data/ 
        new_device/
            pic1.jpg
            pic2.jpg
            ...
        Device_A/
            pic1.jpg  
            pic2.jpg
            ...
       Device_B/
            pic1.jpg  
            pic2.jpg
            ...
        Device_C/
            pic1.jpg  
            pic2.jpg
            ...
        Device_D/
            pic1.jpg  
            pic2.jpg
            ...
        no_device/
            pic1.jpg
            pic2.jpg
     main.py  
     predict.py  
     config.json  
     output.csv  
     README.md  
     requirements.txt
```

 2. ### Description 
    
   - `main.py`is the main program of the training model, and the model will be saved after the training ends.
    - `predict.py` is to use the already trained model to identify new images.
    - 'Config.json' saves the path required by the program. When the above file path changes, you can modify it to suit your needs.
    - Output.csv is the result of running main.py or predict.py
    - Folder `predict/` is the folder to place images to be recognized
    - The folder `train_data/` contains the provided images that are used for training the model. Images have also been extracted from the video files and added to the training data. 
    - `requirements.txt` is the required Python package, you need to use `pip3 install -r requirements.txt` to install. The Python version is 3.7.

3. ### Use the trained model
In order to train the model, run `main.py`. If you want to identify the image directly, put the image you want to recognize in the `predict/` folder, then run `predict.py`. An already trained model is saved it in the save_model folder, you can identify it directly.


