##**This is the implementation of CSAE on Pytorch**

#### 1. Download the project

     git clone https://github.com/anlongstory/CSAE.git

#### 2. Extract the files the dataset in 'data' folder

**Ubuntu:**

    cd path/to/CVAE/data
    unzip MNIST_img.zip
    unzip letter.zip

**Windows:**

unzip directly

#### 3. Generate Predefine evenly-distribute class centroids

    python PEDCC.py

#### 4. Train CSAE

    python main.py

#### 5. Test Model
    python inference_model.py

**Tip: Here, we just take MNIST as example, you can change some parameters   or paths in `config.py`, and chang dataset in `data_transform.py`**


**If you have any question, please feel free to contact me :)**