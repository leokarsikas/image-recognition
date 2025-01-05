# Image recognition of birthmarks to detect skin Cancer. Based on HAM10000 dataset
Project for the course in AI (ID1214) focusing on image recognition on birthmarks to detect skin cancer. The repository contains a demo, program to train a model, and already trained models. The dataset given in the instructions below are to an altered version of the original HAM10000 dataset, which is more balanced. 

## Instructions


### Training model
1. Go to: https://www.kaggle.com/datasets/utkarshps/skin-cancer-mnist10000-ham-augmented-dataset and download the dataset. 

2. Place the **base_dir** inside of the **training_model** folder. 

3. Run the `requirements.txt` script: 

```
python requirements.txt
```

4. Go to **training_model** folder and run the `trainingmodel.py` program to train the model. 

```
python trainingmodel.py
```

### Demo 

1. Go to **demo** and open `prototype`. On line **12** in 

```
model_path=os.path.abspath(os.path.join("..", "trained_models" , "choosen_model.pth")) #choose model path 
```

choose the model from the **trained models** folder or test your trained model. <ins> Don't forget to change the path in that case </ins>.

2. On line **69** change the  

```
image_path = os.path.abspath(os.path.join("test_images" ,"choose_image.jpg")) #choose image path
```

Choose the image from **test_images** folder by putting the name of the image in `choose_image.jpg`. 

3. Run the `prototype.py` program: 

```
python prototype.py 
```
