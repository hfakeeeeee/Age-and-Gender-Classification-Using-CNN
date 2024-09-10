## Age-and-Gender-Classification-Using-CNN

This project implements a deep learning model to classify age and gender using the Adience benchmark dataset. The model is built using Convolutional Neural Networks (CNN) and is trained to predict age and gender from images.

## Features

- **Data Loading**: Combines multiple folds of the Adience dataset for training.
- **Data Visualization**: Visualizes the age distribution within the dataset.
- **Model Architecture**: Implements a CNN with convolutional, pooling, normalization, and dense layers for age and gender classification.
- **Model Training**: Trains the model with early stopping to prevent overfitting, evaluating performance using validation data.
- **Image Processing**: Utilizes image data preprocessing techniques such as resizing and normalization.
- **Prediction**: Supports predicting age and gender from input images.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/hfakeeeeee/Age-and-Gender-Classification-Using-CNN.git
   ```

2. Navigate to the project directory:
```bash
cd /Age-and-Gender-Classification-Using-CNN
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses the [Adience Benchmark Gender and Age Classification dataset](https://datasets.activeloop.ai/docs/ml/datasets/adience-dataset/). Make sure the dataset is properly formatted and placed in the project directory.

## Usage
### Model Training
- Data Preparation: Place the dataset files in the respective directories.
- Run the Jupyter Notebook: Open the main.ipynb notebook and run all the cells to:
    - Load the dataset
    - Visualize the data
    - Train the CNN model
    - Evaluate the model's performance
## Prediction
Use the provided predict.py script to predict age and gender from an image:
```bash
python predict.py
```
Ensure you have the trained models (age_model.h5 and gender_model.h5) in the same directory as predict.py. Update the image filename in the script to test your own image.

## Example:
### In predict.py
```console
image = cv2.imread('test_image.jpg')
```

## Dependencies
- Python 3.x
- TensorFlow
- Keras
- Pandas
- NumPy
- OpenCV
- Matplotlib
- Seaborn