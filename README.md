# Image Classification Project

This project implements a basic image classification system utilizing a pre-trained TensorFlow model to predict the content of input images. The model is capable of classifying images into categories defined by a `labels.txt` file, such as "mars" or "saturn."

## Project Structure

The project directory is organized as follows:

-   `converted_savedmodel/`: This directory contains the pre-trained TensorFlow model.
    -   `labels.txt`: Defines the classification categories.
    -   `model.savedmodel/`: Houses the TensorFlow SavedModel, including its architecture and trained weights.
-   `predict.py`: The primary Python script for performing image classification.
-   `test.jpg`: A sample image provided for testing the prediction functionality.

## Functional Overview

The `predict.py` script executes the following steps:

1.  **Model Loading:** The pre-trained TensorFlow model is loaded from `converted_savedmodel/model.savedmodel` using `TFSMLayer`.
2.  **Image Preprocessing:** The input image (`test.jpg`) is opened, resized to 224x224 pixels, and normalized by scaling its pixel values to the range [0, 1]. This standardization is essential for compatibility with the model's input requirements.
3.  **Prediction Execution:** The preprocessed image is then passed to the loaded model for inference. The model outputs raw probabilities corresponding to each defined classification category.
4.  **Result Interpretation:** The script identifies the class with the highest probability as the predicted category and displays both the predicted class index and the full probability distribution.

## Setup and Execution

To run this project, follow these instructions:


  **Install Dependencies:** Install the required Python libraries using pip:
    ```bash
    pip install tensorflow Pillow numpy
    ```


The script will process `test.jpg` and display the classification result in the console. Users may replace `test.jpg` with their own images, ensuring the filename in `predict.py` is updated accordingly.