# Potato Leaf Disease Classification
### Dataset & Preprocessing:
The dataset is split into training, validation, and testing sets. Images are resized, normalized, and augmented (rotation, zoom, flip) to improve model performance.
### Model Architecture:
A Convolutional Neural Network (CNN) built using TensorFlow/Keras with multiple layers for feature extraction and classification.
### Training: 
The model is trained with early stopping and model checkpoints to prevent overfitting and save the best model.
### Streamlit App:
The trained model is deployed in a Streamlit app where users can upload leaf images for real-time classification, displaying the result and confidence score.
