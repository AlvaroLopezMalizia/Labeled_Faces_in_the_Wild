# Labeled_Faces_in_the_Wild

This final project on Deep Learning utilizes the "Labeled Faces in the Wild (LFW)" dataset through the `fetch_lfw_people` function from scikit-learn. Here's a summary of the main actions taken in the project:

1. **Loading Initial Data:**
   - The `fetch_lfw_people` function is used to obtain images of people's faces.
   - Only images of 7 people with more than 70 available images are selected.
   - Inspection of the sizes of the images is performed.

2. **Initial Visualization:**
   - A `plot_gallery` function is defined to visualize a gallery of portraits.
   - 12 images are displayed with their respective labels.

3. **Data Preparation:**
   - The dataset is split into training and testing sets (80:20).
   - The total number of classes is checked, and some values are inspected.
   - Labels are converted to one-hot encoding format.

4. **Convolutional Neural Network Model:**
   - TensorFlow and Keras are used to build a convolutional neural network model.
   - The architecture includes Conv2D, MaxPooling2D, Dropout, Flatten, and dense layers.
   - Categorical cross-entropy loss function and Adadelta optimizer are used.

5. **Model Training:**
   - The model is trained on training data and validated on test data.
   - The evolution of accuracy and mean squared error (MSE) is shown over epochs.
   - Possible overfitting is observed after approximately 15 epochs.

6. **Model Evaluation:**
   - A personal photo is loaded, displayed in grayscale, and resized to 62x47 pixels.
   - The image is normalized and fed into the trained model to predict its class.
   - The predicted class is compared with images of the same class in the dataset.

7. **Results:**
   - The predicted class for the personal image is displayed.
   - Some images of the predicted class in the dataset are visualized.

This summary highlights the main stages of the project, from data loading to model evaluation with a personal image. The model appears to perform well in classifying faces.
