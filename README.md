# 🍎 Fruit & Vegetable Freshness Classifier

This repository contains the final code for my Pattern Recognition project. It compares the performance of a Support Vector Machine (SVM) and a Convolutional Neural Network (CNN) in classifying images of produce as either "Fresh" or "Rotten".

## 📊 Project Overview
* **Goal:** Automate the quality inspection of fruits and vegetables using Deep Learning.
* **Models Used:** * **SVM (SGDClassifier):** Used as a baseline for traditional machine learning performance.
  * **CNN (Custom Architecture):** Built with TensorFlow/Keras, utilizing Dropout and MaxPooling for optimized pattern recognition.
* **Environment:** Developed and trained using Google Colab.

* **Dataset:** [Download the Fruit & Vegetable Dataset Here](https://www.kaggle.com/datasets/muhriddinmuxiddinov/fruits-and-vegetables-dataset)

## 🚀 How to Run the Code
Because this project utilizes heavy datasets and Google Drive mounting, it is highly recommended to run it in Google Colab.

1. Open the `.ipynb` file in this repository.
2. Click the **"Open in Colab"** button at the top of the file (if using a browser extension) or manually upload it to your Google Colab account.
3. Update the `DATA_PATH` variable to point to your own dataset folder containing `Fresh_Images` and `Rotten_Images`.
4. Run all cells to train the models and see the final accuracy comparison!

## 💻 Technologies & Libraries
* Python, OpenCV (`cv2`) for image preprocessing
* TensorFlow & Keras for deep learning
* Scikit-Learn for SVM implementation and dataset splitting
* Pandas & NumPy for data structuring
