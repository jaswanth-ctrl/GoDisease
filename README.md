# Plant Disease Classification Ensemble

This project implements an ensemble approach for plant disease classification using deep learning and meta-learning. It leverages three powerful convolutional neural network architectures—DenseNet, MobileNet, and Xception—and combines their predictions using a meta-model (Logistic Regression) to achieve high accuracy in identifying multiple plant diseases from images.

## Project Structure
- `densenet.ipynb`: Jupyter notebook for training and evaluating the DenseNet model.
- `mobilenet.ipynb`: Jupyter notebook for training and evaluating the MobileNet model.
- `xception.ipynb`: Jupyter notebook for training and evaluating the Xception model.
- `ensemble_lr.ipynb`: Jupyter notebook for combining the predictions of the above models using a Logistic Regression meta-model.
- `*_best_model.h5`: Saved best weights for each respective model.
- `lr_meta_model.pkl`: Pickled meta-model for ensemble predictions.

## Workflow Overview
1. **Data Preparation**: Reads the training data from a CSV, processes image paths, and splits the dataset into training and validation sets.
2. **Model Training**:
   - Each notebook (`densenet.ipynb`, `mobilenet.ipynb`, `xception.ipynb`) trains a separate deep learning model on the plant disease dataset.
   - Data augmentation and preprocessing are applied using Keras' `ImageDataGenerator`.
3. **Prediction Stacking**:
   - Predictions from each model are generated on the validation set.
   - These predictions are concatenated to form the meta-features for the ensemble.
4. **Meta-Model Training**:
   - A `MultiOutputClassifier` with Logistic Regression is trained on the stacked predictions to learn the final classification.
   - The meta-model is saved as `lr_meta_model.pkl`.
5. **Evaluation**:
   - The ensemble's performance is evaluated using accuracy, Hamming loss, F1-score, and a detailed classification report.

## Requirements
- Python 3.12+
- TensorFlow
- Keras
- scikit-learn
- pandas
- matplotlib
- plotly
- scikit-image

Install dependencies with:
```bash
pip install tensorflow keras scikit-learn pandas matplotlib plotly scikit-image
```

## Usage
1. Place your dataset (CSV and images) as referenced in the notebooks.
2. Run each model notebook (`densenet.ipynb`, `mobilenet.ipynb`, `xception.ipynb`) to train and save the best models.
3. Run `ensemble_lr.ipynb` to stack predictions and train the meta-model.
4. Use the saved models (`*_best_model.h5` and `lr_meta_model.pkl`) for inference on new data.

## Results
The ensemble approach achieves high accuracy, with an accuracy rate of 97.60% on the validation set, outperforming individual models. See the classification report in `ensemble_lr.ipynb` for detailed metrics.

## Acknowledgements
- The project uses open-source libraries and pre-trained models from Keras/TensorFlow.
- Dataset and problem inspired by plant disease detection challenges.

---
*Last updated: 2025-06-19*
