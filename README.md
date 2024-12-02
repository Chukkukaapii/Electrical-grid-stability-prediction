# Electrical Grid Stability Prediction

## Project Overview

This project aims to predict the stability of an electrical grid using machine learning algorithms. It leverages a dataset containing simulated data from electrical grid components, with features such as "tau1", "tau2", "p1", "p2", and "p3", which are indicative of grid performance and stability. A **Random Forest Classifier** is used to predict whether the grid is stable or unstable.

## Technologies Used
- **Python**: Core programming language.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: Machine learning algorithms, specifically Random Forest.
- **Streamlit**: Web framework to build the interactive user interface.
- **KaggleHub**: Dataset integration for direct download.

## Features
- **Dataset Upload**: Option to upload your own dataset or use the pre-existing Kaggle dataset.
- **Model Training**: Train a Random Forest model with adjustable hyperparameters.
- **Model Evaluation**: View performance metrics such as accuracy, confusion matrix, and classification report.
- **Prediction**: Input new grid data to predict its stability.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/grid-stability-prediction.git
