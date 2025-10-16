# SCT_ML_1
🏡 House Price Prediction using Linear Regression

This project demonstrates a complete machine learning pipeline for predicting real estate prices using a Linear Regression model. The script handles data loading, preprocessing, model training, evaluation, and visualization, and it generates a submission file with predictions for a test dataset.

📊 Project Overview

The primary goal of this project is to build a model that can accurately predict the TARGET(PRICE_IN_LACS) based on various features of a property, such as its size, location, and amenities. The project showcases a practical, end-to-end workflow common in data science tasks.

✨ Key Features

Data Preprocessing: Handles categorical data by converting it into a numerical format using one-hot encoding.

Feature Scaling: Utilizes StandardScaler to normalize features, which is a best practice for linear models.

Model Training: Implements a LinearRegression model from scikit-learn.

Model Evaluation: Calculates key performance metrics, including Mean Squared Error (MSE) and R² Score.

Data Visualization: Generates two insightful plots:

A scatter plot of Actual vs. Predicted Prices to evaluate overall model performance.

A plot showing the Linear Trend between SQUARE_FT and price to intuitively understand the linear regression concept.

Submission File: Trains a final model on the entire dataset and generates a submission.csv file with predictions for the test data.

📁 Dataset Files

To run this project, you need the following files in the same directory as the script:

train.csv: The training dataset containing property features and the target price.

test.csv: The test dataset containing property features for which predictions are needed.

sample_submission.csv: A file showing the required format for the final submission.

⚙️ Setup and Installation

Clone the repository (or download the files):

git clone <your-repository-url>
cd <your-repository-directory>


Install the required Python libraries. A requirements.txt file is recommended for larger projects, but for this script, you can install them directly using pip:

pip install pandas numpy scikit-learn matplotlib


🚀 How to Run

Ensure that train.csv, test.csv, and sample_submission.csv are in the same folder as the linear_regression.py script.

Execute the script from your terminal:

python linear_regression.py


The script will:

Print the model's R² score and MSE to the console.

Display the two performance plots.

Generate a submission.csv file in the same directory.

📈 Visualizations Explained

The script produces two plots to help understand the model:

Overall Model Performance: This plot shows how the model's predictions line up against the actual prices. The closer the blue dots are to the red dashed line, the more accurate the model is.

Linear Trend for Square Footage: This plot isolates the relationship between a property's square footage and its price, drawing the exact straight line that the model learns. This is a classic visualization for explaining how linear regression works.
