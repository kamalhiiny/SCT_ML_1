# 1️⃣ Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 2️⃣ Load all datasets
try:
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")
    sample_df = pd.read_csv("sample_submission.csv")
    print("✅ Files loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}")
    print("Please make sure 'train.csv', 'test.csv', and 'sample_submission.csv' are in the same directory.")
    exit()


# 3️⃣ Preprocess Data for Consistent Features
# 👉 The target column name is 'TARGET(PRICE_IN_LACS)' in your CSV.
target_col = "TARGET(PRICE_IN_LACS)"

if target_col not in train_df.columns:
    raise ValueError(f"⚠️ The column '{target_col}' was not found. Please check the CSV file.")

# Separate features (X) and target (y) from the training data
X = train_df.drop(columns=[target_col])
y = train_df[target_col]
X_test_final = test_df.copy() # Use a copy for the final test set predictions

# Align columns before one-hot encoding to prevent mismatches
train_cols = X.columns
test_cols = X_test_final.columns
shared_cols = list(set(train_cols) & set(test_cols))
X = X[shared_cols]
X_test_final = X_test_final[shared_cols]

# Drop the 'ADDRESS' column as it has too many unique values for this model
X = X.drop(columns=['ADDRESS'])
X_test_final = X_test_final.drop(columns=['ADDRESS'])

# Convert categorical columns into numerical ones using one-hot encoding
# Combining train and test temporarily ensures all categories are represented
combined_df = pd.concat([X, X_test_final], axis=0)
combined_df_processed = pd.get_dummies(combined_df, drop_first=True)

# Separate them back into training and final test sets
X_processed = combined_df_processed.iloc[:len(X)]
X_test_final_processed = combined_df_processed.iloc[len(X):]

print("\n✅ Data preprocessing complete. Categorical features converted.")


# 4️⃣ Split the training data for model evaluation and visualization
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# 5️⃣ Scale the features for the evaluation split
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("✅ Features scaled for model evaluation.")

# 6️⃣ Train the Linear Regression model for evaluation
print("\nTraining a Linear Regression model for evaluation...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)
print("✅ Evaluation model trained successfully!")

# 7️⃣ Make predictions and evaluate the model
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Evaluation on Hold-out Set:")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R² Score: {r2:.4f} (closer to 1.0 is better)")

# 8️⃣ Visualization 1 — Overall Model Performance (Actual vs. Predicted)
plt.figure(figsize=(10, 7))
plt.scatter(y_test, y_pred, color='dodgerblue', alpha=0.6, edgecolors='w', label='Predicted vs Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', linewidth=2, label='Ideal Fit Line')
plt.title("Model Performance: Actual vs. Predicted Prices", fontsize=18, fontweight='bold')
plt.xlabel("Actual Price (in Lacs)", fontsize=14)
plt.ylabel("Predicted Price (in Lacs)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Set axis limits as requested to focus on the 100 to 10000 range
plt.xlim(100, 10000)
plt.ylim(100, 10000)

plt.show()

# 8b️⃣ Visualization 2 — Linear Trend for a Single Feature
# The plot above shows the overall model performance. To visualize the
# concept of linear regression more directly, we can plot the relationship
# between a single important feature (like SQUARE_FT) and the price.
# This plot shows the straight line that the model "learns" for this one feature.
print("\nGenerating a bonus plot to show the linear trend for 'SQUARE_FT'...")

# We use the original, non-scaled data for this illustrative plot
X_sqft = train_df[['SQUARE_FT']]
y_price = train_df[target_col]

# Train a simple model with only one feature for visualization purposes
simple_model = LinearRegression()
simple_model.fit(X_sqft, y_price)

# Generate the line of best fit from the simple model
line_of_best_fit = simple_model.predict(X_sqft)

# Create the plot
plt.figure(figsize=(10, 7))
# Plot the actual data points
plt.scatter(X_sqft, y_price, color='dodgerblue', alpha=0.3, edgecolors='w', label='Actual Data Points')
# Plot the regression line
plt.plot(X_sqft, line_of_best_fit, color='red', linewidth=3, label='Learned Regression Line')

plt.title("Linear Regression Trend: Square Footage vs. Price", fontsize=18, fontweight='bold')
plt.xlabel("Square Footage (SQUARE_FT)", fontsize=14)
plt.ylabel("Price (in Lacs)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Set axis limits as requested to focus on the 100 to 10000 range
plt.xlim(100, 10000)
plt.ylim(100, 10000)

plt.show()


# 9️⃣ Train Final Model on ALL Training Data and Generate Submission
print("\nTraining final model on all available training data...")
final_scaler = StandardScaler()
X_processed_scaled = final_scaler.fit_transform(X_processed)

final_model = LinearRegression()
final_model.fit(X_processed_scaled, y)
print("✅ Final model trained successfully!")

# Scale the final test data with the same scaler
X_test_final_scaled = final_scaler.transform(X_test_final_processed)

# Make predictions on the final test data
final_predictions = final_model.predict(X_test_final_scaled)

# 🔟 Create submission file
submission_df = pd.DataFrame({target_col: final_predictions})
submission_df.to_csv('submission.csv', index=False)

print(f"\n✅ Submission file 'submission.csv' created successfully with {len(submission_df)} predictions.")

