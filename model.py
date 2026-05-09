import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load dataset
print("Loading dataset...")
data = pd.read_csv("student_data.csv")

# Display dataset
print("\nDataset:\n")
print(data)

# Input and output
X = data[["Hours"]]
y = data["Marks"]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

print("\nModel Training Completed Successfully!")

# Prediction
hours = float(input("\nEnter study hours: "))
predicted_marks = model.predict([[hours]])

print(f"\nPredicted Marks = {predicted_marks[0]:.2f}")

# Graph
plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.title('Study Hours vs Marks')
plt.xlabel('Study Hours')
plt.ylabel('Marks')
plt.show()
