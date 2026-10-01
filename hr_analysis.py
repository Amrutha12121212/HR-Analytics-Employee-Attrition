import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("dataset/archive (5)/WA_Fn-UseC_-HR-Employee-Attrition.csv")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nAttrition Count:")
print(df["Attrition"].value_counts())

attrition_percentage = df["Attrition"].value_counts(normalize=True) * 100

print("\nAttrition Percentage:")
print(attrition_percentage)


# Employee Attrition Visualization
plt.figure(figsize=(6,4))

sns.countplot(x="Attrition", data=df)

plt.title("Employee Attrition Count")
plt.xlabel("Attrition")
plt.ylabel("Number of Employees")

plt.show()

# Department-wise Attrition
plt.figure(figsize=(8, 5))

sns.countplot(x="Department", hue="Attrition", data=df)

plt.title("Department-wise Employee Attrition")
plt.xlabel("Department")
plt.ylabel("Number of Employees")

plt.xticks(rotation=15)
plt.show()

# Overtime vs Attrition
plt.figure(figsize=(7, 5))

sns.countplot(x="OverTime", hue="Attrition", data=df)

plt.title("OverTime vs Employee Attrition")
plt.xlabel("OverTime")
plt.ylabel("Number of Employees")

plt.show()

# Department-wise Attrition Rate

department_attrition = df.groupby("Department")["Attrition"].apply(
    lambda x: (x == "Yes").mean() * 100
)

print("\nDepartment-wise Attrition Rate:")
print(department_attrition)

# Monthly Income vs Attrition

plt.figure(figsize=(8, 5))

sns.boxplot(x="Attrition", y="MonthlyIncome", data=df)

plt.title("Monthly Income vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Monthly Income")

plt.show()

# Job Satisfaction vs Attrition

plt.figure(figsize=(8, 5))

sns.countplot(x="JobSatisfaction", hue="Attrition", data=df)

plt.title("Job Satisfaction vs Employee Attrition")
plt.xlabel("Job Satisfaction Level")
plt.ylabel("Number of Employees")

plt.show()

# Job Level vs Attrition

plt.figure(figsize=(8, 5))

sns.countplot(x="JobLevel", hue="Attrition", data=df)

plt.title("Job Level vs Employee Attrition")
plt.xlabel("Job Level")
plt.ylabel("Number of Employees")

plt.savefig("job_level_attrition.png")
plt.close()

# Years at Company vs Attrition

plt.figure(figsize=(10, 5))

sns.countplot(x="YearsAtCompany", hue="Attrition", data=df)

plt.title("Years at Company vs Employee Attrition")
plt.xlabel("Years at Company")
plt.ylabel("Number of Employees")

plt.xticks(rotation=45)

plt.show()

# Distance From Home vs Attrition

plt.figure(figsize=(10, 5))

sns.boxplot(x="Attrition", y="DistanceFromHome", data=df)

plt.title("Distance From Home vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Distance From Home")

plt.savefig("distance_from_home_attrition.png")
plt.close()

print("Distance From Home graph saved successfully.")

# Years Since Last Promotion vs Attrition

plt.figure(figsize=(10, 5))

sns.countplot(x="YearsSinceLastPromotion", hue="Attrition", data=df)

plt.title("Years Since Last Promotion vs Employee Attrition")
plt.xlabel("Years Since Last Promotion")
plt.ylabel("Number of Employees")

plt.xticks(rotation=45)

plt.savefig("promotion_attrition.png")
plt.close()

print("Promotion graph saved successfully.")

# Age vs Attrition

plt.figure(figsize=(8, 5))

sns.boxplot(x="Attrition", y="Age", data=df)

plt.title("Age vs Employee Attrition")
plt.xlabel("Attrition")
plt.ylabel("Age")

plt.savefig("age_attrition.png")
plt.close()

print("Age graph saved successfully.")

# Monthly Income by Job Level

plt.figure(figsize=(8, 5))

sns.boxplot(x="JobLevel", y="MonthlyIncome", data=df)

plt.title("Monthly Income by Job Level")
plt.xlabel("Job Level")
plt.ylabel("Monthly Income")

plt.savefig("income_by_job_level.png")
plt.close()

print("Income by Job Level graph saved successfully.")

# Prepare Data for Machine Learning

from sklearn.preprocessing import LabelEncoder

# Create a copy of the dataset
ml_df = df.copy()

# Convert categorical columns into numbers
label_encoder = LabelEncoder()

for column in ml_df.select_dtypes(include="object").columns:
    ml_df[column] = label_encoder.fit_transform(ml_df[column])

print("\nData after encoding:")
print(ml_df.head())

print("\nNew Shape:", ml_df.shape)

# Separate Features and Target

X = ml_df.drop("Attrition", axis=1)
y = ml_df["Attrition"]

print("\nFeatures Shape:", X.shape)
print("Target Shape:", y.shape)

# Split Data into Training and Testing

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train Logistic Regression Model

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nLogistic Regression model trained successfully.")

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.tree import DecisionTreeClassifier

decision_tree = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree.fit(X_train, y_train)

y_pred_tree = decision_tree.predict(X_test)

tree_accuracy = accuracy_score(y_test, y_pred_tree)

print("\nDecision Tree Accuracy:", tree_accuracy)

print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_tree))

print("\nDecision Tree Classification Report:")
print(classification_report(y_test, y_pred_tree))

print("\n========== MODEL COMPARISON ==========")

print("Logistic Regression Accuracy:", accuracy)
print("Decision Tree Accuracy:", tree_accuracy)

print("\nLogistic Regression Attrition Recall:",
      classification_report(y_test, y_pred, output_dict=True)["1"]["recall"])

print("Decision Tree Attrition Recall:",
      classification_report(y_test, y_pred_tree, output_dict=True)["1"]["recall"])

print("\nClass Distribution:")
print(y.value_counts())

print("\nClass Distribution Percentage:")
print(y.value_counts(normalize=True) * 100)

# Improved Logistic Regression using class weights

improved_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

improved_model.fit(X_train, y_train)

y_pred_improved = improved_model.predict(X_test)

print("\nImproved Logistic Regression Accuracy:")
print(accuracy_score(y_test, y_pred_improved))

print("\nImproved Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_improved))

print("\nImproved Classification Report:")
print(classification_report(y_test, y_pred_improved))

print("\n========== FINAL MODEL COMPARISON ==========")

print("Original Logistic Regression")
print("Accuracy:", accuracy)

print("\nBalanced Logistic Regression")
print("Accuracy:", accuracy_score(y_test, y_pred_improved))

print("\nOriginal Classification Report:")
print(classification_report(y_test, y_pred))

print("\nBalanced Classification Report:")
print(classification_report(y_test, y_pred_improved))

# Confusion Matrix Visualization

cm = confusion_matrix(y_test, y_pred_improved)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Stayed", "Left"],
    yticklabels=["Stayed", "Left"]
)

plt.title("Confusion Matrix - Employee Attrition")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("confusion_matrix.png")
plt.close()

print("\nConfusion Matrix graph saved successfully.")

# Feature Importance from Logistic Regression

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": improved_model.coef_[0]
})

feature_importance["Absolute_Importance"] = (
    feature_importance["Importance"].abs()
)

feature_importance = feature_importance.sort_values(
    "Absolute_Importance",
    ascending=False
)

print("\nTop 10 Important Features:")
print(feature_importance.head(10))

# Top 10 Feature Importance Graph

top_features = feature_importance.head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x="Absolute_Importance",
    y="Feature",
    data=top_features
)

plt.title("Top 10 Factors Influencing Employee Attrition")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("\nFeature importance graph saved successfully.")

import shap

# Create SHAP explainer
explainer = shap.LinearExplainer(improved_model, X_train)

# Calculate SHAP values
shap_values = explainer(X_test)

print("\nSHAP analysis completed successfully.")

# SHAP Summary Plot

plt.figure()

shap.summary_plot(
    shap_values,
    X_test,
    show=False
)

plt.tight_layout()
plt.savefig("shap_summary.png", bbox_inches="tight")
plt.close()

print("\nSHAP summary graph saved successfully.")