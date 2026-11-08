import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create output folder
os.makedirs("eda_charts", exist_ok=True)

# Load dataset
df = pd.read_csv("train.csv")

print("\nDATASET LOADED SUCCESSFULLY!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Information:")
df.info()

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe(include="all"))

# Data cleaning
df.drop_duplicates(inplace=True)

# Fill missing numerical values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())

# Fill missing categorical values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Cabin"] = df["Cabin"].fillna("Unknown")

# Save cleaned dataset
df.to_csv("cleaned_titanic.csv", index=False)

print("\nDATA CLEANING COMPLETED!")
print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Exploratory Data Analysis

# Survival count
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Survived")
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Number of Passengers")
plt.savefig("eda_charts/survival_count.png")
plt.close()

# Gender and survival
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived")
plt.title("Survival by Gender")
plt.savefig("eda_charts/survival_by_gender.png")
plt.close()

# Passenger class and survival
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived")
plt.title("Survival by Passenger Class")
plt.savefig("eda_charts/survival_by_class.png")
plt.close()

# Age distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.savefig("eda_charts/age_distribution.png")
plt.close()

# Fare distribution
plt.figure(figsize=(7, 4))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.savefig("eda_charts/fare_distribution.png")
plt.close()

# Correlation heatmap
plt.figure(figsize=(9, 6))
numeric_df = df.select_dtypes(include=np.number)
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_charts/correlation_heatmap.png")
plt.close()

# Survival rate by gender
gender_survival = df.groupby("Sex")["Survived"].mean()
print("\nSurvival Rate by Gender:")
print(gender_survival)

# Survival rate by passenger class
class_survival = df.groupby("Pclass")["Survived"].mean()
print("\nSurvival Rate by Passenger Class:")
print(class_survival)

# Save analysis summary
with open("eda_report.txt", "w") as file:
    file.write("TITANIC DATASET - EDA REPORT\n")
    file.write("=" * 40 + "\n\n")
    file.write(f"Total passengers analyzed: {len(df)}\n\n")
    file.write("Survival rate by gender:\n")
    file.write(gender_survival.to_string())
    file.write("\n\nSurvival rate by passenger class:\n")
    file.write(class_survival.to_string())
    file.write("\n\nKey observations:\n")
    file.write("1. Passenger survival varied by gender.\n")
    file.write("2. Passenger class was related to survival chances.\n")
    file.write("3. Age and fare distributions were explored.\n")
    file.write("4. Numerical relationships were examined using correlation.\n")

print("\nEDA COMPLETED SUCCESSFULLY!")
print("Cleaned dataset saved.")
print("Charts saved in eda_charts folder.")
print("EDA report generated.")