# FINANCIAL RISK ASSESSMENT USING MACHINE LEARNING


# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer

# LOAD DATASET

df = pd.read_csv("Data/financial_risk_assessment.csv")

print("="*60)
print("DATASET INFORMATION")
print("="*60)

print("\nShape of Dataset:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nFirst Five Records:")
print(df.head())

# MISSING VALUES

print("\nMissing Values")
print(df.isnull().sum())

# Numerical Columns
num_cols = df.select_dtypes(include=np.number).columns

num_imputer = SimpleImputer(strategy='median')
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# Categorical Columns
cat_cols = df.select_dtypes(include='object').columns

cat_imputer = SimpleImputer(strategy='most_frequent')
df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])

print("\nMissing Values After Treatment")
print(df.isnull().sum())

# DESCRIPTIVE STATISTICS

print("\nDescriptive Statistics")
print(df.describe())


# GRAPH 1 - RISK CATEGORY DISTRIBUTION

plt.figure(figsize=(8,5))

sns.countplot(
    data=df,
    x='Risk Rating'
)

plt.title('Distribution of Financial Risk Categories')
plt.xlabel('Risk Rating')
plt.ylabel('Number of Customers')

plt.tight_layout()
plt.show()

# GRAPH 2 - AGE DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(
    df['Age'],
    bins=20,
    kde=True
)

plt.title('Age Distribution of Customers')
plt.xlabel('Age')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# GRAPH 3 - INCOME DISTRIBUTION

plt.figure(figsize=(8,5))

sns.histplot(
    df['Income'],
    bins=30,
    kde=True
)

plt.title('Income Distribution')
plt.xlabel('Annual Income')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# GRAPH 4 - CREDIT SCORE BY RISK CATEGORY

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x='Risk Rating',
    y='Credit Score'
)

plt.title('Credit Score Across Risk Categories')
plt.xlabel('Risk Rating')
plt.ylabel('Credit Score')

plt.tight_layout()
plt.show()

# GRAPH 5 - DEBT TO INCOME RATIO

plt.figure(figsize=(8,5))

sns.boxplot(
    data=df,
    x='Risk Rating',
    y='Debt-to-Income Ratio'
)

plt.title('Debt-to-Income Ratio Across Risk Categories')
plt.xlabel('Risk Rating')
plt.ylabel('Debt-to-Income Ratio')

plt.tight_layout()
plt.show()

# GRAPH 6 - CORRELATION HEATMAP

numeric_df = df.select_dtypes(include=['int64','float64'])

plt.figure(figsize=(12,8))

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm',
    fmt='.2f'
)

plt.title('Correlation Heatmap')

plt.tight_layout()
plt.show()

# GRAPH 7 - ASSET VALUE ANALYSIS

plt.figure(figsize=(8,5))

sns.boxplot(
    x=df['Assets Value']
)

plt.title('Asset Value Distribution')
plt.xlabel('Assets Value ($)')

plt.tight_layout()
plt.show()


# RISK CATEGORY PERCENTAGES

risk_dist = df['Risk Rating'].value_counts()

print("\nRisk Distribution")
print(risk_dist)

risk_percent = round(
    (risk_dist / len(df)) * 100,
    2
)

print("\nRisk Percentage")
print(risk_percent)


# SUMMARY STATISTICS FOR REPORT

print("\nProject Summary")

print(f"Total Customers: {len(df):,}")

print(f"Average Age: {df['Age'].mean():.2f}")

print(f"Average Income: ${df['Income'].mean():,.2f}")

print(f"Average Credit Score: {df['Credit Score'].mean():.2f}")

print(f"Average Loan Amount: ${df['Loan Amount'].mean():,.2f}")

print(f"Average Debt-to-Income Ratio: {df['Debt-to-Income Ratio'].mean():.2f}")

print(f"Average Assets Value: ${df['Assets Value'].mean():,.2f}")

# SAVE CLEAN DATASET

df.to_csv(
    "financial_risk_assessment_cleaned.csv",
    index=False
)

print("\nCleaned Dataset Saved Successfully")