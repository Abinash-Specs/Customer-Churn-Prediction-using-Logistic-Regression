#!/usr/bin/env python
# coding: utf-8

# #  Customer Churn Prediction using Logistic Regression
# 
# ###  Objective
# The goal of this project is to build a machine learning model that predicts whether a customer will churn or not using Logistic Regression.
# 
# ---
# 
# ###  Key Tasks:
# - Data preprocessing and cleaning
# - Feature engineering
# - Model training using Logistic Regression
# - Model evaluation using multiple metrics
# - Interpretation using Odds Ratio

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')


# ## 1.Data Loading
# 
# We load both training and testing datasets and merge them to ensure consistent preprocessing1

# In[5]:


train_df = pd.read_csv("churn-bigml-80.csv")
test_df = pd.read_csv("churn-bigml-20.csv")

# Combine datasets
df = pd.concat([train_df, test_df], axis=0)

df.head()


# ## 2.EDA

# In[6]:


print("Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())

# Target distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()


# ## 3.Data Cleaning & Preprocessing

# In[7]:


# Drop unnecessary columns
if 'State' in df.columns:
    df.drop(['State'], axis=1, inplace=True)

# Convert categorical to numeric
le = LabelEncoder()

for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

df.head()


# ## 4.Feature Selection & Splitting

# In[8]:


X = df.drop('Churn', axis=1)
y = df['Churn']

# Split again for modeling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ## 5.Feature Scaling

# In[9]:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ## 6. Model Training (Logistic Regression)

# In[10]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# ## 7.Model Evaluation

# In[11]:


y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# ## 8.Confusion Matrix Visualization

# In[15]:


cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix Heatmap")
plt.show()


# ## 9.ROC Curve Analysis

# In[12]:


y_prob = model.predict_proba(X_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)

plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()


# ## 10.Precision-Recall Curve

# In[13]:


precision, recall, _ = precision_recall_curve(y_test, y_prob)

plt.plot(recall, precision)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()


# ## 11.Model Interpretation (Odds Ratio)
# 

# In[14]:


coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0],
    "Odds Ratio": np.exp(model.coef_[0])
})

coefficients.sort_values(by="Odds Ratio", ascending=False)


# ## 12.Feature Importance

# In[16]:


importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.coef_[0]
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x="Importance", y="Feature", data=importance)
plt.title("Feature Importance")
plt.show()


# ## Business Insights
# 
# - Customers with higher usage are more likely to churn
# - Certain features strongly influence churn probability
# - Logistic Regression provides interpretable insights
# 
# ---
# 
# ## Recommendations:
# - Target high-risk customers
# - Improve service for heavy users
# - Implement retention strategies
