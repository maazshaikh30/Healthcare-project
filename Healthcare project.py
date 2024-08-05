#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
df = pd.read_csv('dataset.csv')


# In[8]:


filtered_df = df.drop_duplicates(subset=df.columns[1:], keep='first')
print(filtered_df)
filtered_df.head(10)


# In[28]:


#print unq categories in symtoms
for column in df.columns[1:]:
    unique_categories = df[column].unique()
    print(f"Unique categories in {column}: {unique_categories}")
print(df)


# In[ ]:





# In[29]:


df_encoded = pd.get_dummies(df, columns=df.columns[1:])
df_encoded.to_csv('encoded_dataset.csv', index=False)


# In[30]:


df_encoded.head(100)


# In[31]:


df_encoded = pd.read_csv('encoded_dataset.csv')
print("Encoded DataFrame:")
print(df_encoded.head())

# Merge common columns by summing up the values
df_merged = df_encoded.groupby(df_encoded.columns, axis=1).sum()
df_merged.to_csv('merged_dataset.csv', index=False)
print("\nMerged DataFrame:")
print(df_merged.head())


# In[32]:


df_merged.tail(10)


# In[33]:


df_merged = pd.read_csv('merged_dataset.csv')

symptom_df = df_merged.drop(columns=['Disease'])

symptom_counts = symptom_df.sum()

top_20_symptoms = symptom_counts.nlargest(20).index

# Keep only the top 20 symptoms in the merged dataset
filtered_df = df_merged[['Disease'] + list(top_20_symptoms)]

unique_symptoms = []

# Keep only one unique occurrence of each symptom
for column in filtered_df.columns[1:]:
    symptom_name = column.split('_')[2]
    if symptom_name not in unique_symptoms:
        unique_symptoms.append(symptom_name)
    else:
        filtered_df.drop(columns=[column], inplace=True)

filtered_df.to_csv('filtered_unique_symptoms.csv', index=False)
print("Filtered Dataset with Unique Symptoms:")
print(filtered_df)


# In[34]:


df_encoded = pd.read_csv('encoded_dataset.csv')

# Calculate the count of 1s in each symptom column
symptom_counts = df_encoded.iloc[:, 1:].sum()

# Group similar symptoms together by summing up their counts
symptom_groups = {}
for column in symptom_counts.index:
    symptom_name = '_'.join(column.split('_')[2:])
    if symptom_name not in symptom_groups:
        symptom_groups[symptom_name] = 0
    symptom_groups[symptom_name] += symptom_counts[column]

symptom_groups_df = pd.DataFrame(symptom_groups.items(), columns=['Symptom', 'Count'])
symptom_groups_df = symptom_groups_df.sort_values(by='Count', ascending=False)
top_20_unique_symptoms = symptom_groups_df.head(20)

print("Top 20 unique symptoms:")
print(top_20_unique_symptoms)


# In[19]:


symptom_disease_mapping = {
    ('fatigue', 'high_fever', 'chills'): 'Influenza',
    ('fatigue', 'vomiting', 'loss_of_appetite', 'nausea'): 'Gastroenteritis',
    ('headache', 'nausea', 'vomiting', 'sweating'): 'Migraine',
    ('abdominal_pain', 'loss_of_appetite', 'dark_urine', 'yellowish_skin', 'yellowing_of_eyes'): 'Jaundice',
    ('chest_pain', 'sweating', 'fatigue'): 'Heart Attack',
    ('joint_pain', 'fatigue'): 'Arthritis',
    ('itching', 'skin_rash'): 'Allergy',
    ('cough', 'malaise', 'sneezing'): 'Common Cold',
    ('diarrhoea', 'abdominal_pain', 'vomiting', 'loss_of_appetite'): 'Gastroenteritis',
    ('irritability', 'fatigue', 'malaise'): 'Common Cold'
}

print("Please enter your symptoms (choose from the following):")
symptoms = list(symptom_disease_mapping.keys())
user_symptoms = []
for symptom in symptoms:
    user_input = input(f"Are you experiencing {symptom}? (Enter 1 for Yes, 0 for No): ")
    user_symptoms.append(int(user_input))

matching_diseases = []
for idx, symptom_set in enumerate(symptoms):
    if all(user_symptoms[i] == 1 for i in range(len(symptom_set))):
        matching_diseases.append(symptom_disease_mapping[symptom_set])

if not matching_diseases:
    possible_diseases = set()
    for symptom_set, disease in symptom_disease_mapping.items():
        for symptom, is_experiencing in zip(symptom_set, user_symptoms):
            if is_experiencing == 1:
                possible_diseases.add(disease)
    print("No matching disease found based on your symptoms.")
    print("Possible diseases based on your symptoms:")
    for disease in possible_diseases:
        print("-", disease)
else:
    print("Predicted disease based on your symptoms:", matching_diseases[0])


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')

symptom_cols = [col for col in df.columns if col.startswith('Symptom')]
df[symptom_cols] = df[symptom_cols].astype('category')

X = df.drop('Disease', axis=1)
y = df['Disease']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

params = {
    'objective': 'multi:softmax', 
    'num_class': len(label_encoder.classes_), 
}

xgb_model = xgb.train(params, dtrain)
y_pred = xgb_model.predict(dtest)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

symptom_cols = [col for col in df.columns if col.startswith('Symptom')]
df[symptom_cols] = df[symptom_cols].astype('category')

X = df.drop('Disease', axis=1)
y = df['Disease']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

params = {
    'objective': 'multi:softmax',  
    'num_class': len(label_encoder.classes_),  
}

xgb_model = xgb.train(params, dtrain)

y_pred = xgb_model.predict(dtest)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.figure(figsize=(9, 4))
sns.kdeplot(y_test, label='Actual Disease', color='blue', shade=True)
sns.kdeplot(y_pred, label='Predicted Disease', color='red', shade=True)
plt.title('Density Plot of Actual vs. Predicted Disease')
plt.xlabel('Disease')
plt.ylabel('Density')
plt.legend()
plt.show()


# In[18]:


from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')

symptom_cols = [col for col in df.columns if col.startswith('Symptom')]
df[symptom_cols] = df[symptom_cols].astype('category')

X = df.drop('Disease', axis=1)
y = df['Disease']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

params = {
    'objective': 'multi:softmax',  
    'num_class': len(label_encoder.classes_),  
}

xgb_model = xgb.train(params, dtrain)

y_pred = xgb_model.predict(dtest)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.title('Scatter Plot of Actual vs Predicted Disease')
plt.xlabel('Actual Disease')
plt.ylabel('Predicted Disease')
plt.subplot(1, 2, 2)
sns.kdeplot(y_test, label='Actual Disease', color='blue', shade=True)
sns.kdeplot(y_pred, label='Predicted Disease', color='red', shade=True)
plt.title('Density Plot of Actual vs. Predicted Disease')
plt.xlabel('Disease')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Load the dataset
df = pd.read_csv('dataset.csv')

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=df.columns[1:])

# Split dataset into features (X) and target (y)
X = df_encoded.drop('Disease', axis=1)
y = df_encoded['Disease']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))

# Define KFold with 100 folds
kf = KFold(n_splits=100, shuffle=True, random_state=42)

# Perform cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y_encoded, cv=kf)

# Perform cross-validation for XGBoost
xgb_cv_scores = cross_val_score(xgb_model, X, y_encoded, cv=kf)

# Calculate mean accuracy scores
rf_mean_accuracy = np.mean(rf_cv_scores)
xgb_mean_accuracy = np.mean(xgb_cv_scores)

# Plotting the results
models = ['Random Forest', 'XGBoost']
mean_accuracies = [rf_mean_accuracy, xgb_mean_accuracy]

plt.figure(figsize=(8, 6))
plt.bar(models, mean_accuracies, color=['skyblue', 'lightgreen'])
plt.title('Mean Cross-Validation Accuracy of Random Forest vs XGBoost on Dataset')
plt.xlabel('Model')
plt.ylabel('Mean Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for better visualization
plt.show()


# In[21]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset.csv')

symptom_cols = [col for col in df.columns if col.startswith('Symptom')]
df[symptom_cols] = df[symptom_cols].astype('category')

df = pd.get_dummies(df, columns=symptom_cols)

X = df.drop('Disease', axis=1)
y = df['Disease']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, color='blue')
plt.title('Scatter Plot of Actual vs Predicted Disease')
plt.xlabel('Actual Disease')
plt.ylabel('Predicted Disease')

plt.subplot(1, 2, 2)
sns.kdeplot(y_test, label='Actual Disease', color='blue', shade=True)
sns.kdeplot(y_pred, label='Predicted Disease', color='red', shade=True)
plt.title('Density Plot of Actual vs. Predicted Disease')
plt.xlabel('Disease')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()


# In[7]:


import xgboost as xgb
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv')
df_encoded = pd.get_dummies(df, columns=df.columns[1:])

X = df_encoded.drop('Disease', axis=1)
y = df_encoded['Disease']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
rf_accuracy = accuracy_score(y_test, rf_model.predict(X_test))

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
params = {'objective': 'multi:softmax', 'num_class': len(label_encoder.classes_)}
xgb_model = xgb.train(params, dtrain)
xgb_accuracy = accuracy_score(y_test, xgb_model.predict(dtest))

models = ['Random Forest', 'XGBoost']
accuracies = [rf_accuracy, xgb_accuracy]

plt.figure(figsize=(8, 6))
bar_width = 0.35
index = range(len(models))

plt.bar(index, accuracies, bar_width, label='Accuracy', color=['skyblue', 'lightgreen'])
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy of Random Forest vs XGBoost on Dataset')
plt.xticks(index, models)
plt.ylim(0, 1) 

for i, acc in enumerate(accuracies):
    plt.text(i, acc, f'{acc:.2f}', ha='center', va='bottom', fontsize=10)

plt.legend()
plt.tight_layout()
plt.show()


# In[12]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Load the dataset
df = pd.read_csv('dataset.csv')

# Perform one-hot encoding for categorical variables
df_encoded = pd.get_dummies(df, columns=df.columns[1:])

# Split dataset into features (X) and target (y)
X = df_encoded.drop('Disease', axis=1)
y = df_encoded['Disease']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Initialize models
rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))

# Define KFold with 100 folds
kf = KFold(n_splits=100, shuffle=True, random_state=42)

# Perform cross-validation for Random Forest
rf_cv_scores = cross_val_score(rf_model, X, y_encoded, cv=kf)

# Perform cross-validation for XGBoost
xgb_cv_scores = cross_val_score(xgb_model, X, y_encoded, cv=kf)

# Calculate mean accuracy scores
rf_mean_accuracy = np.mean(rf_cv_scores)
xgb_mean_accuracy = np.mean(xgb_cv_scores)

# Plotting the results
models = ['Random Forest', 'XGBoost']
mean_accuracies = [rf_mean_accuracy, xgb_mean_accuracy]

# Line plot
plt.figure(figsize=(8, 6))
plt.plot(models, mean_accuracies, marker='o', linestyle='-', color='b')
plt.title('Mean Cross-Validation Accuracy of Random Forest vs XGBoost on Dataset')
plt.xlabel('Model')
plt.ylabel('Mean Accuracy')
plt.ylim(0, 1)  # Set y-axis limit to 0-1 for better visualization
plt.grid(True)
plt.show()


# In[14]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
df = pd.read_csv('dataset.csv')

df_encoded = pd.get_dummies(df, columns=df.columns[1:])

X = df_encoded.drop('Disease', axis=1)
y = df_encoded['Disease']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

rf_model = RandomForestClassifier(random_state=42)
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_))

kf = KFold(n_splits=100, shuffle=True, random_state=42)

rf_cv_scores = cross_val_score(rf_model, X, y_encoded, cv=kf)

xgb_cv_scores = cross_val_score(xgb_model, X, y_encoded, cv=kf)

rf_mean_accuracy = np.mean(rf_cv_scores)
xgb_mean_accuracy = np.mean(xgb_cv_scores)

models = ['Random Forest', 'XGBoost']
mean_accuracies = [rf_mean_accuracy, xgb_mean_accuracy]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

axes[0].plot(models, mean_accuracies, marker='o', linestyle='-', color='b')
axes[0].set_title('Mean Cross-Validation Accuracy of Random Forest')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Mean Accuracy')
axes[0].set_ylim(0, 1) 
axes[0].grid(True)

axes[1].plot(models, mean_accuracies, marker='o', linestyle='-', color='r')
axes[1].set_title('Mean Cross-Validation Accuracy of XGBoost')
axes[1].set_xlabel('Model')
axes[1].set_ylabel('Mean Accuracy')
axes[1].set_ylim(0, 1) 
axes[1].grid(True)

plt.tight_layout()
plt.show()


# In[ ]:




