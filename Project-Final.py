#!/usr/bin/env python
# coding: utf-8

# # Credit Card Fraud
# 
# This dataset consists of credit card transactions in the western United States. It includes information about each transaction including customer details, the merchant and category of purchase, and whether or not the transaction was a fraud.

# In[1]:


import pandas as pd 

data=pd.read_csv('credit_card_fraud.csv') 


# In[2]:


# Display the first few rows of the dataset to ensure it's loaded correctly
print("Displaying the first 5 entries of the dataset:")
display(data.head())


# In[3]:


#Show the size of the dataset
print("\nThe dataset contains {} rows and {} columns.".format(data.shape[0], data.shape[1]))


# In[4]:


# Present a basic description of the dataset
print("\nBasic statistical details of the numerical features:")
display(data.describe())


# In[5]:


# Check for missing values in the dataset
print("\nChecking for missing values in each column:")
display(data.isnull().sum())


# In[6]:


#Display the data types and non-null counts for each column
print("\nData types and non-null counts of each column:")
display(data.info())


# In[7]:


#Display the distribution of the target variable 'is_fraud'
print("\nDistribution of the 'is_fraud' target variable:")
fraud_distribution = data['is_fraud'].value_counts()
display(fraud_distribution)


# In[8]:


# Show the number of unique values in each categorical column
print("\nUnique values in each categorical column:")
categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    print(f"{column}: {data[column].nunique()} unique values")


# In[9]:


#Display a sample of fraudulent transactions
print("\nA sample of fraudulent transactions:")
display(data[data['is_fraud'] == 1].head())


# In[10]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

numeric_data = data.select_dtypes(include=[np.number])  # This will select only the numeric columns
corr_matrix = numeric_data.corr()  # Calculate the correlation matrix

# Continue with the heatmap visualization
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.title('Correlation Matrix of Numeric Features')
plt.show()


# In[11]:


# Bar chart of categories
plt.figure(figsize=(15, 6))
category_counts = data['category'].value_counts()
sns.barplot(x=category_counts.index, y=category_counts.values)
plt.title('Frequency of Transactions by Category')
plt.xticks(rotation=90)  # Rotates the labels on the x-axis to make them readable
plt.xlabel('Category')
plt.ylabel('Number of Transactions')
plt.show()


# In[12]:


# Assuming your date column is named "date_column"
data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])


# In[13]:


# Setting date as the index:
data.set_index('trans_date_trans_time', inplace=True)


# In[14]:


# Dataset Duplicate Value Count assinged a dataframe name 'df'
df = data[data.duplicated()]

#There is no duplicate rows in the data
print(df.head())

# Missing Values/Null Values Count
data.isna().sum()


# In[15]:


# Basic statistics
print("\nBasic statistical details:\n", data.describe())


# In[16]:


import pandas as pd

# Assuming 'data' is your DataFrame with a DatetimeIndex
data['month'] = data.index.month
data['weekday'] = data.index.weekday
data['time_of_day'] = data.index.hour
data['week_number'] = data.index.strftime('%U').astype(int) + 1  # %U gives the week number, add 1 to start from 1
data['date_day'] = data.index.day
data['weekday_name'] = data.index.dayofweek


# In[17]:


from geopy.distance import geodesic

# Define a function to calculate distance
def calculate_distance(row):
    return geodesic((row['lat'], row['long']), (row['merch_lat'], row['merch_long'])).miles

data['distance_from_home'] = data.apply(calculate_distance, axis=1)


# In[18]:


data.head(5)


# In[19]:


# Ensure the datetime index is sorted
data.sort_index(inplace=True)

# Calculate rolling aggregates for the past 7 days
data['rolling_avg_amt_7d'] = data['amt'].rolling('7D').mean()
data['rolling_max_amt_7d'] = data['amt'].rolling('7D').max()
data['rolling_min_amt_7d'] = data['amt'].rolling('7D').min()
data['rolling_std_amt_7d'] = data['amt'].rolling('7D').std()


# In[20]:


data.isna().sum()


# In[21]:


data = data.dropna()


# In[22]:


data.isna().sum()


# In[23]:


data.head(5)


# In[24]:


# Descriptive statistics for rolling window features by fraud status
stats_by_fraud_status = data.groupby('is_fraud')[['rolling_avg_amt_7d', 'rolling_max_amt_7d', 'rolling_min_amt_7d', 'rolling_std_amt_7d']].describe()

print(stats_by_fraud_status)


# In[25]:


import seaborn as sns
import matplotlib.pyplot as plt

# Visualize the distribution of rolling_avg_amt_7d for fraudulent and non-fraudulent transactions
plt.figure(figsize=(10, 6))
sns.histplot(data=data, x='rolling_avg_amt_7d', hue='is_fraud', element='step', stat='density', common_norm=False)
plt.title('Distribution of 7-Day Rolling Average Amount by Fraud Status')
plt.show()


# In[26]:


# Calculate correlations with 'is_fraud'
correlation_with_fraud = data[['rolling_avg_amt_7d', 'rolling_max_amt_7d', 'rolling_min_amt_7d', 'rolling_std_amt_7d', 'is_fraud']].corr()

# Isolate the correlation values with 'is_fraud', excluding the correlation of 'is_fraud' with itself
correlation_with_fraud_isolated = correlation_with_fraud['is_fraud'].drop('is_fraud')

# Display the correlation values
print(correlation_with_fraud_isolated)


# In[27]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Prepare data for model fitting
X = data[['rolling_avg_amt_7d', 'rolling_max_amt_7d', 'rolling_min_amt_7d', 'rolling_std_amt_7d']]
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit a simple logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Check the coefficients
feature_importance = model.coef_


# In[28]:


print(feature_importance)


# In[29]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
# List of numeric columns to check for outliers
numeric_cols = ['amt', 'city_pop', 'distance_from_home', 'rolling_avg_amt_7d', 'rolling_max_amt_7d', 'rolling_min_amt_7d', 'rolling_std_amt_7d']

# Calculate the number of rows needed for subplots based on the number of features
num_features = len(numeric_cols)
num_columns = 4  # You can adjust the number of columns as needed
num_rows = (num_features + num_columns - 1) // num_columns  # Rounds up the division

# Create the box plots
plt.figure(figsize=(20, num_rows * 5))  # Adjust the size based on the number of rows
plt.suptitle("Box Plot for Numeric Features", fontsize=18, y=1.02)

for n, col in enumerate(numeric_cols):
    ax = plt.subplot(num_rows, num_columns, n + 1)
    sns.boxplot(x=data[col], color='red')
    ax.set_title(col)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust the spacing if needed

plt.show()


# In[30]:


import pandas as pd
import numpy as np

def find_outliers_iqr(df, column):
    # Assuming the column is already in numeric type
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers

# List of numeric columns to check for outliers
numeric_cols = ['amt', 'city_pop', 'distance_from_home', 'rolling_avg_amt_7d', 'rolling_max_amt_7d', 'rolling_min_amt_7d', 'rolling_std_amt_7d']

# Apply the function to each numeric column and count the number of outliers
outliers_count = {col: find_outliers_iqr(data, col).shape[0] for col in numeric_cols}

# Print the count of outliers for each column
print("Number of outliers per column:")
for col, count in outliers_count.items():
    print(f"{col}: {count}")


# In[31]:


# Handling Outliers & Outlier treatments
for ftr in numeric_cols:
    print(ftr,'\n')
    q_25 = np.percentile(data[ftr], 25)  # Change 'df' to 'data'
    q_75 = np.percentile(data[ftr], 75)  # Change 'df' to 'data'
    iqr = q_75 - q_25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q_25, q_75, iqr))
    # calculate the outlier cutoff
    cut_off = iqr * 1.5
    lower = q_25 - cut_off
    upper = q_75 + cut_off
    print(f"\nlower = {lower} and upper = {upper} \n ")
    # identify outliers
    outliers = [x for x in data[ftr] if x < lower or x > upper]
    print('Identified outliers: %d' % len(outliers))
    # removing outliers
    if len(outliers) != 0:

        def bin(row):
            if row[ftr] > upper:
                return upper
            if row[ftr] < lower:
                return lower
            else:
                return row[ftr]

        data[ftr] = data.apply(lambda row: bin(row), axis=1)
        print(f"{ftr} Outliers Removed")
    print("\n-------\n")


# In[32]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming 'data' is your DataFrame
# List of numeric columns to check for outliers
numeric_cols = ['amt', 'city_pop', 'distance_from_home', 'rolling_avg_amt_7d', 'rolling_max_amt_7d', 'rolling_min_amt_7d', 'rolling_std_amt_7d']

# Calculate the number of rows needed for subplots based on the number of features
num_features = len(numeric_cols)
num_columns = 4  # You can adjust the number of columns as needed
num_rows = (num_features + num_columns - 1) // num_columns  # Rounds up the division

# Create the box plots
plt.figure(figsize=(20, num_rows * 5))  # Adjust the size based on the number of rows
plt.suptitle("Box Plot for Numeric Features", fontsize=18, y=1.02)

for n, col in enumerate(numeric_cols):
    ax = plt.subplot(num_rows, num_columns, n + 1)
    sns.boxplot(x=data[col], color='cyan')
    ax.set_title(col)
    plt.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust the spacing if needed

plt.show()


# In[ ]:





# In[33]:


data.head(5)


# In[34]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

# Assuming 'data' is your preprocessed DataFrame and 'is_fraud' is the target variable
X = data.drop(['is_fraud', 'trans_num'], axis=1)  # Features without the unique identifier
y = data['is_fraud']  # Target variable

# Select categorical columns with relatively low cardinality
categorical_cols = [cname for cname in X.columns if
                    X[cname].dtype == "object" and cname != 'trans_num']

# Select numerical columns
numerical_cols = [cname for cname in X.columns if 
                  X[cname].dtype in ['int64', 'float64']]

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Bundle preprocessing and modeling code in a pipeline
clf = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
clf.fit(X, y)

# Get feature importances
ohe = (clf.named_steps['preprocessor']
       .named_transformers_['cat']
       .named_steps['onehot'])
feature_names = ohe.get_feature_names_out(input_features=categorical_cols)
feature_names = np.r_[numerical_cols, feature_names]

importances = clf.named_steps['model'].feature_importances_
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame to see the most important features at the top
feature_importance_df.sort_values(by='Importance', ascending=False, inplace=True)

# Display the feature importances
print(feature_importance_df)


# In the above step of project, I employed a RandomForest Classifier to evaluate the importance of various features within my dataset. This crucial phase involved training the model with both numerical and categorically encoded data. The primary outcome of this process was the extraction of feature importance scores, which quantitatively indicated the contribution of each feature to the model's ability to predict fraud. My analysis revealed that features like amt, rolling_max_amt_7d, rolling_avg_amt_7d, and rolling_std_amt_7d emerged as particularly significant. I also observed the presence of several one-hot encoded features, especially those derived from the merchant variable, albeit with lower importance. This insight is pivotal for my project as it helps pinpoint the most influential factors in fraud detection, shaping my approach towards further refining the model and selecting the most relevant features.

# In[35]:


data.head(5)


# #### Applying SMOTE to address the imbalance in is_fraud column

# In[36]:


from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTENC
import pandas as pd

# Assuming 'data' is your DataFrame and 'is_fraud' is the target column
X = data.drop(['is_fraud', 'trans_num'], axis=1)  # Exclude 'trans_num' and target variable
y = data['is_fraud']

# Separate numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Apply median imputation to numeric columns only
numeric_imputer = SimpleImputer(strategy='median')
X_numeric_imputed = pd.DataFrame(numeric_imputer.fit_transform(X[numeric_cols]), columns=numeric_cols)

# Combine imputed numeric columns and categorical columns
X_combined = pd.concat([X_numeric_imputed, X[categorical_cols].reset_index(drop=True)], axis=1)

# Split the combined dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, stratify=y, random_state=42)

# Identify the indices of the categorical features for SMOTENC
categorical_features_indices = [X_train.columns.get_loc(col) for col in categorical_cols]

# Apply SMOTENC to the training set
smotenc = SMOTENC(categorical_features=categorical_features_indices, random_state=42)
X_train_balanced, y_train_balanced = smotenc.fit_resample(X_train, y_train)


# In[37]:


print(y_train.unique())


# In[38]:


# Before SMOTENC
print("Before SMOTENC:")
print(y_train.value_counts())

# After SMOTENC
print("\nAfter SMOTENC:")
print(y_train_balanced.value_counts())


# #### Check Skewness After SMOTE and Identify Skewed Features

# In[39]:


import numpy as np

# Calculate skewness for numeric columns
numeric_cols = X_train_balanced.select_dtypes(include=[np.number]).columns
skewness = X_train_balanced[numeric_cols].skew()

# Find the absolute value
abs_skewness = abs(skewness)

# Set up the threshold
skewness_threshold = 0.5

# Separate features into symmetrical and skewed based on skewness threshold
symmetrical_features = abs_skewness[abs_skewness < skewness_threshold].index
skewed_features = abs_skewness[abs_skewness >= skewness_threshold].index


# #### Apply Power Transformation to Skewed Features

# In[40]:


from sklearn.preprocessing import PowerTransformer

# Initialize PowerTransformer
power_transformer = PowerTransformer()

# Apply power transformation to skewed features
X_train_balanced[skewed_features] = power_transformer.fit_transform(X_train_balanced[skewed_features])
X_test[skewed_features] = power_transformer.transform(X_test[skewed_features])

# Check skewness after transformation
new_skewness = pd.DataFrame(X_train_balanced[skewed_features], columns=skewed_features).skew()
print("New Skewness of Skewed Features:\n", new_skewness)


# #### Scale the Features

# In[41]:


from sklearn.preprocessing import StandardScaler

# Assuming X_train_balanced and X_test are dataframes
# Identify numeric and categorical columns
numeric_cols = X_train_balanced.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = X_train_balanced.select_dtypes(include=['object']).columns

# Initialize the StandardScaler
scaler = StandardScaler()

# Scale only the numeric columns
X_train_balanced_numeric_scaled = scaler.fit_transform(X_train_balanced[numeric_cols])
X_test_numeric_scaled = scaler.transform(X_test[numeric_cols])

# Convert scaled arrays back to DataFrame
X_train_balanced_numeric_scaled = pd.DataFrame(X_train_balanced_numeric_scaled, columns=numeric_cols, index=X_train_balanced.index)
X_test_numeric_scaled = pd.DataFrame(X_test_numeric_scaled, columns=numeric_cols, index=X_test.index)

# Concatenate scaled numeric columns and categorical columns
X_train_balanced = pd.concat([X_train_balanced_numeric_scaled, X_train_balanced[categorical_cols].reset_index(drop=True)], axis=1)
X_test = pd.concat([X_test_numeric_scaled, X_test[categorical_cols].reset_index(drop=True)], axis=1)


# In[42]:


# Display scaled numeric features from the training set
print("Scaled Numeric Features in Training Set:")
print(X_train_balanced[numeric_cols].head())

# Display scaled numeric features from the test set
print("\nScaled Numeric Features in Test Set:")
print(X_test[numeric_cols].head())


# In[43]:


data.head(5)


# In[44]:


X_train_balanced_numeric_scaled.head(5)


# In[45]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Applying PCA without specifying the number of components
pca_full = PCA()
pca_full.fit(X_train_balanced_numeric_scaled)

# Calculate the cumulative variance explained by each component
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# Plotting the cumulative variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance by PCA Components')
plt.grid(True)
plt.show()


# In[46]:


# Select the number of components that capture 95% of the variance
n_components = next(i for i, cum_var in enumerate(cumulative_variance) if cum_var >= 0.95) + 1

# Apply PCA with the selected number of components
pca = PCA(n_components=n_components)
X_train_pca = pca.fit_transform(X_train_balanced_numeric_scaled)
X_test_pca = pca.transform(X_test_numeric_scaled)

# Optionally, print the number of components and shapes of the transformed sets
print(f"Number of PCA components chosen: {n_components}")
print(f"Shape of the PCA-transformed training set: {X_train_pca.shape}")
print(f"Shape of the PCA-transformed test set: {X_test_pca.shape}")


# In[47]:


X_test_numeric_scaled.head(2)


# In[48]:


from sklearn.model_selection import train_test_split

# Split the data into training and temporary data (70% for training, 30% for temp)
X_train_temp, X_temp, y_train_temp, y_temp = train_test_split(X_train_pca, y_train_balanced, test_size=0.3, random_state=58)

# Split the temporary data into testing and validation (50% for each)
X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.5, random_state=58)

# Print the shapes of the resulting datasets
print(f"Shape of X_train: {X_train_temp.shape}, y_train: {y_train_temp.shape}")
print(f"Shape of X_test: {X_test.shape}, y_test: {y_test.shape}")
print(f"Shape of X_val: {X_val.shape}, y_val: {y_val.shape}")


# ### Logistic Regression

# In[49]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize the Logistic Regression model
lr_model = LogisticRegression(random_state=42)

# Train the model on the training data
lr_model.fit(X_train_temp, y_train_temp)

# Make predictions on the test data
y_pred = lr_model.predict(X_test)

# Calculate accuracy and print the classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))


# #### Cross Validation and HyperParameter Tuning  Logistic Regression

# In[50]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train_temp, y_train_temp are your training data

# Initialize Logistic Regression model for cross-validation
lr_model_cv = LogisticRegression(random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(lr_model_cv, X_train_temp, y_train_temp, cv=5, scoring='accuracy')
print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean()}")

# Define the hyperparameter grid to search
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],  # Optimization algorithms
    'max_iter': [100, 200, 300]  # Maximum number of iterations for convergence
}

# Initialize GridSearchCV
grid_search_lr = GridSearchCV(LogisticRegression(random_state=42), param_grid_lr, cv=5, scoring='accuracy', verbose=1)

# Fit GridSearchCV to the training data
grid_search_lr.fit(X_train_temp, y_train_temp)

# Print the best parameters found
print("Best parameters found for Logistic Regression:", grid_search_lr.best_params_)
print("Best cross-validated accuracy for Logistic Regression:", grid_search_lr.best_score_)

# Train the final Logistic Regression model with the best parameters found
best_lr_model = LogisticRegression(**grid_search_lr.best_params_, random_state=42)
best_lr_model.fit(X_train_temp, y_train_temp)

# Predict on the test data and evaluate
y_pred_test_lr = best_lr_model.predict(X_test)
accuracy_test_lr = accuracy_score(y_test, y_pred_test_lr)
print(f"Logistic Regression Test Accuracy with Best Parameters: {accuracy_test_lr:.2f}")
print("Logistic Regression Test Classification Report with Best Parameters:")
print(classification_report(y_test, y_pred_test_lr))


# ### SVM

# In[51]:


from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# Create an SVM model
svm_model = SVC(random_state=58)

# Fit the model to the training data
svm_model.fit(X_train_temp, y_train_temp)

# Predict on the testing and validation data
y_pred_test_svm = svm_model.predict(X_test)
y_pred_val_svm = svm_model.predict(X_val)

# Calculate accuracy and generate a classification report for the testing data
accuracy_test_svm = accuracy_score(y_test, y_pred_test_svm)
print(f"SVM Test Accuracy: {accuracy_test_svm:.2f}")
print("SVM Testing Data Classification Report:")
print(classification_report(y_test, y_pred_test_svm))

# Calculate accuracy and generate a classification report for the validation data
accuracy_val_svm = accuracy_score(y_val, y_pred_val_svm)
print(f"SVM Validation Accuracy: {accuracy_val_svm:.2f}")
print("SVM Validation Data Classification Report:")
print(classification_report(y_val, y_pred_val_svm))


# #### Cross Validation for SVM

# In[64]:


from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np

# Create an SVM model
svm_model_cv = SVC(random_state=58)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(svm_model_cv, X_train_temp, y_train_temp, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of the cross-validation scores
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_mean:.2f}")
print(f"Standard Deviation of CV accuracy: {cv_std:.2f}")


# ### XGBoost

# In[52]:


import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train_temp, X_test, X_val, y_train_temp, y_test, and y_val are already defined

# Initialize XGBoost classifier
xgb_model = xgb.XGBClassifier(random_state=42)

# Fit the model on the training data
xgb_model.fit(X_train_temp, y_train_temp)

# Predict on the test and validation data
y_pred_test_xgb = xgb_model.predict(X_test)
y_pred_val_xgb = xgb_model.predict(X_val)

# Calculate accuracy and generate a classification report for the test data
accuracy_test_xgb = accuracy_score(y_test, y_pred_test_xgb)
print(f"XGBoost Test Accuracy: {accuracy_test_xgb:.2f}")
print("XGBoost Test Classification Report:")
print(classification_report(y_test, y_pred_test_xgb))

# Calculate accuracy and generate a classification report for the validation data
accuracy_val_xgb = accuracy_score(y_val, y_pred_val_xgb)
print(f"XGBoost Validation Accuracy: {accuracy_val_xgb:.2f}")
print("XGBoost Validation Classification Report:")
print(classification_report(y_val, y_pred_val_xgb))


# #### Cross Validation and Hyperparameter Tuning

# In[53]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid to search over
param_grid_xgb = {
    'n_estimators': [100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.7, 0.8],
    'subsample': [0.7, 0.8]
}

# Initialize GridSearchCV
grid_search_xgb = GridSearchCV(xgb.XGBClassifier(random_state=42), param_grid_xgb, cv=5, scoring='accuracy', verbose=1)

# Fit GridSearchCV to the training data
grid_search_xgb.fit(X_train_temp, y_train_temp)

# Print the best parameters found
print("Best parameters found for XGBoost:", grid_search_xgb.best_params_)
print("Best cross-validated accuracy for XGBoost:", grid_search_xgb.best_score_)

# Train the final XGBoost model with the best parameters found
best_xgb_model = xgb.XGBClassifier(**grid_search_xgb.best_params_, random_state=42)
best_xgb_model.fit(X_train_temp, y_train_temp)

# Evaluate the final XGBoost model on the test and validation sets
y_pred_test_best_xgb = best_xgb_model.predict(X_test)
y_pred_val_best_xgb = best_xgb_model.predict(X_val)
accuracy_test_best_xgb = accuracy_score(y_test, y_pred_test_best_xgb)
accuracy_val_best_xgb = accuracy_score(y_val, y_pred_val_best_xgb)

print(f"XGBoost Test Accuracy with Best Parameters: {accuracy_test_best_xgb:.2f}")
print(f"XGBoost Validation Accuracy with Best Parameters: {accuracy_val_best_xgb:.2f}")


# ### Decision Tree

# In[54]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train_temp, X_test, X_val, y_train_temp, y_test, and y_val are already defined

# Initialize the Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=42)

# Fit the classifier to the training data
dt_classifier.fit(X_train_temp, y_train_temp)

# Predict on the test and validation data
y_pred_test_dt = dt_classifier.predict(X_test)
y_pred_val_dt = dt_classifier.predict(X_val)

# Calculate accuracy and generate a classification report for the test data
accuracy_test_dt = accuracy_score(y_test, y_pred_test_dt)
print(f"Decision Tree Test Accuracy: {accuracy_test_dt:.2f}")
print("Decision Tree Test Classification Report:")
print(classification_report(y_test, y_pred_test_dt))

# Calculate accuracy and generate a classification report for the validation data
accuracy_val_dt = accuracy_score(y_val, y_pred_val_dt)
print(f"Decision Tree Validation Accuracy: {accuracy_val_dt:.2f}")
print("Decision Tree Validation Classification Report:")
print(classification_report(y_val, y_pred_val_dt))


# #### Cross Validation for Decision Tree

# In[62]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Initialize the Decision Tree classifier
dt_classifier_cv = DecisionTreeClassifier(random_state=42)

# Perform 5-fold cross-validation
cv_scores = cross_val_score(dt_classifier_cv, X_train_temp, y_train_temp, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of the cross-validation scores
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_mean:.2f}")
print(f"Standard Deviation of CV accuracy: {cv_std:.2f}")


# ### Naive Bayes

# In[55]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score

# Initialize Naive Bayes classifier
nb_classifier = GaussianNB()

# Fit the classifier to the training data
nb_classifier.fit(X_train_temp, y_train_temp)

# Predict on the test and validation data
y_pred_test_nb = nb_classifier.predict(X_test)
y_pred_val_nb = nb_classifier.predict(X_val)

# Evaluate the classifier
accuracy_test_nb = accuracy_score(y_test, y_pred_test_nb)
accuracy_val_nb = accuracy_score(y_val, y_pred_val_nb)
print(f"Naive Bayes Test Accuracy: {accuracy_test_nb}")
print(f"Naive Bayes Validation Accuracy: {accuracy_val_nb}")
print("Naive Bayes Test Classification Report:")
print(classification_report(y_test, y_pred_test_nb))
print("Naive Bayes Validation Classification Report:")
print(classification_report(y_val, y_pred_val_nb))


# #### Cross-Validation for Naive Bayes

# In[60]:


from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Initialize the Naive Bayes classifier
nb_classifier_cv = GaussianNB()

# Perform 5-fold cross-validation
cv_scores = cross_val_score(nb_classifier_cv, X_train_temp, y_train_temp, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of the cross-validation scores
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_mean:.2f}")
print(f"Standard Deviation of CV accuracy: {cv_std:.2f}")


# #### Hyperparameter Tuning for Naive Bayes

# In[61]:


from sklearn.model_selection import GridSearchCV

# Define the hyperparameter grid to search
param_grid_nb = {
    'var_smoothing': np.logspace(0,-9, num=100)
}

# Initialize GridSearchCV
grid_search_nb = GridSearchCV(GaussianNB(), param_grid_nb, cv=5, scoring='accuracy', verbose=1)

# Fit GridSearchCV to the training data
grid_search_nb.fit(X_train_temp, y_train_temp)

# Print the best parameters found
print("Best parameters found for Naive Bayes:", grid_search_nb.best_params_)
print("Best cross-validated accuracy for Naive Bayes:", grid_search_nb.best_score_)


# ### ADABOOST

# In[56]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# Initialize AdaBoost classifier with a Decision Tree as the base estimator
# Note: You can experiment with different base estimators.
ada_classifier = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)

# Fit the classifier to the training data
ada_classifier.fit(X_train_temp, y_train_temp)

# Predict on the test and validation data
y_pred_test_ada = ada_classifier.predict(X_test)
y_pred_val_ada = ada_classifier.predict(X_val)

# Evaluate the classifier
accuracy_test_ada = accuracy_score(y_test, y_pred_test_ada)
accuracy_val_ada = accuracy_score(y_val, y_pred_val_ada)
print(f"AdaBoost Test Accuracy: {accuracy_test_ada:.2f}")
print(f"AdaBoost Validation Accuracy: {accuracy_val_ada:.2f}")
print("AdaBoost Test Classification Report:")
print(classification_report(y_test, y_pred_test_ada))
print("AdaBoost Validation Classification Report:")
print(classification_report(y_val, y_pred_val_ada))


# ### Cross Validation and Hyper parameter tuning for ADABoost

# In[57]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

# Initialize AdaBoost classifier with a Decision Tree as the base estimator
# Note: You can experiment with different base estimators.
ada_classifier = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    random_state=42
)

# Define the hyperparameter grid to search over
param_grid_ada = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.01, 0.1, 0.2],
    'base_estimator__max_depth': [1, 2, 3],
}

# Initialize GridSearchCV
grid_search_ada = GridSearchCV(ada_classifier, param_grid_ada, cv=5, scoring='accuracy', verbose=1)

# Fit GridSearchCV to the training data
grid_search_ada.fit(X_train_temp, y_train_temp)

# Print the best parameters found
print("Best parameters found for AdaBoost:", grid_search_ada.best_params_)
print("Best cross-validated accuracy for AdaBoost:", grid_search_ada.best_score_)

# Train the final AdaBoost model with the best parameters found
best_ada_model = grid_search_ada.best_estimator_
best_ada_model.fit(X_train_temp, y_train_temp)

# Evaluate the final AdaBoost model on the test and validation sets
y_pred_test_best_ada = best_ada_model.predict(X_test)
y_pred_val_best_ada = best_ada_model.predict(X_val)
accuracy_test_best_ada = accuracy_score(y_test, y_pred_test_best_ada)
accuracy_val_best_ada = accuracy_score(y_val, y_pred_val_best_ada)

print(f"AdaBoost Test Accuracy with Best Parameters: {accuracy_test_best_ada:.2f}")
print(f"AdaBoost Validation Accuracy with Best Parameters: {accuracy_val_best_ada:.2f}")


# ### Random Forest

# In[58]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Assuming X_train_temp, X_test, X_val, y_train_temp, y_test, and y_val are already defined

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the classifier to the training data
rf_classifier.fit(X_train_temp, y_train_temp)

# Predict on the test and validation data
y_pred_test_rf = rf_classifier.predict(X_test)
y_pred_val_rf = rf_classifier.predict(X_val)

# Calculate accuracy and generate a classification report for the test data
accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)
print(f"Random Forest Test Accuracy: {accuracy_test_rf:.2f}")
print("Random Forest Test Classification Report:")
print(classification_report(y_test, y_pred_test_rf))

# Calculate accuracy and generate a classification report for the validation data
accuracy_val_rf = accuracy_score(y_val, y_pred_val_rf)
print(f"Random Forest Validation Accuracy: {accuracy_val_rf:.2f}")
print("Random Forest Validation Classification Report:")
print(classification_report(y_val, y_pred_val_rf))


# #### Cross Validation for Random Forest

# In[59]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import numpy as np

# Assuming X_train_temp and y_train_temp are your training data

# Initialize the Random Forest Classifier
rf_classifier_cv = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier_cv, X_train_temp, y_train_temp, cv=5, scoring='accuracy')

# Calculate the mean and standard deviation of the cross-validation scores
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

print(f"Cross-validation accuracy scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_mean:.2f}")
print(f"Standard Deviation of CV accuracy: {cv_std:.2f}")


# In[ ]:




