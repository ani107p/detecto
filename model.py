import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
dataframe = pd.read_csv('insuranceFraud.csv')
dataframe.replace('?', np.nan, inplace=True)

# Fill missing values with mode for categorical variables
dataframe['collision_type'] = dataframe['collision_type'].fillna(dataframe['collision_type'].mode()[0])
dataframe['property_damage'] = dataframe['property_damage'].fillna(dataframe['property_damage'].mode()[0])
dataframe['police_report_available'] = dataframe['police_report_available'].fillna(dataframe['police_report_available'].mode()[0])
dataframe['authorities_contacted'] = dataframe['authorities_contacted'].fillna(dataframe['authorities_contacted'].mode()[0])

# Drop unneeded columns
df = dataframe.drop(columns=['policy_number', 'policy_csl', 'policy_bind_date', 'policy_state', 'insured_zip', 'incident_location', 'incident_date',
                             'incident_state', 'incident_city', 'insured_hobbies', 'auto_make', 'auto_model', 'auto_year'])

# Further drop columns as per your previous steps
df.drop(columns=['age', 'total_claim_amount'], inplace=True)

# Split into features and target
X = df.drop('fraud_reported', axis=1)
y = df['fraud_reported']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Separate categorical and numerical columns
dataframe_cat_train = X_train.select_dtypes(include=['object'])
dataframe_num_train = X_train.select_dtypes(include=['int64'])
dataframe_cat_test = X_test.select_dtypes(include=['object'])
dataframe_num_test = X_test.select_dtypes(include=['int64'])

# Fill missing values in X_test (although you have already handled this before splitting)
dataframe_cat_test['collision_type'] = dataframe_cat_test['collision_type'].fillna(dataframe_cat_test['collision_type'].mode()[0])
dataframe_cat_test['property_damage'] = dataframe_cat_test['property_damage'].fillna(dataframe_cat_test['property_damage'].mode()[0])
dataframe_cat_test['police_report_available'] = dataframe_cat_test['police_report_available'].fillna(dataframe_cat_test['police_report_available'].mode()[0])
dataframe_cat_test['authorities_contacted'] = dataframe_cat_test['authorities_contacted'].fillna(dataframe_cat_test['authorities_contacted'].mode()[0])

# Encode categorical variables using one-hot encoding
dataframe_cat_train = pd.get_dummies(dataframe_cat_train, drop_first=True)
dataframe_cat_test = pd.get_dummies(dataframe_cat_test, drop_first=True)

# Ensure both train and test have the same dummy columns
dataframe_cat_train, dataframe_cat_test = dataframe_cat_train.align(dataframe_cat_test, join='left', axis=1, fill_value=0)

# Save the one-hot encoded column names
one_hot_columns = dataframe_cat_train.columns

# Scale numerical features
scaler = StandardScaler()
scaled_data_train = scaler.fit_transform(dataframe_num_train)
scaled_data_test = scaler.transform(dataframe_num_test)

scaled_dataframe_num_train = pd.DataFrame(data=scaled_data_train, columns=dataframe_num_train.columns, index=X_train.index)
scaled_dataframe_num_test = pd.DataFrame(data=scaled_data_test, columns=dataframe_num_test.columns, index=X_test.index)

# Combine scaled numerical and encoded categorical features
X_train_processed = pd.concat([scaled_dataframe_num_train, dataframe_cat_train], axis=1)
X_test_processed = pd.concat([scaled_dataframe_num_test, dataframe_cat_test], axis=1)

# Verify dimensions match
print(X_train_processed.shape)
print(X_test_processed.shape)

# Now fit models with the correctly processed data
etc = ExtraTreesClassifier()
etc.fit(X_train_processed, y_train)

# Predictions
y_pred = etc.predict(X_test_processed)

# Evaluation
acc_etc = accuracy_score(y_test, y_pred)
print(f"Training Accuracy of Extra Trees Classifier is {accuracy_score(y_train, etc.predict(X_train_processed))}")
print(f"Test Accuracy of Extra Trees Classifier is {acc_etc} \n")
print(f"Confusion Matrix :- \n{confusion_matrix(y_test, y_pred)}\n")
print(f"Classification Report :- \n {classification_report(y_test, y_pred)}")

# Save the model, scaler, and one-hot encoded column names
with open('model.pkl', 'wb') as model_file, open('scaler.pkl', 'wb') as scaler_file, open('one_hot_columns.pkl', 'wb') as columns_file:
    pickle.dump(etc, model_file)
    pickle.dump(scaler, scaler_file)
    pickle.dump(one_hot_columns, columns_file)
