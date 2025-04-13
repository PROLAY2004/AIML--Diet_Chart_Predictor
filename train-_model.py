import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the dataset
df = pd.read_csv('./diet_dataset_with_meals_train.csv')

# Handle missing values
df['Medical Conditions'] = df['Medical Conditions'].fillna('None')
df['Allergies'] = df['Allergies'].fillna('None')

# Enhanced Feature Engineering
# Process Medical Conditions - create binary flags for common conditions
conditions = ['Diabetes', 'PCOS', 'Heart Disease', 'Hypertension', 
              'Asthma', 'Thyroid', 'Arthritis', 'Obesity', 'Stress']
for condition in conditions:
    df[f'Condition_{condition}'] = df['Medical Conditions'].str.contains(condition, na=False).astype(int)

# Process Allergies - create binary flags for common allergies
allergens = ['Milk', 'Nuts', 'Peanuts', 'Gluten', 'Soy', 
             'Eggs', 'Seafood', 'Sesame', 'Wheat']
for allergen in allergens:
    df[f'Allergy_{allergen}'] = df['Allergies'].str.contains(allergen, na=False).astype(int)

# Calculate BMI and BMI difference
df['BMI'] = df['Weight'] / ((df['Height']/100) ** 2)
df['Target_BMI'] = df['Target Weight'] / ((df['Height']/100) ** 2)
df['BMI_Diff'] = df['BMI'] - df['Target_BMI']

# Features and target
X = df[['Age', 'Gender', 'Height', 'Weight', 'Target Weight', 'Fitness Goal', 
        'BMI', 'Target_BMI', 'BMI_Diff'] + 
       [f'Condition_{c}' for c in conditions] + 
       [f'Allergy_{a}' for a in allergens]]
y = df['Recommended Diet Plan']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipeline
numeric_features = ['Age', 'Height', 'Weight', 'Target Weight', 
                   'BMI', 'Target_BMI', 'BMI_Diff']
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_features = ['Gender', 'Fitness Goal']
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

binary_features = [f'Condition_{c}' for c in conditions] + [f'Allergy_{a}' for a in allergens]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features),
        ('binary', 'passthrough', binary_features)
    ])

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        subsample=0.8
    ))
])

print("Training Progress...")
# Train the model
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy:.2f}")

# Save the model
joblib.dump(model, 'diet_recommender_model.joblib')
print("Model saved as 'diet_recommender_model.joblib'")

# Feature importance analysis
if hasattr(model.named_steps['classifier'], 'feature_importances_'):
    feature_names = (numeric_features + 
                    list(model.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].get_feature_names_out(categorical_features)) +
                    binary_features)
    importances = model.named_steps['classifier'].feature_importances_
    print("\nTop important features:")
    for idx in np.argsort(importances)[-10:]:
        print(f"{feature_names[idx]}: {importances[idx]:.4f}")
