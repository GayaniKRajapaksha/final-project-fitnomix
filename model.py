import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score, classification_report
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

# --- Step 1: Load the dataset ---
print("Loading dataset...")
df = pd.read_csv('meals.csv')
# Compat: accept either 'budget' or 'price'
if 'budget' not in df.columns and 'price' in df.columns:
    df = df.rename(columns={'price': 'budget'})
print(f"Dataset shape: {df.shape}")

# --- Step 2: Feature Engineering ---
print("Performing feature engineering...")

# Create BMI feature
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)

# Create BMI x Activity interaction feature
activity_map = {
    'sedentary': 1.0, 'light': 1.5, 'moderate': 2.0, 'active': 3.0,
    'low': 1.0, 'medium': 2.0, 'high': 3.0
}
df['activity_numeric'] = df['physical_activity_level'].map(activity_map).fillna(2.0)
df['BMI_x_activity'] = df['BMI'] * df['activity_numeric']

# Create age groups for better categorical encoding
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])

# Create health risk score
df['health_risk_score'] = (
    df['cholesterol'] / 200 + 
    df['blood_pressure'] / 120 + 
    df['glucose'] / 100
) / 3

# Define enhanced features
features = [
    'age', 'gender', 'weight', 'height', 'disease_type', 'severity',
    'physical_activity_level', 'cholesterol', 'blood_pressure', 'glucose',
    'dietary_restrictions', 'dietary_habits', 'meal_type', 'budget',
    'BMI', 'BMI_x_activity', 'age_group', 'health_risk_score'
]

# Targets remain the same
targets = ['diet_recommendation', 'weekly_exercise_hours', 'meal', 'ingredients']

X = df[features]
y = df[targets]

print(f"Features: {len(features)}")
print(f"Targets: {targets}")

# --- Step 3: Identify categorical and numerical features ---
categorical_features_X = X.select_dtypes(include=['object', 'category']).columns
numerical_features_X = X.select_dtypes(include=['number']).columns

print(f"Categorical features: {list(categorical_features_X)}")
print(f"Numerical features: {list(numerical_features_X)}")

# --- Step 4: Enhanced preprocessing ---
print("Setting up preprocessing...")

# Use RobustScaler for numerical features (less sensitive to outliers)
preprocessor_X = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_features_X),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features_X)
    ])

# --- Step 5: Preprocess target variables ---
print("Preprocessing targets...")
label_encoders = {}
target_encoders = {}
y_processed = {}

# Process weekly_exercise_hours (numerical regression target)
y_processed['weekly_exercise_hours'] = y['weekly_exercise_hours'].values.astype(np.float32)

# Process categorical targets
categorical_targets = ['diet_recommendation', 'meal', 'ingredients']
for target_name in categorical_targets:
    print(f"  Processing {target_name}...")
    
    # Check for missing values
    missing_count = y[target_name].isnull().sum()
    if missing_count > 0:
        print(f"    Warning: {missing_count} missing values found, filling with 'Unknown'")
        y[target_name] = y[target_name].fillna('Unknown')
    
    # Check unique values
    unique_values = y[target_name].unique()
    print(f"    Unique values: {len(unique_values)} - {list(unique_values)}")
    
    # Create and fit label encoder
    le = LabelEncoder()
    integer_encoded = le.fit_transform(y[target_name])
    label_encoders[target_name] = le
    
    # Create and fit one-hot encoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    integer_encoded_reshaped = integer_encoded.reshape(-1, 1)
    onehot_encoded = ohe.fit_transform(integer_encoded_reshaped)
    target_encoders[target_name] = ohe
    y_processed[target_name] = onehot_encoded
    
    print(f"    Encoded shape: {onehot_encoded.shape}")

# --- Step 6: Split data with stratification ---
print("Splitting data...")
X_train, X_test, y_train_orig, y_test_orig = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['diet_recommendation']
)

# --- Step 7: Apply preprocessing ---
print("Applying preprocessing...")
X_train_processed = preprocessor_X.fit_transform(X_train)
X_test_processed = preprocessor_X.transform(X_test)

print(f"Processed training shape: {X_train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")

# Prepare training and test targets
y_train = {
    'weekly_exercise_hours': y_train_orig['weekly_exercise_hours'].values.astype(np.float32),
}

y_test = {
    'weekly_exercise_hours': y_test_orig['weekly_exercise_hours'].values.astype(np.float32),
}

for target_name in categorical_targets:
    integer_train = label_encoders[target_name].transform(y_train_orig[target_name]).reshape(-1, 1)
    integer_test = label_encoders[target_name].transform(y_test_orig[target_name]).reshape(-1, 1)
    y_train[target_name] = target_encoders[target_name].transform(integer_train)
    y_test[target_name] = target_encoders[target_name].transform(integer_test)

# --- Step 8: Enhanced model training with hyperparameter tuning ---
print("\nTraining enhanced models...")

# Train regression model for weekly exercise hours
print("Training exercise hours regressor...")
regressor = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    alpha=0.001,
    learning_rate='adaptive',
    max_iter=3000,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
regressor.fit(X_train_processed, y_train['weekly_exercise_hours'])

# Train classification models with enhanced parameters
print("Training classification models...")
classifiers = {}
for target_name in categorical_targets:
    print(f"  Training {target_name} classifier...")
    classifier = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=3000,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )
    classifier.fit(X_train_processed, y_train[target_name])
    classifiers[target_name] = classifier

print("Training complete.")

# --- Step 9: Make predictions ---
print("\nMaking predictions...")

# Predict weekly exercise hours
y_pred_weekly = regressor.predict(X_test_processed)

# Predict categorical targets
y_pred = {
    'weekly_exercise_hours': y_pred_weekly
}

for target_name in categorical_targets:
    print(f"  Predicting {target_name}...")
    y_pred_proba = classifiers[target_name].predict(X_test_processed)
    
    try:
        # Debug: Check prediction shape and content
        print(f"    Prediction shape: {y_pred_proba.shape}")
        print(f"    Prediction type: {type(y_pred_proba)}")
        
        # Convert to integer indices first
        if hasattr(y_pred_proba, 'ndim') and y_pred_proba.ndim == 2:
            # Multi-class prediction - get class with highest probability
            class_indices = np.argmax(y_pred_proba, axis=1)
        else:
            # Single class prediction
            class_indices = y_pred_proba.ravel()
        
        # Ensure all values are valid integers
        class_indices = np.array(class_indices, dtype=int)
        
        # Check for any invalid indices
        max_valid_index = len(target_encoders[target_name].categories_[0]) - 1
        invalid_indices = (class_indices < 0) | (class_indices > max_valid_index)
        
        if np.any(invalid_indices):
            print(f"    Warning: Found {np.sum(invalid_indices)} invalid indices")
            # Replace invalid indices with 0
            class_indices[invalid_indices] = 0
        
        # Convert to one-hot encoding
        class_indices_reshaped = class_indices.reshape(-1, 1)
        onehot_pred = target_encoders[target_name].transform(class_indices_reshaped)
        
        # Convert back to labels
        y_pred[target_name] = label_encoders[target_name].inverse_transform(
            target_encoders[target_name].inverse_transform(onehot_pred).ravel()
        )
        
        print(f"    Successfully predicted {len(y_pred[target_name])} samples")
        
    except Exception as e:
        print(f"    Error predicting {target_name}: {e}")
        # Fallback: use the most common class
        most_common_class = y_train_orig[target_name].mode()[0]
        y_pred[target_name] = [most_common_class] * len(y_test_orig)
        print(f"    Using fallback prediction: {most_common_class}")

# Create prediction DataFrame
y_pred_df = pd.DataFrame(y_pred, index=y_test_orig.index)

# --- Step 10: Enhanced evaluation with cross-validation ---
print("\n--- ENHANCED MODEL EVALUATION ---")

# Cross-validation for regression
print("Performing cross-validation...")
cv_scores_reg = cross_val_score(regressor, X_train_processed, y_train['weekly_exercise_hours'], 
                               cv=5, scoring='neg_mean_absolute_error')
print(f"Exercise Hours CV MAE: {-cv_scores_reg.mean():.4f} (+/- {cv_scores_reg.std() * 2:.4f})")

# Cross-validation for classification
for target_name in categorical_targets:
    cv_scores_clf = cross_val_score(classifiers[target_name], X_train_processed, y_train[target_name], 
                                   cv=5, scoring='accuracy')
    print(f"{target_name} CV Accuracy: {cv_scores_clf.mean():.4f} (+/- {cv_scores_clf.std() * 2:.4f})")

# --- Step 11: Final metrics ---
print("\n--- FINAL MODEL ACCURACY AND METRICS ---")

# Weekly Exercise Hours (Regression)
mae_weekly = mean_absolute_error(y_test_orig['weekly_exercise_hours'], y_pred_df['weekly_exercise_hours'])
r2_weekly = r2_score(y_test_orig['weekly_exercise_hours'], y_pred_df['weekly_exercise_hours'])
print(f"Weekly Exercise Hours (Regression):")
print(f"  Mean Absolute Error (MAE): {mae_weekly:.4f}")
print(f"  R-squared (R2): {r2_weekly:.4f}")

# Categorical Targets (Classification) with detailed reports
for target_name in categorical_targets:
    accuracy = accuracy_score(y_test_orig[target_name], y_pred_df[target_name])
    print(f"\n{target_name.replace('_', ' ').title()} (Classification):")
    print(f"  Accuracy: {accuracy:.4f}")
    
    # Print detailed classification report
    print("  Detailed Report:")
    report = classification_report(y_test_orig[target_name], y_pred_df[target_name], 
                                 target_names=label_encoders[target_name].classes_)
    for line in report.split('\n'):
        if line.strip():
            print(f"    {line}")

# --- Step 12: Save enhanced models and preprocessors ---
print("\nSaving enhanced models...")
joblib.dump(regressor, 'regressor_model.pkl')
for target_name in categorical_targets:
    joblib.dump(classifiers[target_name], f'{target_name}_classifier.pkl')
joblib.dump(preprocessor_X, 'preprocessor_X.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(target_encoders, 'target_encoders.pkl')

print("Enhanced models and preprocessors saved successfully.")

# --- Step 13: Display example predictions ---
print("\n--- Example Predictions vs. Actual (first 5 samples) ---")
print("Actual:")
print(y_test_orig.head())
print("\nPredicted:")
print(y_pred_df.head())

print("\nModel training and evaluation complete!")
