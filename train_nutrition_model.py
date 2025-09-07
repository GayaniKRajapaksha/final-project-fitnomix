import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

print("Training Nutrition Prediction Model...")

# Load dataset
df = pd.read_csv('meals.csv')
# Compat: accept either 'budget' or 'price'
if 'budget' not in df.columns and 'price' in df.columns:
    df = df.rename(columns={'price': 'budget'})

# Feature engineering for nutrition
df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
activity_map = {
    'sedentary': 1.0, 'light': 1.5, 'moderate': 2.0, 'active': 3.0,
    'low': 1.0, 'medium': 2.0, 'high': 3.0
}
df['activity_numeric'] = df['physical_activity_level'].map(activity_map).fillna(2.0)
df['BMI_x_activity'] = df['BMI'] * df['activity_numeric']
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 50, 65, 100], 
                        labels=['Young', 'Adult', 'Middle', 'Senior', 'Elderly'])
df['health_risk_score'] = (
    df['cholesterol'] / 200 + 
    df['blood_pressure'] / 120 + 
    df['glucose'] / 100
) / 3

# Features for nutrition prediction (exclude meal_type and dietary_habits as they may not be available during prediction)
nutrition_features = [
    'age', 'gender', 'weight', 'height', 'disease_type', 'severity',
    'physical_activity_level', 'cholesterol', 'blood_pressure', 'glucose',
    'dietary_restrictions', 'budget', 'BMI', 'BMI_x_activity', 'age_group', 'health_risk_score'
]

# Nutrition targets (assuming these exist in your dataset or need to be created)
# If these don't exist, we'll create synthetic nutrition data based on user profile
nutrition_targets = ['calories', 'protein_g', 'carbs_g', 'fat_g']

# Check if nutrition targets exist in dataset
if all(target in df.columns for target in nutrition_targets):
    print("Using existing nutrition data from dataset")
    X_nutrition = df[nutrition_features]
    y_nutrition = df[nutrition_targets]
else:
    print("Creating synthetic nutrition data based on user profiles")
    # Create synthetic nutrition data based on user characteristics
    def generate_nutrition_data(row):
        # Base calories based on age, weight, and activity
        base_calories = 2000 if row['age'] < 50 else 1800
        activity_multiplier = row['activity_numeric']
        weight_factor = row['weight'] / 70  # Normalize to 70kg
        
        calories = int(base_calories * activity_multiplier * weight_factor)
        
        # Macronutrient distribution
        protein_g = int(calories * 0.25 / 4)  # 25% protein
        carbs_g = int(calories * 0.45 / 4)    # 45% carbs
        fat_g = int(calories * 0.30 / 9)      # 30% fat
        
        return pd.Series([calories, protein_g, carbs_g, fat_g])
    
    X_nutrition = df[nutrition_features]
    y_nutrition = df.apply(generate_nutrition_data, axis=1)
    y_nutrition.columns = nutrition_targets

# Preprocessing
categorical_features = X_nutrition.select_dtypes(include=['object', 'category']).columns
numerical_features = X_nutrition.select_dtypes(include=['number']).columns

nutrition_preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first'), categorical_features)
    ]
)

# Split data
X_train_nut, X_test_nut, y_train_nut, y_test_nut = train_test_split(
    X_nutrition, y_nutrition, test_size=0.2, random_state=42
)

# Apply preprocessing
X_train_nut_processed = nutrition_preprocessor.fit_transform(X_train_nut)
X_test_nut_processed = nutrition_preprocessor.transform(X_test_nut)

# Train nutrition model (multi-output regression)
print("Training nutrition prediction model...")
nutrition_model = MLPRegressor(
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

nutrition_model.fit(X_train_nut_processed, y_train_nut)

# Evaluate nutrition model
y_pred_nutrition = nutrition_model.predict(X_test_nut_processed)

print("\nNutrition Model Performance:")
for i, target in enumerate(nutrition_targets):
    mae = mean_absolute_error(y_test_nut.iloc[:, i], y_pred_nutrition[:, i])
    r2 = r2_score(y_test_nut.iloc[:, i], y_pred_nutrition[:, i])
    print(f"{target}:")
    print(f"  MAE: {mae:.2f}")
    print(f"  R²: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(nutrition_model, X_train_nut_processed, y_train_nut, 
                           cv=5, scoring='neg_mean_absolute_error')
print(f"\nCross-validation MAE: {-cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")

# Save nutrition model
joblib.dump(nutrition_model, 'nutrition_model.pkl')
joblib.dump(nutrition_preprocessor, 'nutrition_preprocessor.pkl')

print("Nutrition model saved successfully!") 