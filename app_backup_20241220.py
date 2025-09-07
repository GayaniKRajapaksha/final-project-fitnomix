from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning
import warnings
import mysql.connector
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
import sys

# Ensure UTF-8 console output on Windows
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a secure secret key

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # Change to your MySQL username
    'password': '',  # Change to your MySQL password
    'database': 'meal_planner'
}

def test_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        if conn.is_connected():
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT DATABASE();")
            db_name = cursor.fetchone()
            print("✓ Successfully connected to MySQL database:", db_name['DATABASE()'])
            
            # Test if users table exists
            cursor.execute("SHOW TABLES LIKE 'users';")
            if cursor.fetchone():
                print("✓ Users table found")
            else:
                print("✗ Users table not found. Please run the database.sql script")
                sys.exit(1)
                
            cursor.close()
            conn.close()
            return True
    except mysql.connector.Error as err:
        print("✗ Error connecting to MySQL database:", err)
        print("Please check your database configuration and ensure MySQL is running.")
        sys.exit(1)
    return False

# Load the trained models and preprocessors (enhanced models from model.py)
use_enhanced_models = False
nutrition_model = None
classifiers = {}
label_encoders = None
target_encoders = None
regressor = None
preprocessor_X = None

print("Loading enhanced models...")

# Try to load enhanced models first (from model.py)
try:
    regressor = joblib.load('regressor_model.pkl')
    preprocessor_X = joblib.load('preprocessor_X.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
    target_encoders = joblib.load('target_encoders.pkl')
    
    # Load individual classifiers
    classifiers = {
        'diet_recommendation': joblib.load('diet_recommendation_classifier.pkl'),
        'meal': joblib.load('meal_classifier.pkl'),
        'ingredients': joblib.load('ingredients_classifier.pkl')
    }
    
    use_enhanced_models = True
    print("✓ Loaded enhanced models from model.py")
    
    # Try to load nutrition model
    try:
        nutrition_model = joblib.load('nutrition_model.pkl')
        # Load matching preprocessor if available
        try:
            nutrition_preprocessor = joblib.load('nutrition_preprocessor.pkl')
        except Exception:
            nutrition_preprocessor = None
        print("✓ Loaded nutrition model")
    except Exception:
        print("ℹ Nutrition model not available")
        
except Exception as e:
    print("ℹ Enhanced models not available, trying legacy models:", e)
    try:
        # Fallback to legacy models
        clf_pipeline = joblib.load('diet_recommendation_model.pkl')
        reg1_pipeline = joblib.load('exercise_hours_model.pkl')
        diet_label_encoder = joblib.load('diet_label_encoder.pkl')
        print("✓ Loaded legacy pipeline-based models")
        
        # Try to load nutrition model
        try:
            nutrition_model = joblib.load('nutrition_model.pkl')
            print("✓ Loaded nutrition model")
        except Exception:
            print("ℹ Nutrition model not available")
            
    except Exception as le:
        print("✗ Error loading all models:", le)
        sys.exit(1)

# Load meals dataset once for estimating meal costs
try:
    meals_df = pd.read_csv('meals.csv')
    # Compat: accept either 'budget' or 'price'
    if 'budget' not in meals_df.columns and 'price' in meals_df.columns:
        meals_df = meals_df.rename(columns={'price': 'budget'})
    # Ensure expected columns exist
    if not {'meal', 'budget'}.issubset(meals_df.columns):
        print("✗ meals.csv missing required columns 'meal' and 'budget' (or 'price')")
        meals_df = pd.DataFrame(columns=['meal', 'budget'])
    else:
        # Normalize meal names to consistent strings
        meals_df['meal'] = meals_df['meal'].astype(str)
        meals_df['budget'] = pd.to_numeric(meals_df['budget'], errors='coerce')
        # Coerce nutrition columns if present
        for col in ['calories', 'protein_g', 'carbs_g', 'fat_g']:
            if col in meals_df.columns:
                meals_df[col] = pd.to_numeric(meals_df[col], errors='coerce')
    print("✓ Loaded meals.csv for budget and nutrition lookup")
except Exception as e:
    print("✗ Could not load meals.csv for budget estimation:", e)
    meals_df = pd.DataFrame(columns=['meal', 'budget', 'calories', 'protein_g', 'carbs_g', 'fat_g'])

# Database connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(**db_config)
        return conn
    except mysql.connector.Error as err:
        print("Error connecting to database:", err)
        return None

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login first.')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

# Admin required decorator
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_role' not in session or session['user_role'] != 'admin':
            flash('Admin access required.')
            return redirect(url_for('home'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Hash the password
        hashed_password = generate_password_hash(password)
        
        conn = get_db_connection()
        if not conn:
            flash('Error connecting to database. Please try again later.')
            return render_template('register.html')
            
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute(
                "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)",
                (username, email, hashed_password)
            )
            conn.commit()
            flash('Registration successful! Please login.')
            return redirect(url_for('login'))
        except mysql.connector.Error as err:
            flash(f'Registration failed: {err}')
        finally:
            cursor.close()
            conn.close()
            
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        print(f"\nLogin attempt for username: {username}")  # Debug log
        
        conn = get_db_connection()
        if not conn:
            print("Database connection failed")  # Debug log
            flash('Error connecting to database. Please try again later.')
            return render_template('login.html')
            
        cursor = conn.cursor(dictionary=True)
        
        try:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            user = cursor.fetchone()
            
            if user:
                print(f"User found: {user}")  # Debug log
                print(f"User role: {user['role']}")  # Debug log
                is_valid = check_password_hash(user['password'], password)
                print(f"Password valid: {is_valid}")  # Debug log
                
                if is_valid:
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    session['user_role'] = user['role']
                    
                    print(f"Session data set: {session}")  # Debug log
                    
                    if user['role'] == 'admin':
                        print("Redirecting to admin dashboard")  # Debug log
                        return redirect(url_for('admin_dashboard'))
                    else:
                        print("Redirecting to home page")  # Debug log
                        flash('Welcome back! You have successfully logged in.')
                        return redirect(url_for('home'))
                else:
                    print("Invalid password")  # Debug log
            else:
                print("User not found")  # Debug log
            
            flash('Invalid username or password')
        except mysql.connector.Error as err:
            print(f"Database error: {err}")  # Debug log
            flash(f'Login failed: {err}')
        finally:
            cursor.close()
            conn.close()
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.')
    return redirect(url_for('home'))

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    print(f"\nAdmin dashboard access attempt")
    print(f"Session data: {session}")
    print(f"User role: {session.get('user_role')}")
    
    if 'user_role' not in session or session['user_role'] != 'admin':
        print("Access denied: Not an admin")
        flash('Admin access required.')
        return redirect(url_for('home'))
        
    conn = get_db_connection()
    if not conn:
        print("Database connection failed")
        flash('Error connecting to database. Please try again later.')
        return redirect(url_for('home'))
        
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SELECT * FROM users WHERE role = 'customer'")
        customers = cursor.fetchall()
        print(f"Found {len(customers)} customers")
        return render_template('admin_dashboard.html', customers=customers)
    except Exception as e:
        print(f"Error in admin dashboard: {e}")
        flash('Error loading customer data.')
        return redirect(url_for('home'))
    finally:
        cursor.close()
        conn.close()

@app.route('/customer/dashboard')
@login_required
def customer_dashboard():
    return render_template('prediction.html')

@app.route('/customer/index')
@login_required
def customer_index():
    return render_template('customer_dashboard.html')

@app.route('/profile')
@login_required
def profile():
    conn = get_db_connection()
    if not conn:
        flash('Error connecting to database. Please try again later.')
        return redirect(url_for('home'))
        
    cursor = conn.cursor(dictionary=True)
    
    try:
        # Get user info
        cursor.execute("SELECT * FROM users WHERE id = %s", (session['user_id'],))
        user = cursor.fetchone()
        
        # Get latest health progress
        cursor.execute("""
            SELECT * FROM health_progress 
            WHERE user_id = %s 
            ORDER BY measurement_date DESC 
            LIMIT 1
        """, (session['user_id'],))
        latest_health = cursor.fetchone()
        
        # Get health progress history for chart
        cursor.execute("""
            SELECT weight, height, measurement_date 
            FROM health_progress 
            WHERE user_id = %s 
            ORDER BY measurement_date ASC
        """, (session['user_id'],))
        health_history = cursor.fetchall()
        
        # Format data for chart
        health_progress = {
            'dates': [],
            'weights': [],
            'bmis': []
        }
        
        for record in health_history:
            health_progress['dates'].append(record['measurement_date'].strftime('%Y-%m-%d'))
            health_progress['weights'].append(record['weight'])
            # Calculate BMI: weight(kg) / (height(m))²
            bmi = record['weight'] / ((record['height']/100) ** 2)
            health_progress['bmis'].append(round(bmi, 2))
        
        # Get recent meal plans
        cursor.execute("""
            SELECT * FROM meal_plans 
            WHERE user_id = %s 
            ORDER BY created_at DESC 
            LIMIT 5
        """, (session['user_id'],))
        meal_plans = cursor.fetchall()
        
        return render_template('profile.html',
                             user=user,
                             latest_health=latest_health,
                             health_progress=health_progress,
                             meal_plans=meal_plans)
                             
    except Exception as e:
        print(f"Error in profile route: {e}")
        flash('Error loading profile data.')
        return redirect(url_for('home'))
    finally:
        cursor.close()
        conn.close()

@app.route('/update_health', methods=['POST'])
@login_required
def update_health():
    
    if request.method == 'POST':
        try:
            weight = float(request.form['weight'])
            height = float(request.form['height'])
            cholesterol = float(request.form['cholesterol'])
            blood_pressure = float(request.form['blood_pressure'])
            glucose = float(request.form['glucose'])
            physical_activity_level = request.form['physical_activity_level']
            
            # Calculate BMI
            bmi = weight / ((height/100) ** 2)
            
            conn = get_db_connection()
            if not conn:
                flash('Error connecting to database. Please try again later.')
                return redirect(url_for('profile'))
                
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO health_progress 
                    (user_id, weight, height, bmi, cholesterol, blood_pressure, glucose, physical_activity_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    session['user_id'],
                    weight,
                    height,
                    bmi,
                    cholesterol,
                    blood_pressure,
                    glucose,
                    physical_activity_level
                ))
                
                conn.commit()
                flash('Health data updated successfully!')
                
            except mysql.connector.Error as err:
                print(f"Database error: {err}")
                flash('Error updating health data.')
                
            finally:
                cursor.close()
                conn.close()
                
        except ValueError as e:
            flash('Invalid input values.')
            
    return redirect(url_for('profile'))

# Update the predict route to save meal plans
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    try:
        # Get form data
        data = {
            'age': float(request.form['age']),
            'gender': request.form['gender'],
            'weight': float(request.form['weight']),
            'height': float(request.form['height']),
            'disease_type': request.form['disease_type'],
            'severity': request.form['severity'],
            'physical_activity_level': request.form['physical_activity_level'],
            'cholesterol': float(request.form['cholesterol']),
            'blood_pressure': float(request.form['blood_pressure']),
            'glucose': float(request.form['glucose']),
            'dietary_restrictions': request.form['dietary_restrictions'],
            'dietary_habits': request.form['dietary_habits'],
            'meal_type': request.form['meal_type'],
            'budget': float(request.form['budget'])  # Use 'budget' to match enhanced models
        }
        
        # Keep original budget value for calculations
        original_budget = float(request.form['budget'])

        # Feature engineering (matching model.py)
        bmi = data['weight'] / ((data['height'] / 100) ** 2)
        activity_map = {
            'sedentary': 1.0, 'light': 1.5, 'moderate': 2.0, 'active': 3.0,
            'low': 1.0, 'medium': 2.0, 'high': 3.0
        }
        activity_numeric = activity_map.get(data['physical_activity_level'].lower(), 2.0)
        bmi_x_activity = bmi * activity_numeric
        
        # Create age group
        age = data['age']
        if age <= 25:
            age_group = 'Young'
        elif age <= 35:
            age_group = 'Adult'
        elif age <= 50:
            age_group = 'Middle'
        elif age <= 65:
            age_group = 'Senior'
        else:
            age_group = 'Elderly'
        
        # Create health risk score
        health_risk_score = (
            data['cholesterol'] / 200 + 
            data['blood_pressure'] / 120 + 
            data['glucose'] / 100
        ) / 3

        # Create enhanced input with all features
        enriched_input = dict(data)
        enriched_input['BMI'] = bmi
        enriched_input['BMI_x_activity'] = bmi_x_activity
        enriched_input['age_group'] = age_group
        enriched_input['health_risk_score'] = health_risk_score

        # Convert to DataFrame
        input_df = pd.DataFrame([enriched_input])

        # Suppress warnings during prediction
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=DataConversionWarning)
            
            # Make predictions using enhanced models if available
            try:
                predictions = {}
                if use_enhanced_models:
                    print("Using enhanced models for prediction...")
                    
                    # Preprocess input
                    X_processed = preprocessor_X.transform(input_df)
                    
                    # Weekly exercise hours
                    weekly_exercise = regressor.predict(X_processed)[0]
                    predictions['weekly_exercise_hours'] = round(float(weekly_exercise), 2)

                    # Get predictions for categorical targets
                    for target_name in ['diet_recommendation', 'meal', 'ingredients']:
                        if target_name in classifiers and label_encoders and target_encoders:
                            y_pred_proba = classifiers[target_name].predict(X_processed)
                            
                            try:
                                # Convert to integer indices first
                                if hasattr(y_pred_proba, 'ndim') and y_pred_proba.ndim == 2:
                                    class_indices = np.argmax(y_pred_proba, axis=1)
                                else:
                                    class_indices = y_pred_proba.ravel()
                                
                                # Ensure all values are valid integers
                                class_indices = np.array(class_indices, dtype=int)
                                
                                # Check for any invalid indices
                                max_valid_index = len(target_encoders[target_name].categories_[0]) - 1
                                invalid_indices = (class_indices < 0) | (class_indices > max_valid_index)
                                
                                if np.any(invalid_indices):
                                    class_indices[invalid_indices] = 0
                                
                                # Convert to one-hot encoding
                                class_indices_reshaped = class_indices.reshape(-1, 1)
                                onehot_pred = target_encoders[target_name].transform(class_indices_reshaped)
                                
                                # Convert back to labels
                                predictions[target_name] = label_encoders[target_name].inverse_transform(
                                    target_encoders[target_name].inverse_transform(onehot_pred).ravel()
                                )[0]
                                
                            except Exception as e:
                                print(f"Error predicting {target_name}: {e}")
                                # Use fallback recommendations
                                if target_name == 'meal':
                                    predictions[target_name] = get_fallback_meal_recommendation(data)
                                elif target_name == 'ingredients':
                                    predictions[target_name] = get_fallback_ingredients(predictions.get('meal', ''))
                                else:
                                    predictions[target_name] = 'Balanced'
                        else:
                            # Use fallback recommendations
                            if target_name == 'meal':
                                predictions[target_name] = get_fallback_meal_recommendation(data)
                            elif target_name == 'ingredients':
                                predictions[target_name] = get_fallback_ingredients(predictions.get('meal', ''))
                            else:
                                predictions[target_name] = 'Balanced'

                    # Nutrition prediction
                    try:
                        if nutrition_model is not None:
                            # Use nutrition features (exclude meal_type and dietary_habits)
                            nutrition_input = dict(enriched_input)
                            nutrition_input.pop('meal_type', None)
                            nutrition_input.pop('dietary_habits', None)
                            
                            # Compat: ensure both 'budget' and 'price' keys exist
                            if 'budget' in nutrition_input and 'price' not in nutrition_input:
                                nutrition_input['price'] = nutrition_input['budget']
                            
                            nutrition_df = pd.DataFrame([nutrition_input])
                            
                            # Apply preprocessor if available
                            if 'nutrition_preprocessor' in globals() and nutrition_preprocessor is not None:
                                X_nut = nutrition_preprocessor.transform(nutrition_df)
                                nutri_vals = nutrition_model.predict(X_nut)
                            else:
                                nutri_vals = nutrition_model.predict(nutrition_df)
                            
                            if hasattr(nutri_vals, 'tolist'):
                                nutri_vals = nutri_vals.tolist()
                            vals = nutri_vals[0] if isinstance(nutri_vals, list) else nutri_vals
                            cals, prot, carbs, fat = [float(x) for x in vals]
                            predictions['nutrition'] = {
                                'calories': round(cals, 1),
                                'protein_g': round(prot, 1),
                                'carbs_g': round(carbs, 1),
                                'fat_g': round(fat, 1)
                            }
                        else:
                            # Use fallback nutrition
                            predictions['nutrition'] = get_fallback_nutrition(data)
                        
                        # If meals.csv has nutrition for the selected meal, override predictions
                        meal_name = str(predictions.get('meal', '')).strip()
                        if meal_name and 'meal' in meals_df.columns:
                            matched = meals_df[meals_df['meal'].str.strip().str.lower() == meal_name.lower()]
                            if not matched.empty:
                                row = matched.iloc[0]
                                if {'calories','protein_g','carbs_g','fat_g'}.issubset(matched.columns):
                                    predictions['nutrition'] = {
                                        'calories': float(row.get('calories')) if pd.notna(row.get('calories')) else predictions['nutrition']['calories'],
                                        'protein_g': float(row.get('protein_g')) if pd.notna(row.get('protein_g')) else predictions['nutrition']['protein_g'],
                                        'carbs_g': float(row.get('carbs_g')) if pd.notna(row.get('carbs_g')) else predictions['nutrition']['carbs_g'],
                                        'fat_g': float(row.get('fat_g')) if pd.notna(row.get('fat_g')) else predictions['nutrition']['fat_g']
                                    }
                        
                    except Exception as e:
                        print(f"Error predicting nutrition: {e}")
                        # Use fallback nutrition
                        predictions['nutrition'] = get_fallback_nutrition(data)
                    
                else:
                    # Legacy path (fallback to old models)
                    print("Using legacy models for prediction...")
                    try:
                        clf_pipeline = joblib.load('diet_recommendation_model.pkl')
                        reg1_pipeline = joblib.load('exercise_hours_model.pkl')
                        diet_label_encoder = joblib.load('diet_label_encoder.pkl')
                        
                        # Diet recommendation
                        pred_class_encoded = clf_pipeline.predict(input_df)
                        try:
                            pred_diet = diet_label_encoder.inverse_transform(pred_class_encoded)[0]
                        except Exception:
                            pred_diet = str(pred_class_encoded[0])
                        predictions['diet_recommendation'] = pred_diet

                        # Weekly exercise hours
                        weekly_exercise = float(reg1_pipeline.predict(input_df)[0])
                        predictions['weekly_exercise_hours'] = round(weekly_exercise, 2)

                        # Use fallback for meal and ingredients
                        predictions['meal'] = get_fallback_meal_recommendation(data)
                        predictions['ingredients'] = get_fallback_ingredients(predictions['meal'])
                        
                        # Use fallback nutrition
                        predictions['nutrition'] = get_fallback_nutrition(data)
                        
                    except Exception as e:
                        print(f"Error with legacy models: {e}")
                        # Complete fallback
                        predictions = {
                            'diet_recommendation': 'Balanced',
                            'weekly_exercise_hours': 3.0,
                            'meal': get_fallback_meal_recommendation(data),
                            'ingredients': get_fallback_ingredients(get_fallback_meal_recommendation(data)),
                            'nutrition': get_fallback_nutrition(data)
                        }

                # Estimate meal cost and budget usage
                estimated_cost = None
                budget_usage_percent = None
                within_budget = None
                try:
                    meal_name = str(predictions.get('meal', ''))
                    if meal_name and not meals_df.empty:
                        matched = meals_df[meals_df['meal'] == meal_name]
                        if not matched.empty:
                            estimated_cost = float(matched['budget'].mean(skipna=True))
                    if estimated_cost is not None and original_budget > 0:
                        budget_usage_percent = round((estimated_cost / original_budget) * 100.0, 2)
                        within_budget = estimated_cost <= original_budget
                except Exception:
                    pass

                predictions['estimated_meal_cost'] = None if estimated_cost is None else round(estimated_cost, 2)
                predictions['budget_entered'] = original_budget
                predictions['budget_usage_percent'] = budget_usage_percent
                predictions['within_budget'] = within_budget

                # Save the predictions to the database
                conn = get_db_connection()
                if conn:
                    cursor = conn.cursor()
                    try:
                        cursor.execute("""
                            INSERT INTO meal_plans 
                            (user_id, diet_recommendation, weekly_exercise_hours, meal_type, recommended_meal, ingredients)
                            VALUES (%s, %s, %s, %s, %s, %s)
                        """, (
                            session['user_id'],
                            predictions.get('diet_recommendation', ''),
                            predictions.get('weekly_exercise_hours', 0.0),
                            data['meal_type'],
                            predictions.get('meal', ''),
                            predictions.get('ingredients', '')
                        ))
                        conn.commit()
                    except mysql.connector.Error as err:
                        print(f"Error saving meal plan: {err}")
                    finally:
                        cursor.close()
                        conn.close()

                # Save health progress if not exists for today
                save_health_progress(data)

                return jsonify({
                    'success': True,
                    'predictions': predictions
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Error in prediction: {str(e)}'
                })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error in input processing: {str(e)}'
        })

def save_health_progress(data):
    try:
        conn = get_db_connection()
        if not conn:
            return
            
        cursor = conn.cursor()
        
        # Calculate BMI
        bmi = data['weight'] / ((data['height']/100) ** 2)
        
        try:
            # Check if entry exists for today
            cursor.execute("""
                SELECT id FROM health_progress 
                WHERE user_id = %s 
                AND DATE(measurement_date) = CURDATE()
            """, (session['user_id'],))
            
            if not cursor.fetchone():
                # Insert new record
                cursor.execute("""
                    INSERT INTO health_progress 
                    (user_id, weight, height, bmi, cholesterol, blood_pressure, glucose, physical_activity_level)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """, (
                    session['user_id'],
                    data['weight'],
                    data['height'],
                    bmi,
                    data['cholesterol'],
                    data['blood_pressure'],
                    data['glucose'],
                    data['physical_activity_level']
                ))
                conn.commit()
                
        except mysql.connector.Error as err:
            print(f"Error saving health progress: {err}")
            
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        print(f"Error in save_health_progress: {e}")

def get_fallback_meal_recommendation(data):
    """Provide fallback meal recommendations based on dietary preferences"""
    meal_type = data.get('meal_type', 'Dinner')
    dietary_habits = data.get('dietary_habits', 'Non-Vegetarian')
    dietary_restrictions = data.get('dietary_restrictions', 'None')
    
    # Simple fallback meal recommendations
    meals = {
        'Breakfast': {
            'Vegetarian': 'Oatmeal with fruits and nuts',
            'Non-Vegetarian': 'Scrambled eggs with whole grain toast',
            'Vegan': 'Avocado toast with tomato'
        },
        'Lunch': {
            'Vegetarian': 'Quinoa salad with vegetables',
            'Non-Vegetarian': 'Grilled chicken with brown rice',
            'Vegan': 'Lentil curry with brown rice'
        },
        'Dinner': {
            'Vegetarian': 'Vegetable stir-fry with tofu',
            'Non-Vegetarian': 'Baked salmon with sweet potato',
            'Vegan': 'Chickpea curry with quinoa'
        },
        'Snack': {
            'Vegetarian': 'Greek yogurt with berries',
            'Non-Vegetarian': 'Hard-boiled eggs with nuts',
            'Vegan': 'Hummus with vegetable sticks'
        }
    }
    
    meal = meals.get(meal_type, meals['Dinner']).get(dietary_habits, 'Balanced meal')
    
    # Adjust for dietary restrictions
    if 'Low_Sodium' in dietary_restrictions:
        meal += ' (low sodium)'
    elif 'Low_Sugar' in dietary_restrictions:
        meal += ' (low sugar)'
        
    return meal

def get_fallback_ingredients(meal):
    """Provide fallback ingredients based on meal recommendation"""
    ingredients_map = {
        'Oatmeal': 'Oats, banana, almonds, honey, milk',
        'Scrambled eggs': 'Eggs, whole grain bread, butter, salt, pepper',
        'Avocado toast': 'Avocado, whole grain bread, tomato, lemon, salt',
        'Quinoa salad': 'Quinoa, cucumber, tomato, olive oil, lemon',
        'Grilled chicken': 'Chicken breast, brown rice, vegetables, olive oil',
        'Lentil curry': 'Red lentils, onion, garlic, ginger, spices, brown rice',
        'Vegetable stir-fry': 'Mixed vegetables, tofu, soy sauce, garlic, ginger',
        'Baked salmon': 'Salmon fillet, sweet potato, olive oil, herbs',
        'Chickpea curry': 'Chickpeas, onion, tomato, spices, quinoa',
        'Greek yogurt': 'Greek yogurt, mixed berries, honey',
        'Hard-boiled eggs': 'Eggs, mixed nuts, salt',
        'Hummus': 'Chickpeas, tahini, olive oil, lemon, vegetables'
    }
    
    for key, ingredients in ingredients_map.items():
        if key.lower() in meal.lower():
            return ingredients
    
    return 'Fresh vegetables, whole grains, lean protein, healthy fats'

def get_fallback_nutrition(data):
    """Provide fallback nutrition values based on user profile"""
    age = data.get('age', 30)
    weight = data.get('weight', 70)
    physical_activity = data.get('physical_activity_level', 'Moderate')
    
    # Base calorie calculation (simplified)
    base_calories = 2000 if age < 50 else 1800
    
    # Adjust for activity level
    activity_multiplier = {
        'Sedentary': 1.0,
        'Moderate': 1.2,
        'Active': 1.4
    }.get(physical_activity, 1.2)
    
    calories = int(base_calories * activity_multiplier)
    
    # Simple macronutrient distribution
    protein_g = int(calories * 0.25 / 4)  # 25% of calories from protein
    carbs_g = int(calories * 0.45 / 4)    # 45% of calories from carbs
    fat_g = int(calories * 0.30 / 9)      # 30% of calories from fat
    
    return {
        'calories': calories,
        'protein_g': protein_g,
        'carbs_g': carbs_g,
        'fat_g': fat_g
    }

if __name__ == '__main__':
    print("\nTesting database connection...")
    test_db_connection()
    print("\nStarting Flask application...")
    app.run(debug=True, port=5001) 