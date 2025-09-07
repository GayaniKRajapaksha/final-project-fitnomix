-- Create the database
CREATE DATABASE IF NOT EXISTS meal_planner;
USE meal_planner;

-- Create users table
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    role ENUM('admin', 'customer') NOT NULL DEFAULT 'customer',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

-- Create meal plans table
CREATE TABLE IF NOT EXISTS meal_plans (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    diet_recommendation VARCHAR(255),
    weekly_exercise_hours FLOAT,
    meal_type VARCHAR(50),
    recommended_meal VARCHAR(255),
    ingredients TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
);

-- Create health progress table
CREATE TABLE IF NOT EXISTS health_progress (
    id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    weight FLOAT,
    height FLOAT,
    bmi FLOAT,
    cholesterol FLOAT,
    blood_pressure FLOAT,
    glucose FLOAT,
    physical_activity_level VARCHAR(50),
    measurement_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
); 