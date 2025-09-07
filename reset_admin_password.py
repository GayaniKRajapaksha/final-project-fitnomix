import mysql.connector
from werkzeug.security import generate_password_hash

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # Change to your MySQL username
    'password': '',  # Change to your MySQL password
    'database': 'meal_planner'
}

def reset_admin_password():
    # New admin password
    new_password = 'admin123'  # You can change this to any password you want
    
    try:
        # Connect to database
        print("Connecting to database...")
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Check if admin user exists
        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
        admin_user = cursor.fetchone()
        
        if admin_user:
            # Generate new password hash
            hashed_password = generate_password_hash(new_password)
            
            # Update admin password
            print("Updating admin password...")
            cursor.execute("""
                UPDATE users 
                SET password = %s 
                WHERE username = 'admin'
            """, (hashed_password,))
            
            # Commit changes
            conn.commit()
            
            print("\nAdmin password reset successfully!")
            print("---------------------------")
            print("Username: admin")
            print("New Password:", new_password)
            print("---------------------------")
            print("\nYou can now log in with these credentials.")
            
        else:
            print("Admin user not found. Creating new admin user...")
            # Create admin user if doesn't exist
            hashed_password = generate_password_hash(new_password)
            cursor.execute("""
                INSERT INTO users (username, email, password, role)
                VALUES (%s, %s, %s, %s)
            """, ('admin', 'admin@example.com', hashed_password, 'admin'))
            
            conn.commit()
            
            print("\nAdmin user created successfully!")
            print("---------------------------")
            print("Username: admin")
            print("Password:", new_password)
            print("---------------------------")
            print("\nYou can now log in with these credentials.")
        
    except mysql.connector.Error as err:
        print("Error:", err)
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    reset_admin_password() 