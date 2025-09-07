import mysql.connector
from werkzeug.security import generate_password_hash

# Database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',  # Change to your MySQL username
    'password': '',  # Change to your MySQL password
    'database': 'meal_planner'
}

def setup_admin():
    # Admin credentials
    admin_user = {
        'username': 'admin',
        'email': 'admin@example.com',
        'password': 'admin123',  # This will be hashed
        'role': 'admin'
    }
    
    try:
        # Connect to database
        print("Connecting to database...")
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()
        
        # Delete existing admin if exists
        print("Removing old admin user if exists...")
        cursor.execute("DELETE FROM users WHERE username = 'admin'")
        
        # Generate password hash
        hashed_password = generate_password_hash(admin_user['password'])
        
        # Insert new admin user
        print("Creating new admin user...")
        cursor.execute("""
            INSERT INTO users (username, email, password, role)
            VALUES (%s, %s, %s, %s)
        """, (
            admin_user['username'],
            admin_user['email'],
            hashed_password,
            admin_user['role']
        ))
        
        # Commit changes
        conn.commit()
        
        print("\nAdmin user created successfully!")
        print("---------------------------")
        print("Username:", admin_user['username'])
        print("Password:", admin_user['password'])
        print("---------------------------")
        print("\nYou can now log in with these credentials.")
        
    except mysql.connector.Error as err:
        print("Error:", err)
    finally:
        if 'conn' in locals() and conn.is_connected():
            cursor.close()
            conn.close()

if __name__ == "__main__":
    setup_admin() 