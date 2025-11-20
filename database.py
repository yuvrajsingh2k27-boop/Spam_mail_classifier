import sqlite3

# Connect to the database
def connect_db():
    conn = sqlite3.connect("users.db")
    cur = conn.cursor()
    return conn, cur

# Close the connection
def close_conn(conn, cur):
    cur.close()
    conn.close()

# Create users table if it doesn't exist
def initialize_db():
    conn, cur = connect_db()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            name TEXT NOT NULL,
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    close_conn(conn, cur)

# Sign up a new user
def signup(name, username, password):
    conn, cur = connect_db()
    try:
        cur.execute("INSERT INTO users (name, username, password) VALUES (?, ?, ?)", (name, username, password))
        conn.commit()
        print("Signup successful!")
    except sqlite3.IntegrityError:
        print("Username already exists.")
    close_conn(conn, cur)

# Log in an existing user
def log_in(username, password):
    conn, cur = connect_db()
    cur.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, password))
    user = cur.fetchone()
    close_conn(conn, cur)
    if user:
        print(f"Welcome back, {user[0]}!")
    else:
        print("Invalid username or password.")

# Initialize the database
initialize_db()

# Example usage
# signup("Ankita_Dhiman", "ankita", "pass123")
# log_in("Ankita_Dhiman", "pass123")