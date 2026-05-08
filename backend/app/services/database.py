import sqlite3
import logging

logger = logging.getLogger(__name__)

def init_db():
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS user_sessions (
            uid TEXT PRIMARY KEY,
            name TEXT,
            login_stamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_session(uid: str, username: str, reqdatetime: str):
    try:
        conn = sqlite3.connect('users.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO user_sessions (uid, name, login_stamp) VALUES (?, ?, ?)", (uid, username, reqdatetime))
        conn.commit()
        conn.close()
        logger.info(f"Successfully saved {username} to SQLite users.db")
        return True
    except Exception as e:
        logger.error(f"Failed to save to SQLite: {e}")
        return False
