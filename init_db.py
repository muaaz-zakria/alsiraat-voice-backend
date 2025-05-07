import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("db-init")

def init_database():
    try:
        conn = sqlite3.connect('tickets.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_name TEXT NOT NULL,
                email TEXT NOT NULL,
                query TEXT NOT NULL,
                status TEXT DEFAULT 'unresolved',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved_at TIMESTAMP
            )
        ''')
        
        conn.commit()
        logger.info("Database initialized successfully!")
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tickets'")
        if cursor.fetchone():
            logger.info("Tickets table exists and is ready to use!")
        else:
            logger.error("Failed to create tickets table!")
            
    except sqlite3.Error as e:
        logger.error(f"Database error: {e}")
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    init_database()