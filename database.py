import sqlite3
from datetime import datetime
import logging

logger = logging.getLogger("rag-agent")

class TicketDatabase:
    def __init__(self, db_path="tickets.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialize the database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
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
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

    def create_ticket(self, user_name: str, email: str, query: str) -> int:
        """Create a new ticket and return its ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO tickets (user_name, email, query)
                    VALUES (?, ?, ?)
                ''', (user_name, email, query))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Error creating ticket: {e}")
            raise

    def get_ticket(self, ticket_id: int) -> dict:
        """Get ticket details by ID"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT * FROM tickets WHERE id = ?', (ticket_id,))
                ticket = cursor.fetchone()
                if ticket:
                    return {
                        'id': ticket[0],
                        'user_name': ticket[1],
                        'email': ticket[2],
                        'query': ticket[3],
                        'status': ticket[4],
                        'created_at': ticket[5],
                        'resolved_at': ticket[6]
                    }
                return None
        except Exception as e:
            logger.error(f"Error getting ticket: {e}")
            raise

    def update_ticket_status(self, ticket_id: int, status: str):
        """Update ticket status"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                resolved_at = datetime.now() if status == 'resolved' else None
                cursor.execute('''
                    UPDATE tickets 
                    SET status = ?, resolved_at = ?
                    WHERE id = ?
                ''', (status, resolved_at, ticket_id))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating ticket status: {e}")
            raise