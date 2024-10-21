from app import app, db, init_db
import logging

logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    try:
        init_db()
        logging.info("Database initialized successfully.")
    except Exception as e:
        logging.error(f"An error occurred while initializing the database: {str(e)}")
        raise
