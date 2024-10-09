from app import app, db
import logging
from sqlalchemy import inspect

logging.basicConfig(level=logging.INFO)

def init_db():
    try:
        with app.app_context():
            # Check if the database already exists
            inspector = inspect(db.engine)
            if not inspector.has_table('project'):
                logging.info("Creating database tables...")
                db.create_all()
                logging.info("Database tables created successfully.")
            else:
                logging.info("Database tables already exist. Skipping creation.")
    except Exception as e:
        logging.error(f"An error occurred while initializing the database: {str(e)}")
        raise

if __name__ == '__main__':
    init_db()
