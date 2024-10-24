from app import app, db, Issue
import logging
from sqlalchemy import inspect, text

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
                logging.info("Database tables already exist. Checking for updates...")
                update_issue_table()
    except Exception as e:
        logging.error(f"An error occurred while initializing the database: {str(e)}")
        raise

def update_issue_table():
    try:
        with app.app_context():
            inspector = inspect(db.engine)
            columns = [column['name'] for column in inspector.get_columns('issue')]
            
            if 'start_timestamp' not in columns:
                logging.info("Adding 'start_timestamp' column to 'issue' table...")
                db.engine.execute(text("ALTER TABLE issue ADD COLUMN start_timestamp FLOAT"))
            
            if 'end_timestamp' not in columns:
                logging.info("Adding 'end_timestamp' column to 'issue' table...")
                db.engine.execute(text("ALTER TABLE issue ADD COLUMN end_timestamp FLOAT"))
            
            if 'image_paths' not in columns:
                logging.info("Adding 'image_paths' column to 'issue' table...")
                db.engine.execute(text("ALTER TABLE issue ADD COLUMN image_paths TEXT"))
            
            if 'timestamp' in columns:
                logging.info("Migrating data from 'timestamp' to 'start_timestamp'...")
                db.engine.execute(text("UPDATE issue SET start_timestamp = timestamp"))
                logging.info("Dropping 'timestamp' column from 'issue' table...")
                db.engine.execute(text("ALTER TABLE issue DROP COLUMN timestamp"))
            
            db.session.commit()
            logging.info("Issue table updated successfully.")
    except Exception as e:
        logging.error(f"An error occurred while updating the issue table: {str(e)}")
        db.session.rollback()
        raise

if __name__ == '__main__':
    init_db()
