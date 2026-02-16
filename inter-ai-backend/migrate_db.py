from app import app, db
from sqlalchemy import text

with app.app_context():
    try:
        # Check if column exists first to avoid error? 
        # PostgreSQL doesn't support IF NOT EXISTS in ADD COLUMN easily in all versions, 
        # but we can try catch.
        print("Attempting to add 'title' column to 'practice_history'...")
        db.session.execute(text("ALTER TABLE practice_history ADD COLUMN title VARCHAR(255);"))
        db.session.commit()
        print("Migration successful: Added title column.")
    except Exception as e:
        print(f"Migration output: {e}")
