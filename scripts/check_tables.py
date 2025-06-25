#!/usr/bin/env python3
"""Quick script to check available database tables and their data."""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

def main():
    load_dotenv()
    
    # Create engine
    db_url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
    )
    engine = create_engine(db_url)
    
    print("🔍 Checking available tables...")
    
    with engine.connect() as conn:
        # List all tables with 'historical' in name
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_name LIKE '%historical%'
            AND table_schema = 'public'
            ORDER BY table_name
        """))
        
        tables = [row[0] for row in result.fetchall()]
        print(f"📊 Found tables: {tables}")
        
        # Check each table
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.fetchone()[0]
                
                if count > 0:
                    result = conn.execute(text(f"SELECT MIN(bar_date), MAX(bar_date) FROM {table}"))
                    min_date, max_date = result.fetchone()
                    print(f"  ✅ {table}: {count:,} records, {min_date} to {max_date}")
                else:
                    print(f"  ❌ {table}: empty")
                    
            except Exception as e:
                print(f"  ❌ {table}: error - {e}")

if __name__ == "__main__":
    main()
