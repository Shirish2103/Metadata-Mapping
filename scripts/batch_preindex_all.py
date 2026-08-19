import os
import sys
import time
import json
from typing import List, Dict, Any

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager
from pipeline.processor import MetadataProcessor

def run_full_dataset_preindexing():
    db_path = "data/transcript_metadata.db"
    
    print("=" * 75)
    print("🗑️ STEP 1: CLEARING OLD DATABASE & CREATING FRESH SQLITE DB")
    print("=" * 75)
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print("✅ Old SQLite database removed.")

    db = DatabaseManager(db_path)
    print("✅ Fresh SQLite database initialized.")

    loader = DatasetLoader()
    processor = MetadataProcessor()

    movies = loader.get_available_movies()
    print(f"📦 Total Movies in Index: {len(movies)}")

    print("\n" + "=" * 75)
    print("🚀 STEP 2: PRE-INDEXING ALL 2,600+ MOVIES INTO SQLITE DATABASE")
    print("=" * 75)

    start_time = time.time()
    success_count = 0
    fail_count = 0

    for idx, m in enumerate(movies, start=1):
        imdb_id = m.get("imdb_id")
        title = m.get("title")

        try:
            res = processor.process_transcript(imdb_id)
            db.save_extracted_metadata(res)
            success_count += 1
            if success_count % 50 == 0 or success_count <= 5:
                print(f"[{idx}/{len(movies)}] Saved: {title} ({imdb_id}) -> {res.character_presence.total_movie_scenes} scenes, {len(res.character_presence.characters)} characters")
        except Exception as e:
            fail_count += 1

    elapsed = time.time() - start_time
    total_stored = len(db.get_all_metadata())

    print("\n" + "=" * 75)
    print("🎉 FULL PRE-INDEXING COMPLETED SUCCESSFULLY!")
    print(f"  - Total Dataset Movies: {len(movies)}")
    print(f"  - Successfully Extracted & Saved to DB: {success_count}")
    print(f"  - Failed/Skipped: {fail_count}")
    print(f"  - Verified SQLite Database Records: {total_stored}")
    print(f"  - Total Execution Time: {elapsed:.2f} seconds ({elapsed/60:.2f} mins)")
    print("=" * 75)

if __name__ == "__main__":
    run_full_dataset_preindexing()
