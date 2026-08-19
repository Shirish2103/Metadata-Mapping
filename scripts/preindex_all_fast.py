import os
import sys
import time
import concurrent.futures

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager
from pipeline.processor import MetadataProcessor

def preindex_full_database():
    db_path = "data/transcript_metadata.db"
    
    print("=" * 70)
    db = DatabaseManager(db_path)
    existing_records = db.get_all_metadata()
    existing_ids = {r.get("imdb_id", "").lower() for r in existing_records if r.get("imdb_id")}
    existing_titles = {r.get("title", "").lower() for r in existing_records if r.get("title")}
    print(f"✅ Found {len(existing_records)} existing valid records in database. Resuming...")

    loader = DatasetLoader()
    processor = MetadataProcessor()

    all_movies = loader.get_available_movies()
    movies = [m for m in all_movies if m.get("imdb_id", "").lower() not in existing_ids and m.get("title", "").lower() not in existing_titles]
    print(f"📦 Found {len(movies)} remaining movies to index out of {len(all_movies)} total.")

    start_time = time.time()
    success_count = 0
    fail_count = 0

    def process_single_movie(movie_data):
        imdb_id = movie_data.get("imdb_id")
        title = movie_data.get("title")
        try:
            res = processor.process_transcript(imdb_id)
            if res.character_presence and len(res.character_presence.characters) > 0 and res.scene_breakdown_count > 4:
                db.save_extracted_metadata(res)
                return True, title, imdb_id, len(res.character_presence.characters), res.character_presence.total_movie_scenes
        except Exception:
            pass
        return False, title, imdb_id, 0, 0

    total = len(movies)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        futures = {executor.submit(process_single_movie, m): m for m in movies}
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            success, title, imdb_id, num_chars, num_scenes = future.result()
            if success:
                success_count += 1
                if success_count % 50 == 0 or success_count <= 10:
                    print(f"[{completed}/{total}] Saved: {title} ({imdb_id}) -> {num_scenes} scenes, {num_chars} chars")
            else:
                fail_count += 1

    elapsed = time.time() - start_time
    total_stored = len(db.get_all_metadata())

    print("\n" + "=" * 70)
    print("🎉 FULL PRE-INDEXING COMPLETE!")
    print(f"  - Total Processed: {total}")
    print(f"  - Successfully Extracted & Saved to DB: {success_count}")
    print(f"  - Skipped/Failed: {fail_count}")
    print(f"  - Total Verified Records in SQLite DB: {total_stored}")
    print(f"  - Elapsed Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("=" * 70)

if __name__ == "__main__":
    preindex_full_database()
