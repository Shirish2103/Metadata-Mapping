import os
import sys
import time

# Ensure project root is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure UTF-8 stdout encoding for Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager
from pipeline.processor import MetadataProcessor

def preindex_all_dataset_movies(limit: int = None):
    """
    Pre-computes and caches metadata for all available movies in the dataset archive into SQLite DB.
    Allows 0-delay instant retrieval on the web frontend.
    """
    print("=" * 70)
    print("🎬 STARTING DATASET PRE-INDEXING & CACHE WARMUP")
    print("=" * 70)

    loader = DatasetLoader()
    db = DatabaseManager()
    processor = MetadataProcessor()

    movies = loader.get_available_movies()
    if not movies:
        print("❌ No movies found in dataset archive!")
        return

    if limit:
        movies = movies[:limit]

    print(f"📦 Total dataset movies to check/index: {len(movies)}")
    
    indexed_count = 0
    skipped_count = 0
    failed_count = 0

    start_time_all = time.time()

    for idx, m in enumerate(movies, start=1):
        title = m['title']
        imdb_id = m['imdb_id']
        year = m.get('year', 'N/A')

        print(f"\n[{idx}/{len(movies)}] Processing: '{title}' ({imdb_id}) [{year}]...")

        # 1. Check if already stored in SQLite Database
        existing = db.get_metadata(imdb_id_or_title=imdb_id)
        if not existing:
            existing = db.get_metadata(imdb_id_or_title=title)

        if existing:
            print(f"  ⚡ Already cached in SQLite database! Skipping...")
            skipped_count += 1
            continue

        # 2. Extract metadata and auto-save to SQLite
        try:
            t0 = time.time()
            result = processor.process_transcript(imdb_id_or_title=imdb_id)
            elapsed = time.time() - t0
            print(f"  ✅ Extracted & Saved to SQLite DB in {elapsed:.2f}s!")
            indexed_count += 1
        except Exception as e:
            print(f"  ❌ Failed to extract metadata for '{title}': {str(e)}")
            failed_count += 1

    total_time = time.time() - start_time_all
    print("\n" + "=" * 70)
    print("🎉 PRE-INDEXING COMPLETED!")
    print(f"  - Newly Indexed & Saved: {indexed_count}")
    print(f"  - Already Cached (Skipped): {skipped_count}")
    print(f"  - Failed/Errors: {failed_count}")
    print(f"  - Total Elapsed Time: {total_time:.2f} seconds")
    print("=" * 70)

if __name__ == "__main__":
    preindex_all_dataset_movies()
