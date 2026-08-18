import argparse
import json
import sys
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline.processor import MetadataProcessor
from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager

def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Transcript Metadata Extraction Platform"
    )
    parser.add_argument("--list", action="store_true", help="List available movie transcripts")
    parser.add_argument("--process", type=str, help="Process a movie transcript by Title or IMDB ID")
    parser.add_argument("--process-srt", type=str, help="Process an external SRT transcript file")
    parser.add_argument("--start", type=str, help="Start time for time-window extraction (e.g. 00:10:00 or 10:00)")
    parser.add_argument("--end", type=str, help="End time for time-window extraction (e.g. 00:25:00 or 25:00)")
    parser.add_argument("--query", type=str, help="Query extracted metadata from database by Title or IMDB ID")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    processor = MetadataProcessor(config_path=args.config)

    if args.list:
        loader = DatasetLoader()
        movies = loader.get_available_movies()
        print(f"\n=======================================================")
        print(f" Available Movie Transcripts ({len(movies)} total)")
        print(f"=======================================================")
        for idx, m in enumerate(movies, 1):
            print(f" [{idx:02d}] IMDB ID: {m['imdb_id']} | Title: {m['title']} ({m['year']}) | Genres: {m['genres']}")
        print(f"... and {len(movies) - 25} more scripts available.\n")
        return

    if args.process:
        tw_info = f" (Time Window: {args.start} to {args.end})" if args.start or args.end else ""
        print(f"\n[AI Pipeline] Processing script for: '{args.process}'{tw_info}...")
        try:
            result = processor.process_transcript(args.process, start_time=args.start, end_time=args.end)
            print("\n=======================================================")
            print(f" Extracted Metadata Result: {result.title} ({result.imdb_id})")
            print("=======================================================")
            print(json.dumps(result.model_dump(), indent=2))
            print("\n[DB] Successfully stored metadata in SQLite (data/transcript_metadata.db)!\n")
        except Exception as e:
            print(f"\n[Error] Processing failed: {e}\n")
            sys.exit(1)
        return

    if args.process_srt:
        tw_info = f" (Time Window: {args.start} to {args.end})" if args.start or args.end else ""
        print(f"\n[AI Pipeline] Processing SRT transcript file: '{args.process_srt}'{tw_info}...")
        try:
            result = processor.process_srt_file(args.process_srt, start_time=args.start, end_time=args.end)
            print("\n=======================================================")
            print(f" Extracted Metadata Result: {result.title}")
            print("=======================================================")
            print(json.dumps(result.model_dump(), indent=2))
            print("\n[DB] Successfully stored SRT metadata in SQLite!\n")
        except Exception as e:
            print(f"\n[Error] SRT Processing failed: {e}\n")
            sys.exit(1)
        return

    if args.query:
        db = DatabaseManager()
        record = db.get_metadata(args.query, start_time=args.start, end_time=args.end)
        tw_info = f" [{args.start} - {args.end}]" if args.start or args.end else ""
        if record:
            print("\n=======================================================")
            print(f" Retrieved Stored Metadata for Query: '{args.query}'{tw_info}")
            print("=======================================================")
            print(json.dumps(record, indent=2))
            print()
        else:
            print(f"\n[DB] No extracted metadata found for query: '{args.query}'{tw_info}. Run --process or --process-srt first!\n")
        return

    parser.print_help()

if __name__ == "__main__":
    main()
