import argparse
import json
import sys
import warnings
warnings.filterwarnings("ignore")

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline.processor import MetadataProcessor
from data.dataset_loader import DatasetLoader
from data.db_manager import DatabaseManager

def print_character_presence_table(result_dict: dict):
    presence_data = result_dict.get("character_presence", {}) or {}
    characters = presence_data.get("characters", [])
    total_scenes = presence_data.get("total_movie_scenes", 0)
    tot_dur = result_dict.get("total_duration") or result_dict.get("time_range", {}).get("total_duration", "N/A")

    if not characters:
        return

    print("\n" + "=" * 105)
    print(f" ⏱️ CHARACTER SCENE PRESENCE & SCREEN TIME PACING (Total Scenes: {total_scenes} | Movie Duration: {tot_dur})")
    print("=" * 105)
    print(f" {'Character':<20} | {'Role':<10} | {'Screen Time %':<13} | {'Scenes':<8} | {'Lines':<6} | {'Entry [Scene (TS)]':<18} | {'Exit [Scene (TS)]':<18}")
    print("-" * 105)

    for char in characters[:15]:
        name = char.get("character_name", "Unknown")[:19]
        role = char.get("role_type", "Minor")
        st_pct = f"{char.get('screen_time_percentage', 0.0):.1f}%"
        sc_cnt = str(char.get("scene_count", 0))
        lines = str(char.get("dialogue_line_count", 0))
        entry = f"Scene {char.get('first_scene_idx')} ({char.get('first_timestamp', '00:00:00')})"
        exit_p = f"Scene {char.get('last_scene_idx')} ({char.get('last_timestamp', '00:00:00')})"

        print(f" {name:<20} | {role:<10} | {st_pct:<13} | {sc_cnt:<8} | {lines:<6} | {entry:<18} | {exit_p:<18}")

    if len(characters) > 15:
        print(f" ... (+ {len(characters) - 15} more characters identified)")
    print("=" * 105 + "\n")

def print_pretty_dialogue_breakdown(result_dict: dict):
    dialogues = result_dict.get("dialogues_in_window", [])
    time_range = result_dict.get("time_range", {}) or {}
    s_time = time_range.get("start_time", "00:00:00")
    e_time = time_range.get("end_time", "00:00:00")
    tot_dur = result_dict.get("total_duration") or time_range.get("total_duration") or "N/A"
    title = result_dict.get("title", "Movie Transcript")
    speakers = result_dict.get("speaker_list", [])

    print(f"\n" + "=" * 80)
    print(f" TIMELINE DIALOGUE BREAKDOWN [{s_time} - {e_time}]")
    print(f" Title                  : {title}")
    print(f" ⏱️ Total Movie Duration : {tot_dur}")
    print("=" * 80)

    if not dialogues:
        print("  (No dialogues found in this timestamp range)")
    else:
        current_loc = None
        for item in dialogues[:35]:
            loc = item.get("location") or "SCENE"
            ts = item.get("timestamp", "00:00:00")
            spk = item.get("speaker", "Unknown")
            text = item.get("text", "")
            
            if loc != current_loc:
                current_loc = loc
                print(f"\n [LOCATION] {current_loc}")

            print(f"   [{ts}] {spk:<16} : \"{text}\"")

        if len(dialogues) > 35:
            print(f"\n   ... (+ {len(dialogues) - 35} more dialogues in this timestamp duration)")

    print("-" * 80)
    print(f" Active Speakers ({len(speakers)}): {', '.join(speakers[:15])}")
    print("=" * 80 + "\n")

    print_character_presence_table(result_dict)


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Transcript Metadata Extraction Platform"
    )
    parser.add_argument("command_or_title", nargs="*", help="Process a movie by title e.g. analyze 'Full Metal Jacket' or 'Full Metal Jacket'")
    parser.add_argument("--list", action="store_true", help="List available movie transcripts")
    parser.add_argument("--process", "--movie", "-m", type=str, help="Process a movie transcript by Title or IMDB ID")
    parser.add_argument("--process-srt", "--srt", "-s", type=str, help="Process an external SRT transcript file")
    parser.add_argument("--start", type=str, help="Start time for time-window extraction (e.g. 00:10:00 or 10:00)")
    parser.add_argument("--end", type=str, help="End time for time-window extraction (e.g. 00:25:00 or 25:00)")
    parser.add_argument("--query", type=str, help="Query extracted metadata from database by Title or IMDB ID")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config.yaml")

    args = parser.parse_args()

    # Determine target movie title if positional arguments are passed e.g. analyze "Full Metal Jacket"
    target_movie = args.process
    if not target_movie and args.command_or_title:
        pos_args = args.command_or_title
        if pos_args[0].lower() in {"analyze", "process"} and len(pos_args) > 1:
            target_movie = " ".join(pos_args[1:])
        else:
            target_movie = " ".join(pos_args)

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

    if target_movie:
        tw_info = f" (Time Window: {args.start} to {args.end})" if args.start or args.end else ""
        print(f"\n[AI Pipeline] Processing script for: '{target_movie}'{tw_info}...")
        try:
            result = processor.process_transcript(target_movie, start_time=args.start, end_time=args.end)

            res_dict = result.model_dump()
            print_pretty_dialogue_breakdown(res_dict)
            print("=======================================================")
            print(f" Extracted Metadata Result: {result.title} ({result.imdb_id})")
            print(f" ⏱️ Total Movie Duration: {result.total_duration}")
            print("=======================================================")
            print(json.dumps(res_dict, indent=2))
            print("\n" + "=" * 60)
            print(f" ⏱️ TOTAL MOVIE DURATION  : {result.total_duration}")
            print(f" 🎯 TIME WINDOW PROCESSED : [{args.start or '00:00:00'} - {args.end or result.total_duration}]")
            print("=" * 60)
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
            res_dict = result.model_dump()
            print_pretty_dialogue_breakdown(res_dict)
            print("=======================================================")
            print(f" Extracted Metadata Result: {result.title}")
            print(f" ⏱️ Total Movie Duration: {result.total_duration}")
            print("=======================================================")
            print(json.dumps(res_dict, indent=2))
            print("\n" + "=" * 60)
            print(f" ⏱️ TOTAL MOVIE DURATION  : {result.total_duration}")
            print(f" 🎯 TIME WINDOW PROCESSED : [{args.start or '00:00:00'} - {args.end or result.total_duration}]")
            print("=" * 60)
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
            print_pretty_dialogue_breakdown(record)
            print("=======================================================")
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
