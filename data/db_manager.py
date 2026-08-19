import sqlite3
import json
import os
from typing import List, Dict, Any, Optional
from models.schema import MovieMetadata, SceneSegment, ScriptMetadataResult

class DatabaseManager:
    def __init__(self, db_path: str = "data/transcript_metadata.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path, timeout=60.0)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS movies (
                    imdb_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    year TEXT,
                    genres TEXT,
                    directors TEXT,
                    writers TEXT,
                    plot TEXT
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS scenes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    imdb_id TEXT NOT NULL,
                    scene_idx INTEGER NOT NULL,
                    location TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    is_estimated BOOLEAN,
                    raw_text TEXT,
                    FOREIGN KEY (imdb_id) REFERENCES movies(imdb_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS dialogues (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    imdb_id TEXT NOT NULL,
                    scene_idx INTEGER NOT NULL,
                    speaker TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    text TEXT NOT NULL,
                    FOREIGN KEY (imdb_id) REFERENCES movies(imdb_id)
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS extracted_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    imdb_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    primary_category TEXT,
                    sentiment TEXT,
                    emotions TEXT,
                    main_topics TEXT,
                    keywords TEXT,
                    people_entities TEXT,
                    location_entities TEXT,
                    org_entities TEXT,
                    full_json TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (imdb_id) REFERENCES movies(imdb_id)
                )
            """)
            conn.commit()

    def save_movie(self, meta: MovieMetadata):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO movies (imdb_id, title, year, genres, directors, writers, plot)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                meta.imdb_id,
                meta.title,
                meta.year,
                ", ".join(meta.genres),
                ", ".join(meta.directors),
                ", ".join(meta.writers),
                meta.plot
            ))
            conn.commit()

    def save_scenes(self, imdb_id: str, scenes: List[SceneSegment]):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM scenes WHERE imdb_id = ?", (imdb_id,))
            cursor.execute("DELETE FROM dialogues WHERE imdb_id = ?", (imdb_id,))

            for sc in scenes:
                start_ts = sc.time_range.start_time if sc.time_range else "00:00:00"
                end_ts = sc.time_range.end_time if sc.time_range else "00:00:00"
                is_est = sc.time_range.is_estimated if sc.time_range else False

                cursor.execute("""
                    INSERT INTO scenes (imdb_id, scene_idx, location, start_time, end_time, is_estimated, raw_text)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (imdb_id, sc.scene_idx, sc.location, start_ts, end_ts, is_est, sc.action_text))

                for d in sc.dialogues:
                    d_start = d.time_range.start_time if d.time_range else start_ts
                    d_end = d.time_range.end_time if d.time_range else end_ts
                    cursor.execute("""
                        INSERT INTO dialogues (imdb_id, scene_idx, speaker, start_time, end_time, text)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (imdb_id, sc.scene_idx, d.speaker, d_start, d_end, d.text))

            conn.commit()

    def save_extracted_metadata(self, result: ScriptMetadataResult):
        if result.movie_info:
            self.save_movie(result.movie_info)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM extracted_metadata WHERE imdb_id = ?", (result.imdb_id,))
            
            full_json = result.model_dump_json()

            cursor.execute("""
                INSERT INTO extracted_metadata (
                    imdb_id, title, primary_category, sentiment, emotions,
                    main_topics, keywords, people_entities, location_entities, org_entities, full_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.imdb_id,
                result.title,
                result.category.primary_category,
                result.sentiment.sentiment,
                ", ".join(result.sentiment.emotions),
                ", ".join(result.topics.main_topics),
                ", ".join(result.topics.keywords),
                ", ".join(result.entities.people),
                ", ".join(result.entities.locations),
                ", ".join(result.entities.organizations),
                full_json
            ))
            conn.commit()

    def get_metadata(
        self, 
        imdb_id_or_title: str, 
        start_time: Optional[str] = None, 
        end_time: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        target = imdb_id_or_title.strip().lower()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            if start_time or end_time:
                s_ts = start_time if start_time else "00:00:00"
                e_ts = end_time if end_time else "99:59:59"
                cursor.execute("""
                    SELECT full_json FROM extracted_metadata 
                    WHERE (LOWER(imdb_id) = ? OR LOWER(title) = ?) 
                    ORDER BY id DESC
                """, (target, target))
                rows = cursor.fetchall()
                for row in rows:
                    rec = json.loads(row[0])
                    tr = rec.get("time_range", {}) or {}
                    if tr.get("start_time") == s_ts or tr.get("end_time") == e_ts:
                        return rec

            cursor.execute("""
                SELECT full_json FROM extracted_metadata 
                WHERE LOWER(imdb_id) = ? OR LOWER(title) = ?
                ORDER BY id DESC
            """, (target, target))
            rows = cursor.fetchall()
            for row in rows:
                rec = json.loads(row[0])
                tr = rec.get("time_range", {}) or {}
                tot_dur = rec.get("total_duration") or tr.get("total_duration")
                end_ts = tr.get("end_time")
                sc_cnt = rec.get("scene_breakdown_count", 0)
                # Skip 1-hour windowed or dummy records for full movie queries
                if end_ts and tot_dur and end_ts == "01:00:00" and end_ts != tot_dur and sc_cnt <= 4:
                    continue
                return rec
        return None

    def get_metadata_exact_window(
        self, 
        imdb_id_or_title: str, 
        start_time: Optional[str] = None, 
        end_time: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        target = imdb_id_or_title.strip().lower()
        if not start_time and not end_time:
            return self.get_metadata(target)
            
        s_ts = start_time if start_time else "00:00:00"
        e_ts = end_time if end_time else "99:59:59"
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT full_json FROM extracted_metadata 
                WHERE (LOWER(imdb_id) = ? OR LOWER(title) LIKE ?) 
                AND (title LIKE ? OR full_json LIKE ? OR full_json LIKE ?)
                ORDER BY id DESC
            """, (
                target, 
                f"%{target}%", 
                f"%[{s_ts}%", 
                f'%"start_time": "{s_ts}"%', 
                f'%"end_time": "{e_ts}"%'
            ))
            row = cursor.fetchone()
            if row:
                return json.loads(row[0])
        return None

    def get_all_metadata(self) -> List[Dict[str, Any]]:
        records = []
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT full_json FROM extracted_metadata ORDER BY id ASC")
            rows = cursor.fetchall()
            for row in rows:
                try:
                    records.append(json.loads(row[0]))
                except Exception:
                    pass
        return records
