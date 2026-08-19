import re
from typing import List, Dict, Any, Optional
from models.schema import TimeRange, SceneSegment, Dialogue

def format_seconds_to_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs:02d}:{mins:02d}:{secs:02d}"

def parse_srt_timestamp(ts_str: str) -> float:
    ts_str = ts_str.replace(',', '.').strip()
    parts = ts_str.split(':')
    if len(parts) == 3:
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s
    elif len(parts) == 2:
        m, s = float(parts[0]), float(parts[1])
        return m * 60 + s
    return 0.0

def timestamp_to_seconds(ts_str: str) -> float:
    if not ts_str:
        return 0.0
    ts_str = str(ts_str).strip()
    try:
        return float(ts_str)
    except ValueError:
        pass
    parts = ts_str.replace(',', '.').split(':')
    try:
        if len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 1:
            return float(parts[0])
    except ValueError:
        return 0.0
    return 0.0

class TimestampParser:
    @staticmethod
    def estimate_screenplay_timestamps(
        scenes: List[SceneSegment], 
        lines_per_minute: float = 22.0
    ) -> List[SceneSegment]:
        current_line_count = 0
        for scene in scenes:
            scene_lines = 1
            if scene.action_text:
                scene_lines += len(scene.action_text.splitlines())
            for d in scene.dialogues:
                scene_lines += 1 + len(d.text.splitlines())

            start_sec = (current_line_count / lines_per_minute) * 60.0
            end_sec = ((current_line_count + scene_lines) / lines_per_minute) * 60.0

            scene.time_range = TimeRange(
                start_time=format_seconds_to_timestamp(start_sec),
                end_time=format_seconds_to_timestamp(end_sec),
                is_estimated=True
            )

            d_current = current_line_count + 1
            for d in scene.dialogues:
                d_lines = 1 + len(d.text.splitlines())
                d_start = (d_current / lines_per_minute) * 60.0
                d_end = ((d_current + d_lines) / lines_per_minute) * 60.0
                d.time_range = TimeRange(
                    start_time=format_seconds_to_timestamp(d_start),
                    end_time=format_seconds_to_timestamp(d_end),
                    is_estimated=True
                )
                d_current += d_lines

            current_line_count += scene_lines

        return scenes

    @staticmethod
    def filter_scenes_by_timerange(
        scenes: List[SceneSegment], 
        start_time_str: str, 
        end_time_str: str
    ) -> List[SceneSegment]:
        win_start = timestamp_to_seconds(start_time_str)
        win_end = timestamp_to_seconds(end_time_str)

        filtered_scenes: List[SceneSegment] = []
        for scene in scenes:
            s_start = timestamp_to_seconds(scene.time_range.start_time) if scene.time_range else 0.0
            s_end = timestamp_to_seconds(scene.time_range.end_time) if scene.time_range else 0.0

            if s_end >= win_start and s_start <= win_end:
                filtered_dialogues = []
                for d in scene.dialogues:
                    d_start = timestamp_to_seconds(d.time_range.start_time) if d.time_range else s_start
                    d_end = timestamp_to_seconds(d.time_range.end_time) if d.time_range else s_end
                    if d_end >= win_start and d_start <= win_end:
                        filtered_dialogues.append(d)

                scene_copy = scene.model_copy(deep=True)
                scene_copy.dialogues = filtered_dialogues
                filtered_scenes.append(scene_copy)

        return filtered_scenes

    @staticmethod
    def parse_srt(srt_content: str) -> List[Dict[str, Any]]:
        if not srt_content or not srt_content.strip():
            return []

        # 1. Normalize line endings
        srt_content = srt_content.replace('\r\n', '\n').replace('\r', '\n')
        blocks = re.split(r'\n\s*\n', srt_content.strip())
        results = []

        for block in blocks:
            lines = [l.strip() for l in block.splitlines() if l.strip()]
            if not lines:
                continue

            # Find timestamp line containing '-->'
            time_line_idx = -1
            for idx, l in enumerate(lines):
                if '-->' in l:
                    time_line_idx = idx
                    break

            if time_line_idx != -1:
                time_line = lines[time_line_idx]
                times = time_line.split('-->')
                if len(times) >= 2:
                    start_ts = times[0].strip()
                    end_ts = times[1].strip()
                    text_lines = lines[time_line_idx + 1:]
                    if not text_lines:
                        continue
                    
                    full_text = " ".join(text_lines)
                    full_text = re.sub(r'<[^>]+>', '', full_text).strip()
                    
                    speaker = "Unknown"
                    if ':' in full_text and not full_text.startswith('http'):
                        spk, txt = full_text.split(':', 1)
                        spk_clean = re.sub(r'^[-*\s()]+|[-*\s()]+$', '', spk).strip()
                        spk_clean_no_paren = re.sub(r'\([^\)]*\)', '', spk_clean).strip()
                        if spk_clean_no_paren and len(spk_clean_no_paren.split()) <= 3:
                            speaker = spk_clean_no_paren
                            full_text = txt.strip()

                    full_text = re.sub(r'\([^\)]*\)', '', full_text)
                    full_text = re.sub(r'\[[^\]]*\]', '', full_text)
                    full_text = re.sub(r'^[-*\s]+', '', full_text).strip()

                    speaker = re.sub(r'^[-*\s()]+|[-*\s()]+$', '', speaker).strip()
                    if not speaker or len(speaker) <= 1 or (speaker.isupper() and len(speaker.split()) > 3):
                        speaker = "Unknown"
                    else:
                        speaker = speaker.title()

                    if not full_text:
                        continue

                    results.append({
                        "start_time": format_seconds_to_timestamp(parse_srt_timestamp(start_ts)),
                        "end_time": format_seconds_to_timestamp(parse_srt_timestamp(end_ts)),
                        "speaker": speaker,
                        "text": full_text,
                        "is_estimated": False
                    })

        # 2. Fallback line-by-line scanner if block splitting produced no results
        if not results:
            lines = [l.strip() for l in srt_content.splitlines() if l.strip()]
            i = 0
            while i < len(lines):
                if '-->' in lines[i]:
                    times = lines[i].split('-->')
                    start_ts = times[0].strip()
                    end_ts = times[1].strip()
                    txt_parts = []
                    i += 1
                    while i < len(lines) and '-->' not in lines[i] and not (lines[i].isdigit() and len(lines[i]) < 5):
                        txt_parts.append(lines[i])
                        i += 1
                    
                    full_text = " ".join(txt_parts)
                    full_text = re.sub(r'<[^>]+>', '', full_text).strip()
                    full_text = re.sub(r'\([^\)]*\)', '', full_text)
                    full_text = re.sub(r'\[[^\]]*\]', '', full_text).strip()
                    
                    if full_text:
                        results.append({
                            "start_time": format_seconds_to_timestamp(parse_srt_timestamp(start_ts)),
                            "end_time": format_seconds_to_timestamp(parse_srt_timestamp(end_ts)),
                            "speaker": "Unknown",
                            "text": full_text,
                            "is_estimated": False
                        })
                else:
                    i += 1

        return results
