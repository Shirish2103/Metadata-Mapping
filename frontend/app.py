import os
import requests
import pandas as pd
import streamlit as st

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="AI Movie Metadata Intelligence",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Base URL for FastAPI backend
BACKEND_URL = os.getenv("BACKEND_URL")
if not BACKEND_URL and hasattr(st, "secrets") and "BACKEND_URL" in st.secrets:
    BACKEND_URL = st.secrets["BACKEND_URL"]
if not BACKEND_URL:
    BACKEND_URL = "http://127.0.0.1:8000"

# Custom CSS for Premium Design & Visual Excellence
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&family=Outfit:wght@400;600;700;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .main-title {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }
    
    .hero-container {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 24px 32px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .title-gradient {
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.4rem;
        margin-bottom: 4px;
    }
    
    .card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 22px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .card:hover {
        border-color: rgba(168, 85, 247, 0.4);
        box-shadow: 0 8px 25px rgba(168, 85, 247, 0.15);
    }
    
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
    }
    
    .badge-positive {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-negative {
        background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }
    
    .badge-neutral {
        background: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .badge-mixed {
        background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.85rem;
    }

    .chip {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #a5b4fc;
        border: 1px solid rgba(129, 140, 248, 0.3);
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .chip-entity {
        display: inline-block;
        background: rgba(236, 72, 153, 0.15);
        color: #f472b6;
        border: 1px solid rgba(244, 114, 182, 0.3);
        border-radius: 20px;
        padding: 3px 12px;
        margin: 3px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .dialogue-item {
        background: rgba(15, 23, 42, 0.5);
        border-left: 4px solid #818cf8;
        padding: 10px 16px;
        margin-bottom: 10px;
        border-radius: 0 8px 8px 0;
    }
    
    .dialogue-speaker {
        font-weight: 700;
        color: #c084fc;
        font-size: 0.95rem;
    }
    
    .dialogue-time {
        font-family: 'JetBrains Mono', monospace;
        color: #64748b;
        font-size: 0.8rem;
        float: right;
    }
    
    .location-header {
        font-weight: 700;
        color: #38bdf8;
        background: rgba(56, 189, 248, 0.1);
        padding: 6px 14px;
        border-radius: 6px;
        margin-top: 14px;
        margin-bottom: 8px;
        display: inline-block;
        border: 1px solid rgba(56, 189, 248, 0.2);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def fetch_available_movies():
    """Fetches list of available movies from FastAPI backend."""
    try:
        res = requests.get(f"{BACKEND_URL}/api/movies", timeout=5)
        if res.status_code == 200:
            return res.json()
    except Exception:
        pass
    return []


def render_sentiment_badge(sentiment: str):
    s = (sentiment or "").strip().capitalize()
    if s == "Positive":
        return f'<span class="badge-positive">🟢 {s}</span>'
    elif s == "Negative":
        return f'<span class="badge-negative">🔴 {s}</span>'
    elif s == "Mixed":
        return f'<span class="badge-mixed">🟠 {s}</span>'
    else:
        return f'<span class="badge-neutral">🔵 {s if s else "Neutral"}</span>'


def render_pretty_metadata(result: dict):
    """Renders the metadata result in a clean, pretty format."""
    movie_info = result.get("movie_info", {}) or {}
    time_range = result.get("time_range", {}) or {}
    char_pres = result.get("character_presence", {}) or {}
    topics = result.get("topics", {}) or {}
    entities = result.get("entities", {}) or {}
    sentiment = result.get("sentiment", {}) or {}
    category = result.get("category", {}) or {}
    dialogues = result.get("dialogues_in_window", []) or []
    speakers = result.get("speaker_list", []) or []
    
    title = result.get("title") or movie_info.get("title") or "Movie Analysis"
    imdb_id = result.get("imdb_id") or movie_info.get("imdb_id") or "N/A"
    total_duration = result.get("total_duration") or time_range.get("total_duration") or "N/A"
    start_ts = time_range.get("start_time", "00:00:00")
    end_ts = time_range.get("end_time", total_duration)

    fetch_src = result.get("fetch_source")
    if fetch_src == "database":
        st.markdown("""
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.4); border-radius: 10px; padding: 12px 20px; margin-bottom: 20px; color: #34d399; font-weight: 700; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.2rem;">⚡</span> <span>Data Fetched directly from SQLite Database Cache</span>
        </div>
        """, unsafe_allow_html=True)
    elif fetch_src == "extracted_and_saved":
        st.markdown("""
        <div style="background: rgba(129, 140, 248, 0.15); border: 1px solid rgba(129, 140, 248, 0.4); border-radius: 10px; padding: 12px 20px; margin-bottom: 20px; color: #a5b4fc; font-weight: 700; display: flex; align-items: center; gap: 10px;">
            <span style="font-size: 1.2rem;">✨</span> <span>Fresh Metadata Extracted & Saved to SQLite Database</span>
        </div>
        """, unsafe_allow_html=True)

    # --- HERO SUMMARY CONTAINER ---
    st.markdown(f"""
    <div class="hero-container">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;">
            <div>
                <div class="title-gradient">🎬 {title}</div>
                <div style="color: #94a3b8; font-size: 0.95rem; margin-bottom: 12px;">
                    IMDB ID: <strong style="color:#e2e8f0">{imdb_id}</strong> | Year: <strong style="color:#e2e8f0">{movie_info.get('year', 'N/A')}</strong> | 
                    Genres: <span style="color:#a5b4fc">{', '.join(movie_info.get('genres', [])) if movie_info.get('genres') else 'Entertainment'}</span>
                </div>
            </div>
            <div>
                {render_sentiment_badge(sentiment.get('sentiment', 'Neutral'))}
            </div>
        </div>
        <div style="color: #cbd5e1; font-size: 0.95rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px; margin-top: 6px;">
            <strong>📖 Plot / Window Overview:</strong> {movie_info.get('plot', 'No plot summary available.')}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- METRICS ROW ---
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">⏱️ Total Runtime</div>
            <div class="metric-value">{total_duration}</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">🎯 Time Window</div>
            <div class="metric-value" style="font-size:1.2rem;">[{start_ts} - {end_ts}]</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">🎬 Scenes Filtered</div>
            <div class="metric-value">{result.get('scene_breakdown_count', 0)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">🗣️ Active Speakers</div>
            <div class="metric-value">{len(speakers)}</div>
        </div>
        """, unsafe_allow_html=True)
    with m5:
        st.markdown(f"""
        <div class="card">
            <div class="metric-label">🏷️ Primary Genre</div>
            <div class="metric-value" style="font-size:1.1rem; color:#c084fc;">{category.get('primary_category', 'General')}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # --- TABBED METADATA BREAKDOWN ---
    tab_pacing, tab_dialogue, tab_topics, tab_entities, tab_json = st.tabs([
        "🎭 Character Screen Time & Pacing", 
        "💬 Timeline Dialogue Breakdown", 
        "🏷️ Topics & Classification", 
        "🔍 Named Entities & Sentiment",
        "📄 Raw Data (JSON)"
    ])

    # 1. CHARACTER PACING & SCREEN TIME TABLE
    with tab_pacing:
        st.subheader("⏱️ Character Scene Presence & Pacing Matrix")
        characters = char_pres.get("characters", [])
        total_scenes = char_pres.get("total_movie_scenes", 0)

        if not characters:
            st.info("No detailed character screen time data available for this selection.")
        else:
            st.caption(f"Total Scenes Analyzed: **{total_scenes}** | Total Duration: **{total_duration}**")
            
            table_data = []
            for c in characters:
                st_pct = round(float(c.get("screen_time_percentage", 0.0) or 0.0), 2)

                table_data.append({
                    "Character": c.get("character_name", "Unknown"),
                    "Role": c.get("role_type", "Minor"),
                    "Screen Time %": st_pct,
                    "Scenes": c.get("scene_count", 0),
                    "Lines": c.get("dialogue_line_count", 0),
                    "First Scene Entry": f"Scene {c.get('first_scene_idx', 0)} ({c.get('first_timestamp', '00:00:00')})",
                    "Last Scene Exit": f"Scene {c.get('last_scene_idx', 0)} ({c.get('last_timestamp', '00:00:00')})"
                })
            
            df = pd.DataFrame(table_data)
            st.dataframe(
                df,
                use_container_width=True,
                column_config={
                    "Screen Time %": st.column_config.ProgressColumn(
                        "Screen Time %",
                        format="%.1f%%",
                        min_value=0,
                        max_value=100,
                    ),
                },
                hide_index=True
            )

    # 2. TIMELINE DIALOGUE BREAKDOWN
    with tab_dialogue:
        st.subheader(f"💬 Timeline Dialogue Stream [{start_ts} - {end_ts}]")
        
        if not dialogues:
            st.info("No dialogues found in the specified timestamp range.")
        else:
            col_filter1, col_filter2 = st.columns([1, 2])
            with col_filter1:
                selected_speaker = st.selectbox(
                    "Filter by Speaker:", 
                    ["All Speakers"] + sorted(list(set(d.get("speaker", "Unknown") for d in dialogues)))
                )
            with col_filter2:
                search_query = st.text_input("Search Dialogue Text:", placeholder="Type a keyword to filter dialogues...")

            filtered_dialogues = dialogues
            if selected_speaker != "All Speakers":
                filtered_dialogues = [d for d in filtered_dialogues if d.get("speaker") == selected_speaker]
            if search_query:
                filtered_dialogues = [d for d in filtered_dialogues if search_query.lower() in d.get("text", "").lower()]

            st.caption(f"Showing **{len(filtered_dialogues)}** dialogues")

            current_location = None
            for d in filtered_dialogues[:60]:
                loc = d.get("location") or "SCENE LOCATION"
                ts = d.get("timestamp", "00:00:00")
                spk = d.get("speaker", "Unknown")
                text = d.get("text", "")

                if loc != current_location:
                    current_location = loc
                    st.markdown(f'<div class="location-header">📍 LOCATION: {current_location}</div>', unsafe_allow_html=True)

                st.markdown(f"""
                <div class="dialogue-item">
                    <span class="dialogue-speaker">🗣️ {spk}</span>
                    <span class="dialogue-time">⏱️ {ts}</span>
                    <div style="margin-top:4px; color:#e2e8f0; font-size:0.95rem;">"{text}"</div>
                </div>
                """, unsafe_allow_html=True)

            if len(filtered_dialogues) > 60:
                st.info(f"Showing first 60 of {len(filtered_dialogues)} matching dialogues.")

    # 3. TOPICS & CLASSIFICATION
    with tab_topics:
        col_cat, col_top = st.columns(2)
        
        with col_cat:
            st.subheader("🎯 Primary & Secondary Category")
            p_cat = category.get("primary_category", "N/A")
            s_cats = category.get("secondary_categories", [])
            conf = category.get("confidence", 1.0)
            reason = category.get("reasoning", "")

            st.markdown(f"**Primary Category:** <span class='chip' style='font-size:1rem; font-weight:700;'>{p_cat}</span>", unsafe_allow_html=True)
            if s_cats:
                s_chips = "".join([f"<span class='chip'>{sc}</span>" for sc in s_cats])
                st.markdown(f"**Secondary Categories:** {s_chips}", unsafe_allow_html=True)
            
            st.progress(min(1.0, max(0.0, float(conf))), text=f"Category Confidence: {conf * 100:.1f}%")
            
            if reason:
                st.info(f"**Reasoning:** {reason}")

        with col_top:
            st.subheader("🏷️ Topic & Keyword Extraction")
            main_t = topics.get("main_topics", [])
            subj = topics.get("subjects", [])
            freq_terms = topics.get("frequently_mentioned_terms", [])
            kw = topics.get("keywords", [])

            if main_t:
                st.markdown("**Main Thematic Topics:**")
                st.markdown("".join([f"<span class='chip'>{t}</span>" for t in main_t]), unsafe_allow_html=True)

            if subj:
                st.markdown("**Subjects & Themes:**")
                st.markdown("".join([f"<span class='chip'>{s}</span>" for s in subj]), unsafe_allow_html=True)

            if freq_terms:
                st.markdown("**TF-IDF Key Terms:**")
                st.markdown("".join([f"<span class='chip'>{item.split(':')[0]}</span>" for item in freq_terms]), unsafe_allow_html=True)

    # 4. NAMED ENTITIES & SENTIMENT
    with tab_entities:
        c_ent, c_sent = st.columns(2)
        
        with c_ent:
            st.subheader("🔍 Extracted Named Entities")
            peop = entities.get("people", [])
            locs = entities.get("locations", [])
            orgs = entities.get("organizations", [])
            prods = entities.get("products", [])

            if peop:
                st.markdown("**👤 People / Characters:**")
                st.markdown("".join([f"<span class='chip-entity'>{p}</span>" for p in peop[:15]]), unsafe_allow_html=True)

            if locs:
                st.markdown("**📍 Locations:**")
                st.markdown("".join([f"<span class='chip-entity'>{l}</span>" for l in locs[:15]]), unsafe_allow_html=True)

            if orgs:
                st.markdown("**🏢 Organizations:**")
                st.markdown("".join([f"<span class='chip-entity'>{o}</span>" for o in orgs[:15]]), unsafe_allow_html=True)

            if prods:
                st.markdown("**📦 Products / Items:**")
                st.markdown("".join([f"<span class='chip-entity'>{pr}</span>" for pr in prods[:15]]), unsafe_allow_html=True)

        with c_sent:
            st.subheader("🎭 Sentiment & Emotional Tone")
            sent_val = sentiment.get("sentiment", "Neutral")
            emotions = sentiment.get("emotions", [])
            conf = sentiment.get("confidence", 1.0)

            st.markdown(f"**Overall Sentiment:** {render_sentiment_badge(sent_val)}", unsafe_allow_html=True)
            st.progress(min(1.0, max(0.0, float(conf))), text=f"Sentiment Confidence: {conf * 100:.1f}%")

            if emotions:
                st.markdown("**Detected Emotional Tones:**")
                st.markdown("".join([f"<span class='chip'>{e}</span>" for e in emotions]), unsafe_allow_html=True)

    # 5. RAW DATA JSON
    with tab_json:
        st.subheader("📄 Complete Extracted Metadata Payload")
        st.json(result)


def main():
    # --- HEADER & SIDEBAR ---
    st.sidebar.image("https://img.icons8.com/color/96/movie-projector.png", width=64)
    st.sidebar.title("AI Metadata Control")
    st.sidebar.markdown("Configure analysis target and time duration window.")

    available_movies = fetch_available_movies()
    movie_options = [f"{m['title']} ({m['year']})" for m in available_movies] if available_movies else []

    # Navigation Modes
    mode = st.sidebar.radio(
        "Select Operation Mode:", 
        ["🎬 Analyze Movie by Name & Duration", "📁 Upload SRT Subtitle File"]
    )

    if mode == "🎬 Analyze Movie by Name & Duration":
        st.markdown("<h1 class=\"title-gradient\">🎬 AI Screenplay Metadata Extractor</h1>", unsafe_allow_html=True)
        st.markdown("Enter any movie title and specify an optional time duration to extract structured metadata in real-time.")

        col_input1, col_input2, col_input3 = st.columns([2, 1, 1])

        with col_input1:
            if movie_options:
                selected_from_list = st.selectbox("Select from Available Dataset Movies:", ["-- Or type custom movie below --"] + movie_options)
            else:
                selected_from_list = "-- Or type custom movie below --"

            custom_title = st.text_input(
                "Movie Title or IMDB ID:", 
                value="" if selected_from_list != "-- Or type custom movie below --" else "Full Metal Jacket",
                placeholder="e.g. Full Metal Jacket, The Godfather, tt0093058"
            )
            
            # Resolve actual query title
            if selected_from_list != "-- Or type custom movie below --":
                target_movie_name = selected_from_list.split(" (")[0]
            else:
                target_movie_name = custom_title

        with col_input2:
            start_time_in = st.text_input("Start Time (Duration):", value="", placeholder="e.g. 00:10:00 or 10:00")

        with col_input3:
            end_time_in = st.text_input("End Time (Duration):", value="", placeholder="e.g. 00:30:00 or 30:00")

        force_refresh = st.checkbox("🔄 Force Re-extract (Bypass Database Cache)", value=False)

        analyze_btn = st.button("🚀 Fetch / Extract Metadata", type="primary", use_container_width=True)

        if analyze_btn:
            if not target_movie_name.strip():
                st.error("Please enter or select a movie title!")
                return

            with st.spinner(f"Retrieving/Extracting metadata for '{target_movie_name}'..."):
                try:
                    payload = {
                        "title_or_imdb": target_movie_name.strip(),
                        "start_time": start_time_in.strip() if start_time_in else None,
                        "end_time": end_time_in.strip() if end_time_in else None,
                        "force_refresh": force_refresh
                    }
                    res = requests.post(f"{BACKEND_URL}/api/process", json=payload, timeout=60)
                    
                    if res.status_code == 200:
                        st.session_state["current_result"] = res.json()
                        st.success("Metadata loaded successfully!")
                    else:
                        st.error(f"Backend Error ({res.status_code}): {res.json().get('detail', res.text)}")
                except Exception as e:
                    st.error(f"Failed to communicate with FastAPI backend ({BACKEND_URL}). Error: {str(e)}")

        if "current_result" in st.session_state:
            render_pretty_metadata(st.session_state["current_result"])

    elif mode == "📁 Upload SRT Subtitle File":
        st.markdown("<h1 class=\"title-gradient\">📁 Upload SRT Subtitle Transcript</h1>", unsafe_allow_html=True)
        st.markdown("Upload any timestamped `.srt` subtitle file to perform automated metadata extraction.")

        uploaded_file = st.file_uploader("Choose an SRT transcript file:", type=["srt"])
        
        col_srt1, col_srt2 = st.columns(2)
        with col_srt1:
            srt_start = st.text_input("SRT Start Time Filter:", value="", placeholder="e.g. 00:05:00")
        with col_srt2:
            srt_end = st.text_input("SRT End Time Filter:", value="", placeholder="e.g. 00:25:00")

        if uploaded_file and st.button("🚀 Process SRT Metadata", type="primary"):
            with st.spinner("Parsing SRT and running metadata extraction..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/x-subrip")}
                    data = {
                        "start_time": srt_start.strip() if srt_start else "",
                        "end_time": srt_end.strip() if srt_end else ""
                    }
                    res = requests.post(f"{BACKEND_URL}/api/process-srt", files=files, data=data, timeout=60)
                    
                    if res.status_code == 200:
                        st.session_state["srt_result"] = res.json()
                        st.success("SRT Metadata extracted successfully!")
                    else:
                        st.error(f"SRT Error ({res.status_code}): {res.json().get('detail', res.text)}")
                except Exception as e:
                    st.error(f"SRT processing request failed: {str(e)}")

        if "srt_result" in st.session_state:
            render_pretty_metadata(st.session_state["srt_result"])


if __name__ == "__main__":
    main()
