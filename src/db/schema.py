# SQL DDL Definitions

SQL_CREATE_SUBTITLES = """
CREATE TABLE IF NOT EXISTS subtitles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_hash TEXT,
    file_path TEXT,
    movie_name TEXT,
    year INTEGER,
    season INTEGER,
    episode INTEGER,
    line_index INTEGER,
    start_time TEXT,
    end_time TEXT,
    content TEXT,
    embedding BLOB,
    embedding_model TEXT,
    embedding_dim INTEGER
)
"""

SQL_CREATE_MOVIES_META = """
CREATE TABLE IF NOT EXISTS movies_meta (
    movie_name TEXT PRIMARY KEY,
    poster_url TEXT,
    release_year INTEGER,
    highlight_status INTEGER DEFAULT 0  -- 0: 未挖掘, 1: 正在挖掘, 2: 已挖掘, 3: 失败
)
"""

SQL_CREATE_GOLDEN_QUOTES = """
CREATE TABLE IF NOT EXISTS golden_quotes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    movie_name TEXT,
    quote_content TEXT,
    timestamp TEXT,
    reason TEXT,
    FOREIGN KEY(movie_name) REFERENCES movies_meta(movie_name)
)
"""

SQL_CREATE_BGM_LIBRARY = """
CREATE TABLE IF NOT EXISTS bgm_library (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_name TEXT UNIQUE,
    valence REAL,
    energy REAL,
    tempo INTEGER,
    tags TEXT
)
"""

SQL_CREATE_TAGS = """
CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    UNIQUE(name, type)
)
"""

SQL_CREATE_SUBTITLE_TAGS = """
CREATE TABLE IF NOT EXISTS subtitle_tags (
    subtitle_id INTEGER,
    tag_id INTEGER,
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'manual',
    PRIMARY KEY (subtitle_id, tag_id),
    FOREIGN KEY(subtitle_id) REFERENCES subtitles(id),
    FOREIGN KEY(tag_id) REFERENCES tags(id)
)
"""

SQL_CREATE_MOVIE_TAGS = """
CREATE TABLE IF NOT EXISTS movie_tags (
    movie_name TEXT,
    tag_id INTEGER,
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'manual',
    PRIMARY KEY (movie_name, tag_id),
    FOREIGN KEY(movie_name) REFERENCES movies_meta(movie_name),
    FOREIGN KEY(tag_id) REFERENCES tags(id)
)
"""

SQL_CREATE_QUOTE_TAGS = """
CREATE TABLE IF NOT EXISTS quote_tags (
    quote_id INTEGER,
    tag_id INTEGER,
    confidence REAL DEFAULT 1.0,
    source TEXT DEFAULT 'llm',
    PRIMARY KEY (quote_id, tag_id),
    FOREIGN KEY(quote_id) REFERENCES golden_quotes(id),
    FOREIGN KEY(tag_id) REFERENCES tags(id)
)
"""

DEFAULT_TAGS = [
    # theme 主题
    ("遗憾", "theme"), ("时间", "theme"), ("父子", "theme"), ("死亡", "theme"), ("孤独", "theme"), ("重逢", "theme"),
    # emotion 情绪
    ("悲伤", "emotion"), ("释然", "emotion"), ("愤怒", "emotion"), ("燃", "emotion"), ("治愈", "emotion"), ("压抑", "emotion"),
    # scene 场景
    ("雨夜", "scene"), ("告别", "scene"), ("奔跑", "scene"), ("独白", "scene"), ("争吵", "scene"), ("沉默", "scene"),
    # usage 用途
    ("开场", "usage"), ("转场", "usage"), ("高潮", "usage"), ("结尾", "usage"), ("金句升华", "usage"),
    # style 风格
    ("文艺", "style"), ("悬疑", "style"), ("史诗", "style"), ("日常", "style"), ("黑色幽默", "style")
]
