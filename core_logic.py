import sqlite3
import os
import re
import chardet
import guessit
import time
import hashlib
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

DB_NAME = "subtitle_library.db"

# ==========================================
# 模块 1：数据库基础设施
# ==========================================
def get_db_connection():
    """获取数据库连接的安全工厂函数"""
    conn = sqlite3.connect(DB_NAME, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;") 
    return conn

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    # --- 创建表与索引 ---
    c.execute('''
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
            embedding BLOB
        )
    ''')
    # --- Schema 迁移：补缺失列 ---
    c.execute("PRAGMA table_info(subtitles)")
    existing_columns = {row[1] for row in c.fetchall()}
    required_columns = {
        "line_index": "INTEGER",
        "embedding": "BLOB",
        "file_hash": "TEXT",
        "embedding_model": "TEXT",
        "embedding_dim": "INTEGER"
    }
    for col_name, col_type in required_columns.items():
        if col_name not in existing_columns:
            print(f"🔄 检测到数据库缺失列 '{col_name}'，正在自动修补...")
            try:
                c.execute(f"ALTER TABLE subtitles ADD COLUMN {col_name} {col_type}")
            except Exception as e:
                print(f"❌ 修补列 '{col_name}' 失败: {e}")
    c.execute('CREATE INDEX IF NOT EXISTS idx_content ON subtitles (content)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_hash ON subtitles (file_hash)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_movie ON subtitles (movie_name)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_context ON subtitles (file_hash, line_index)')
    conn.commit()
    conn.close()

init_db()

def calculate_file_hash(file_bytes):
    return hashlib.md5(file_bytes).hexdigest()

# ==========================================
# 模块 2：解析引擎
# ==========================================
def detect_encoding(file_bytes):
    result = chardet.detect(file_bytes)
    encoding = result['encoding']
    if not encoding or result['confidence'] < 0.5: return 'utf-8-sig'
    if encoding.lower() == 'gb2312': return 'gb18030'
    return encoding

def clean_subtitle_text(text):
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'\{[^}]+\}', '', text)
    return text.strip()

def parse_ass_content(content):
    parsed_data = []
    lines = content.split('\n')
    for line in lines:
        if line.startswith('Dialogue:'):
            try:
                parts = line.split(',', 9)
                if len(parts) >= 10:
                    clean_text = parts[9].replace(r'\N', ' ').replace(r'\n', ' ')
                    clean_text = clean_subtitle_text(clean_text)
                    if clean_text:
                        parsed_data.append({'start': parts[1].strip(), 'end': parts[2].strip(), 'text': clean_text})
            except: continue
    return parsed_data

def parse_srt_content(content):
    content = content.replace('\r\n', '\n').replace('\r', '\n')
    # --- 方案 A：标准 SRT 正则 ---
    pattern_std = re.compile(
        r'\d+\s*\n'
        r'(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})'
        r'\s*\n([\s\S]*?)(?=\n\n|\n\d+\s*\n|\Z)',
        re.MULTILINE
    )
    matches = pattern_std.findall(content)
    if len(matches) > 0:
        parsed_data = []
        for m in matches:
            clean_text = clean_subtitle_text(m[2]).replace('\n', ' ')
            if clean_text:
                parsed_data.append({
                    'start': m[0].replace(',', '.'), 
                    'end': m[1].replace(',', '.'), 
                    'text': clean_text
                })
        return parsed_data
    # --- 方案 B：备用兼容（野生格式）---
    print("⚠️ 标准解析失败，启用备用兼容模式...")
    parsed_data = []
    pattern_fallback = re.compile(
        r'(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})'
        r'(.*)'
    )
    lines = content.split('\n')
    current_entry = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
        match = pattern_fallback.search(line)
        if match:
            if current_entry:
                parsed_data.append(current_entry)
            start, end, inline_text = match.groups()
            text_content = inline_text.strip()
            text_content = re.sub(r'\[.*?\]', '', text_content).strip()
            current_entry = {'start': start.replace(',', '.'), 'end': end.replace(',', '.'), 'text': text_content}
        else:
            if (
                current_entry and 
                not line.isdigit() and 
                not line.startswith('<') and 
                not line.startswith('[')
            ):
                current_entry['text'] = (current_entry['text'] + " " + line).strip()
    if current_entry:
        parsed_data.append(current_entry)
    return parsed_data

def parse_vtt_content(content):
    return parse_srt_content(content.replace('WEBVTT', ''))

# ==========================================
# 模块 3：业务处理（文件名解析、批量处理与入库）
# ==========================================
def analyze_filenames(uploaded_files):
    results = []
    for file in uploaded_files:
        info = guessit.guessit(file.name)
        title = info.get('title')
        if not title or title == '未知电影' or '[' in str(title):
            anime_match = re.search(r'^\[.*?\]\[(.*?)\]', file.name)
            if anime_match: title = anime_match.group(1).replace('_', ' ').strip()
            else:
                movie_match = re.search(r'^\[(.*?)\]', file.name)
                if movie_match: title = movie_match.group(1).strip()
        if not title: title = file.name
        s_num = info.get('season', 0)
        e_num = info.get('episode', 0)
        results.append({
            "原始文件名": file.name,
            "识别片名": title,
            "年份": info.get('year', 0),
            "season_num": s_num,
            "episode_num": e_num,
            "剧集": f"S{str(s_num).zfill(2)}E{str(e_num).zfill(2)}" if e_num else "",
            "状态": "待确认"
        })
    return results

# --- 批量处理：解析、落盘、待入库 ---
def _process_one_batch(file_objects, metadata_list, target_folder, conn, model, model_name=""):
    logs = []
    processed_files = []
    pending_rows = []
    stats = {"success": 0, "fail": 0, "duplicate": 0}
    c = conn.cursor()
    for idx, (file_obj, meta) in enumerate(zip(file_objects, metadata_list)):
        try:
            file_obj.seek(0)
            raw_bytes = file_obj.read()
        except Exception as e:
            logs.append(f"❌ 读取失败: {getattr(file_obj, 'name', 'file')} - {e}")
            stats["fail"] += 1
            continue
        f_hash = calculate_file_hash(raw_bytes)
        c.execute("SELECT movie_name FROM subtitles WHERE file_hash = ?", (f_hash,))
        if c.fetchone():
            logs.append(f"⚠️ 跳过重复: {getattr(file_obj, 'name', 'file')}")
            stats["duplicate"] += 1
            continue
        try:
            encoding = detect_encoding(raw_bytes)
            content = raw_bytes.decode(encoding, errors='replace')
        except Exception as e:
            logs.append(f"❌ 解码失败: {getattr(file_obj, 'name', 'file')} - {e}")
            stats["fail"] += 1
            continue
        ext = (getattr(file_obj, 'name', '') or '').split('.')[-1].lower()
        if '[Script Info]' in content[:1000] and 'Dialogue:' in content:
            subs = parse_ass_content(content)
        elif ext in ['ass', 'ssa']:
            subs = parse_ass_content(content)
        elif ext == 'vtt':
            subs = parse_vtt_content(content)
        else:
            subs = parse_srt_content(content)
        if not subs:
            logs.append(f"⚠️ 无有效字幕: {getattr(file_obj, 'name', 'file')}")
            stats["fail"] += 1
            continue
        embeddings = []
        if model:
            try:
                texts = [s['text'] for s in subs]
                if texts:
                    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            except Exception as e:
                logs.append(f"⚠️ 向量化计算失败: {e}")
        clean_title = str(meta.get('识别片名', '')).replace('/', '_').replace(':', ' ')
        new_name = f"{clean_title}"
        if meta.get('年份'): new_name += f" ({meta['年份']})"
        if meta.get('season_num') or meta.get('episode_num'):
            new_name += f".S{str(meta.get('season_num', 0)).zfill(2)}E{str(meta.get('episode_num', 0)).zfill(2)}"
        new_name += ".srt"
        save_path = os.path.join(target_folder, new_name)
        try:
            with open(save_path, 'w', encoding='utf-8') as f_out:
                f_out.write(content)
        except Exception as e:
            logs.append(f"❌ 写入失败: {new_name} - {e}")
            stats["fail"] += 1
            continue
        current_dim = 0
        if model and len(embeddings) > 0:
            current_dim = embeddings.shape[1]
        processed_files.append({"name": new_name, "content": content})
        for i, s in enumerate(subs):
            emb_blob = embeddings[i].tobytes() if (model and i < len(embeddings)) else None
            row = (f_hash, save_path, meta.get('识别片名'), meta.get('年份'),
                   meta.get('season_num'), meta.get('episode_num'),
                   i, s['start'], s['end'], s['text'], emb_blob,
                   model_name if model else None,
                   current_dim if model else 0
                   )
            pending_rows.append(row)
        status_tag = "(含索引)" if model else "(极速/无索引)"
        logs.append(f"✅ 处理完成 {status_tag}: {new_name}")
        stats["success"] += 1
    return logs, processed_files, stats, pending_rows

def process_only(file_objects, metadata_list, target_folder, model, model_name=""):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    conn = get_db_connection()
    try:
        return _process_one_batch(file_objects, metadata_list, target_folder, conn, model, model_name)
    finally:
        conn.close()

def commit_pending_to_db(pending_rows):
    if not pending_rows: return
    conn = get_db_connection()
    try:
        c = conn.cursor()
        c.executemany('INSERT INTO subtitles VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)', pending_rows)
        conn.commit()
    finally:
        conn.close()


# ==========================================
# 模块 4：统计与全量扫描
# ==========================================
def get_library_stats():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT movie_name) FROM subtitles")
    m_count = c.fetchone()[0]
    c.execute("SELECT COUNT(*) FROM subtitles")
    l_count = c.fetchone()[0]
    conn.close()
    return {"movie_count": m_count, "line_count": l_count, "last_update": time.strftime("%Y-%m-%d %H:%M:%S")}

def scan_library_path(library_path, model, model_name=""):
    """扫描目录并向量化入库，使用传入的模型对象。"""
    if not os.path.exists(library_path): 
        yield "❌ 错误: 路径不存在", None
        return
    if not model:
        yield "❌ 模型加载失败", None
        return
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT file_path FROM subtitles")
    db_files = {os.path.normpath(row[0]) for row in c.fetchall()}
    new_count = 0
    valid_exts = ('.srt', '.txt', '.ass', '.ssa', '.vtt')
    disk_files = set()
    for root, dirs, files in os.walk(library_path):
        for file in files:
            if file.lower().endswith(valid_exts):
                full_path = os.path.normpath(os.path.join(root, file))
                disk_files.add(full_path)
                if full_path not in db_files:
                    try:
                        with open(full_path, 'rb') as f: raw_bytes = f.read()
                        f_hash = calculate_file_hash(raw_bytes)
                        encoding = detect_encoding(raw_bytes)
                        content = raw_bytes.decode(encoding, errors='replace')
                        ext = file.lower().split('.')[-1]
                        if ext == 'txt': subs = [{'start':'0','end':'0','text':l.strip()} for l in content.split('\n') if l.strip()]
                        elif ext in ['ass','ssa']: subs = parse_ass_content(content)
                        elif ext == 'vtt': subs = parse_vtt_content(content)
                        else: subs = parse_srt_content(content)
                        if not subs: continue
                        texts = [s['text'] for s in subs]
                        embeddings = []
                        current_dim = 0
                        if texts:
                            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
                            if len(embeddings) > 0: current_dim = embeddings.shape[1]
                        info = guessit.guessit(file)
                        title = info.get('title') or file
                        s_num = info.get('season', 0)
                        e_num = info.get('episode', 0)
                        data_to_insert = []
                        for i, s in enumerate(subs):
                            emb_blob = embeddings[i].tobytes() if i < len(embeddings) else None
                            data_to_insert.append((
                                f_hash, full_path, title, info.get('year', 0), s_num, e_num,
                                i, s['start'], s['end'], s['text'], emb_blob,
                                model_name, current_dim
                            ))
                        c.executemany('INSERT INTO subtitles VALUES (NULL,?,?,?,?,?,?,?,?,?,?,?,?,?)', data_to_insert)
                        new_count += 1
                        yield f"✅ 已向量化入库: {file}", new_count
                    except Exception as e:
                        yield f"❌ 失败 {file}: {e}", new_count
    conn.commit()
    missing = list(db_files - disk_files)
    conn.close()
    yield "DONE", {"success": True, "new_added": new_count, "missing_files": missing}

def delete_missing_records(paths):
    conn = get_db_connection()
    c = conn.cursor()
    for p in paths: c.execute("DELETE FROM subtitles WHERE file_path = ?", (p,))
    conn.commit()
    conn.close()
    return True, "清理成功"

# ==========================================
# 模块 5：检索与 AI 接口
# ==========================================
def get_all_movies():
    """返回已入库电影名列表。"""
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT DISTINCT movie_name FROM subtitles ORDER BY movie_name")
    movies = [row[0] for row in c.fetchall()]
    conn.close()
    return movies

def search_db_keyword(keyword):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT movie_name, season, episode, start_time, content FROM subtitles WHERE content LIKE ? LIMIT 50", (f'%{keyword}%',))
    rows = c.fetchall()
    conn.close()
    return [{"movie": r[0], "season": r[1], "episode": r[2], "time": r[3], "content": r[4]} for r in rows]

def get_context_lines(movie_name, season, episode, line_index, window=2):
    conn = get_db_connection()
    c = conn.cursor()
    query = '''
        SELECT start_time, content, line_index 
        FROM subtitles 
        WHERE movie_name = ? AND season = ? AND episode = ?
          AND line_index BETWEEN ? AND ?
        ORDER BY line_index ASC
    '''
    start_idx = max(0, line_index - window)
    end_idx = line_index + window
    c.execute(query, (movie_name, season, episode, start_idx, end_idx))
    rows = c.fetchall()
    conn.close()
    return [{"time": r[0], "text": r[1], "is_target": (r[2] == line_index)} for r in rows]

def search_db_semantic(query, model, top_k=20):
    """基于 Embedding 的语义检索。"""
    if not model:
        return [{"movie": "系统错误", "time": "", "content": "模型对象为空"}]
    query_embedding = model.encode(query, convert_to_numpy=True)
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT movie_name, season, episode, start_time, content, embedding FROM subtitles WHERE embedding IS NOT NULL")
    rows = c.fetchall()
    conn.close()
    if not rows:
        return []
    db_embeddings = []
    valid_rows = []
    for r in rows:
        vec = np.frombuffer(r[5], dtype=np.float32)
        db_embeddings.append(vec)
        valid_rows.append(r)

    if not db_embeddings:
        return []
    try:
        corpus_embeddings = np.stack(db_embeddings)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    except RuntimeError as e:
        error_msg = str(e)
        if "shapes cannot be multiplied" in error_msg or "mat1 and mat2" in error_msg:
            return [{
                "movie": "❌ 维度错误", 
                "time": "00:00:00", 
                "content": f"模型维度不匹配！当前库是旧模型的索引。请去[数据库管理]点击[重新扫描硬盘]以重建索引。"
            }]
        raise e
    top_results = []
    top_scores, top_indices = cos_scores.topk(min(top_k, len(cos_scores)))
    for score, idx in zip(top_scores, top_indices):
        r = valid_rows[idx]
        top_results.append({
            "movie": r[0],
            "season": r[1],
            "episode": r[2],
            "time": r[3],
            "content": r[4],
            "score": float(score)
        })
    return top_results

# --- LLM 调用 ---
def call_deepseek_llm(system_prompt, user_prompt, api_key):
    """DeepSeek/OpenAI 兼容调用。"""
    if not api_key:
        raise ValueError("未提供 API Key")
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=False,
            temperature=0.7,
            max_tokens=4000
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ API 调用失败：{str(e)}"

def generate_script(movie_name, api_key):
    """调用 LLM 生成混剪脚本。"""
    sys_prompt = "你是一位电影预告片剪辑大师..."
    usr_prompt = f"电影名称：《{movie_name}》..."
    return call_deepseek_llm(sys_prompt, usr_prompt, api_key)

def extract_golden_quotes(subtitle_text, api_key):
    """金句提炼（预留）。"""
    sys_prompt = "你是一个文学鉴赏家。请从以下文本中找出 5 句最富有哲理的金句。"
    usr_prompt = subtitle_text[:5000]
    return call_deepseek_llm(sys_prompt, usr_prompt, api_key)