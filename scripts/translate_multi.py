import os, re, json, time, hashlib, html
from urllib.parse import urlparse
from datetime import datetime, timezone
from email.utils import format_datetime
from dateutil import parser as dateparser

import requests
import feedparser

STATE_PATH = "data/state.json"
OUT_DIR = "public"
FEEDS_DIR = os.path.join(OUT_DIR, "feeds")
ALL_PATH = os.path.join(OUT_DIR, "all.xml")
INDEX_PATH = os.path.join(OUT_DIR, "index.html")

AI_BASE_URL = (os.environ.get("AI_BASE_URL", "") or "").rstrip("/")
AI_API_KEY = os.environ.get("AI_API_KEY", "") or ""
AI_MODEL = os.environ.get("AI_MODEL", "") or ""

RSSHUB_BASE_URL = (os.environ.get("RSSHUB_BASE_URL", "https://rsshub.app") or "").rstrip("/")
FEED_URLS_RAW = (os.environ.get("FEED_URLS", "") or "").strip()

MAX_NEW_ITEMS_PER_FEED = int(os.environ.get("MAX_NEW_ITEMS_PER_FEED", "6"))
MAX_TOTAL_NEW_ITEMS = int(os.environ.get("MAX_TOTAL_NEW_ITEMS", "40"))
KEEP_ITEMS_PER_FEED = int(os.environ.get("KEEP_ITEMS_PER_FEED", "120"))

REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "30"))

UA = os.environ.get(
    "HTTP_USER_AGENT",
    "Mozilla/5.0 (compatible; rss-bilingual-bot/1.0; +https://github.com/)"
)

# rsshub:// 简写别名
RSSHUB_ALIAS = {
    "hackernews": "hn/newest",
}

# -----------------------
# State
# -----------------------

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"seen": {}, "translations": {}}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# -----------------------
# Text helpers
# -----------------------

def strip_html(text: str) -> str:
    """
    将HTML压成纯文本，但尽量保留链接信息和列表/段落结构线索，方便AI生成排版。
    """
    text = text or ""
    # a 标签：保留 href
    text = re.sub(r'<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', r'\2 (\1)', text, flags=re.I|re.S)
    # 段落/换行/列表做结构提示
    text = re.sub(r'</p\s*>', '\n\n', text, flags=re.I)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.I)
    text = re.sub(r'</li\s*>', '\n', text, flags=re.I)
    text = re.sub(r'<li[^>]*>', '- ', text, flags=re.I)

    # 去掉脚本样式
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S|re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S|re.I)
    # 去掉其余标签
    text = re.sub(r"<[^>]+>", " ", text)

    text = html.unescape(text)
    # 保留换行（别全压成一行）
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def clip(text: str, n: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    while u.endswith("?"):
        u = u[:-1]
    if u.startswith("rsshub://"):
        route = u[len("rsshub://"):].lstrip("/")
        if "/" not in route and route in RSSHUB_ALIAS:
            route = RSSHUB_ALIAS[route]
        return f"{RSSHUB_BASE_URL}/{route}"
    return u

def slugify(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s or "feed"

def derive_name_and_slug(name_hint: str | None, url: str) -> tuple[str, str]:
    if name_hint:
        return name_hint.strip(), slugify(name_hint)

    u = urlparse(url)
    host = (u.netloc or "").lower()
    path = (u.path or "").strip("/")

    m = re.search(r"reddit\.com/r/([^/]+)/?\.rss", url, re.I)
    if m:
        n = f"reddit/{m.group(1)}"
        return n, slugify(f"reddit-{m.group(1)}")

    if "xgo.ing" in host:
        last = path.split("/")[-1] if path else "xgo"
        n = f"xgo/{last[:12]}"
        return n, slugify(f"xgo-{last[:12]}")

    if "rsshub" in host:
        n = f"rsshub/{path}"
        return n, slugify(path)

    base = host if host else "feed"
    n = f"{base}/{path}" if path else base
    return n, slugify(n)

def parse_feed_lines(raw: str):
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            name, url = line.split("|", 1)
            yield name.strip(), url.strip()
            continue
        parts = line.split()
        if len(parts) >= 2 and (parts[1].startswith("http") or parts[1].startswith("rsshub://")):
            yield parts[0].strip(), " ".join(parts[1:]).strip()
        else:
            yield None, line

# -----------------------
# Fetch feed
# -----------------------

def fetch_feed(url: str):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return feedparser.parse(r.content)

def entry_uid(entry) -> str:
    if getattr(entry, "id", None):
        return str(entry.id)
    link = getattr(entry, "link", "") or ""
    title = getattr(entry, "title", "") or ""
    raw = f"{link}||{title}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def parse_time(entry) -> datetime:
    for k in ("published", "updated", "created"):
        v = getattr(entry, k, None)
        if v:
            try:
                dt = dateparser.parse(v)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.astimezone(timezone.utc)
            except Exception:
                pass
    return datetime.now(timezone.utc)

# -----------------------
# LLM call (OpenAI-compatible variants)
# -----------------------

def chat_completions_urls(base: str) -> list[str]:
    base = (base or "").rstrip("/")
    urls = []
    if base.endswith("/v1"):
        urls.append(base + "/chat/completions")
    else:
        urls.append(base + "/v1/chat/completions")
        urls.append(base + "/chat/completions")
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out

def call_llm(prompt: str) -> str:
    if not (AI_BASE_URL and AI_API_KEY and AI_MODEL):
        raise RuntimeError("Missing AI_BASE_URL / AI_API_KEY / AI_MODEL")

    headers = {"Authorization": f"Bearer {AI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": AI_MODEL,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "你是一个高质量RSS中文摘要与改写助手。必须严格按用户要求输出JSON。"},
            {"role": "user", "content": prompt},
        ],
    }

    last_err = None
    for url in chat_completions_urls(AI_BASE_URL):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            if resp.status_code == 404:
                last_err = RuntimeError(f"404 for {url}")
                continue
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"All chat endpoints failed. Last error: {last_err}")

# -----------------------
# Translate + format to HTML
# -----------------------

def translate_to_zh_html(title_en: str, summary_en: str, state) -> tuple[str, str]:
    title_en = (title_en or "").strip()
    summary_en = (summary_en or "").strip()

    cache_key = hashlib.sha256((title_en + "\n" + summary_en).encode("utf-8")).hexdigest()
    if cache_key in state["translations"]:
        t = state["translations"][cache_key]
        return t["zh_title"], t["zh_html"]

    prompt = f"""
把下面RSS条目改写成中文，并排版成适合RSS展示的HTML：
要求：
1) zh_title：<=40字，像新闻标题；不要带英文；不要用“/”
2) zh_html：必须是HTML片段，用<p>分段；有要点就用<ol><li>…</li></ol>
3) 若出现URL，请尽量变成<a href="URL">简短文本</a>，不要裸URL
4) 内容过长：先给“长话短说”摘要（<=180字），再给要点
5) 如果出现“来源/Source/Sources: [1] ... [2] ...”这类，请整理成“来源”段落并用<a>链接
6) 不要添加原文没有的信息
7) 输出严格JSON，键只能有 zh_title 和 zh_html

英文标题：
{title_en}

英文摘要/片段：
{summary_en}
""".strip()

    content = call_llm(prompt)

    zh_title, zh_html = "", ""
    try:
        start = content.find("{")
        end = content.rfind("}")
        obj = json.loads(content[start:end + 1])
        zh_title = (obj.get("zh_title") or "").strip()
        zh_html = (obj.get("zh_html") or "").strip()
    except Exception:
        zh_title = ""
        zh_html = f"<p>{html.escape(content.strip())}</p>"

    # 最小兜底：确保至少<p>包裹
    if zh_html and not re.search(r"<\s*p\b|<\s*ol\b|<\s*ul\b", zh_html, flags=re.I):
        zh_html = f"<p>{zh_html}</p>"

    state["translations"][cache_key] = {"zh_title": zh_title, "zh_html": zh_html}
    return zh_title, zh_html

# -----------------------
# Read existing items (prefer content:encoded)
# -----------------------

def read_existing_items(path: str):
    if not os.path.exists(path):
        return []
    d = feedparser.parse(path)
    items = []
    for e in d.entries:
        html_val = ""
        if getattr(e, "content", None) and isinstance(e.content, list) and e.content:
            html_val = e.content[0].get("value", "") or ""
        if not html_val:
            html_val = e.get("summary", "") or e.get("description", "") or ""

        items.append({
            "id": e.get("id") or e.get("guid") or e.get("link") or hashlib.sha256((e.get("title","")+e.get("link","")).encode("utf-8")).hexdigest(),
            "title": e.get("title", ""),
            "html": html_val,
            "link": e.get("link", ""),
            "published": parse_time(e),
        })
    return items

# -----------------------
# Write RSS with CDATA + content:encoded (avoid "一坨")
# -----------------------

def _cdata(s: str) -> str:
    s = (s or "").replace("]]>", "]]]]><![CDATA[>")
    return f"<![CDATA[{s}]]>"

def _esc(s: str) -> str:
    return html.escape(s or "", quote=True)

def _rfc822(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return format_datetime(dt.astimezone(timezone.utc))

def write_feed(path: str, title: str, desc: str, items: list[dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rss = []
    rss.append('<?xml version="1.0" encoding="UTF-8"?>\n')
    rss.append('<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">\n')
    rss.append('<channel>\n')
    rss.append(f'<title>{_esc(title)}</title>\n')
    rss.append(f'<link>{_esc("https://github.com/")}</link>\n')
    rss.append(f'<description>{_esc(desc)}</description>\n')

    for it in items:
        link = it.get("link", "")
        html_fragment = it.get("html", "") or ""

        # 永远补一个“原文链接”
        if link:
            html_fragment += f"<p><a href='{_esc(link)}'>原文链接</a></p>"

        rss.append('<item>\n')
        rss.append(f'<title>{_esc(it.get("title",""))}</title>\n')
        rss.append(f'<link>{_esc(link)}</link>\n')
        rss.append(f"<guid isPermaLink='false'>{_esc(it.get('id',''))}</guid>\n")
        rss.append(f'<pubDate>{_esc(_rfc822(it.get("published")))}</pubDate>\n')
        rss.append(f'<description>{_cdata(html_fragment)}</description>\n')
        rss.append(f'<content:encoded>{_cdata(html_fragment)}</content:encoded>\n')
        rss.append('</item>\n')

    rss.append('</channel>\n</rss>\n')

    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(rss))

# -----------------------
# Main
# -----------------------

def main():
    if not FEED_URLS_RAW:
        raise RuntimeError("FEED_URLS is empty")

    state = load_state()
    os.makedirs(FEEDS_DIR, exist_ok=True)

    feeds = []
    for name_hint, raw_url in parse_feed_lines(FEED_URLS_RAW):
        url = normalize_url(raw_url)
        name, slug = derive_name_and_slug(name_hint, url)
        feeds.append({"name": name, "slug": slug, "url": url})

    total_new = 0
    all_new_items = []

    for f in feeds:
        if total_new >= MAX_TOTAL_NEW_ITEMS:
            break

        try:
            parsed = fetch_feed(f["url"])
        except Exception as e:
            print(f"[WARN] fetch failed: {f['url']} -> {e}")
            continue

        out_path = os.path.join(FEEDS_DIR, f"{f['slug']}.xml")
        existing = read_existing_items(out_path)

        new_items = []
        for e in parsed.entries:
            if total_new >= MAX_TOTAL_NEW_ITEMS:
                break
            if len(new_items) >= MAX_NEW_ITEMS_PER_FEED:
                break

            uid = f"{f['slug']}::{entry_uid(e)}"
            if uid in state["seen"]:
                continue

            title_en = (getattr(e, "title", "") or "").strip()
            summary_raw = getattr(e, "summary", "") or getattr(e, "description", "") or ""
            # 输入适当截断，控费
            summary_en = clip(strip_html(summary_raw), 800)

            link = (getattr(e, "link", "") or "").strip()
            published_dt = parse_time(e)

            zh_title, zh_html = translate_to_zh_html(title_en, summary_en, state)
            out_title = zh_title or title_en

            item = {
                "id": uid,
                "title": out_title,
                "html": zh_html,
                "link": link,
                "published": published_dt,
                "source": f["name"],
            }

            new_items.append(item)
            all_new_items.append(item)

            state["seen"][uid] = int(time.time())
            total_new += 1

            time.sleep(0.35)

        merged = new_items + existing
        dedup = {it["id"]: it for it in merged}
        final_items = sorted(dedup.values(), key=lambda x: x["published"], reverse=True)[:KEEP_ITEMS_PER_FEED]

        write_feed(
            out_path,
            title=f"ZH - {f['name']}",
            desc=f"Auto summarized/rewritten feed for {f['name']} (中文)",
            items=final_items,
        )
        print(f"[OK] {f['name']}: new={len(new_items)} -> {out_path}")

    # 聚合 all.xml：标题加来源前缀
    existing_all = read_existing_items(ALL_PATH)
    for it in all_new_items:
        it["title"] = f"[{it['source']}] {it['title']}"

    merged_all = all_new_items + existing_all
    dedup_all = {it["id"]: it for it in merged_all}
    final_all = sorted(dedup_all.values(), key=lambda x: x["published"], reverse=True)[:KEEP_ITEMS_PER_FEED]

    write_feed(
        ALL_PATH,
        title="ZH RSS - All",
        desc="Auto summarized/rewritten aggregated feed (中文)",
        items=final_all,
    )

    # index.html
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for f in feeds:
        rows.append(f'<li><a href="feeds/{f["slug"]}.xml">{html.escape(f["name"])}</a></li>')
    index = f"""<!doctype html>
<html><meta charset="utf-8"><title>ZH RSS Feeds</title>
<body>
<h1>ZH RSS Feeds</h1>
<p><a href="all.xml">聚合 all.xml</a></p>
<ul>
{''.join(rows)}
</ul>
</body></html>"""
    with open(INDEX_PATH, "w", encoding="utf-8") as w:
        w.write(index)

    save_state(state)
    print(f"[DONE] total_new={total_new} all={ALL_PATH}")

if __name__ == "__main__":
    main()
