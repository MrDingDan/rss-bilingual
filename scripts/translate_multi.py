import os, re, json, time, hashlib, html
from urllib.parse import urlparse
from datetime import datetime, timezone
from dateutil import parser as dateparser

import requests
import feedparser
from feedgen.feed import FeedGenerator

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

# rsshub:// 简写别名（你写了 rsshub://hackernews）
RSSHUB_ALIAS = {
    "hackernews": "hn/newest",  # 如果你发现不通，直接把那行改成 https://hnrss.org/newest 也行
}

def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"seen": {}, "translations": {}}

def save_state(state):
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def strip_html(text: str) -> str:
    text = text or ""
    text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.S|re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.S|re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def clip(text: str, n: int = 800) -> str:
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rstrip() + "…"

def normalize_url(u: str) -> str:
    u = (u or "").strip()
    # 去掉末尾多余的 ?
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

    # reddit 子版块
    m = re.search(r"reddit\.com/r/([^/]+)/?\.rss", url, re.I)
    if m:
        n = f"reddit/{m.group(1)}"
        return n, slugify(f"reddit-{m.group(1)}")

    # xgo.ing
    if "xgo.ing" in host:
        last = path.split("/")[-1] if path else "xgo"
        n = f"xgo/{last[:12]}"
        return n, slugify(f"xgo-{last[:12]}")

    # rsshub 常见
    if "rsshub" in host:
        n = f"rsshub/{path}"
        return n, slugify(path)

    # 兜底：host+path
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

def chat_completions_urls(base: str) -> list[str]:
    base = (base or "").rstrip("/")
    urls = []

    # 1) base 本身就是 .../v1
    if base.endswith("/v1"):
        urls.append(base + "/chat/completions")
    else:
        # 2) OpenAI/多数兼容：.../v1/chat/completions
        urls.append(base + "/v1/chat/completions")
        # 3) DeepSeek 这类：.../chat/completions
        urls.append(base + "/chat/completions")

    # 去重保持顺序
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
            {"role": "system", "content": "你是一个高质量的中英双语RSS翻译助手。输出必须严格按用户要求的JSON格式。"},
            {"role": "user", "content": prompt},
        ],
    }

    last_err = None
    for url in chat_completions_urls(AI_BASE_URL):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
            # 404：换下一个 endpoint 继续试
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


def translate_bilingual(title_en: str, summary_en: str, state) -> tuple[str, str]:
    title_en = (title_en or "").strip()
    summary_en = (summary_en or "").strip()

    # 缓存 key：避免重复花钱
    cache_key = hashlib.sha256((title_en + "\n" + summary_en).encode("utf-8")).hexdigest()
    if cache_key in state["translations"]:
        t = state["translations"][cache_key]
        return t["zh_title"], t["zh_summary"]

    prompt = f"""
把下面的RSS条目翻译成中文（简体），要求：
1) 中文自然、简洁，不要机翻腔
2) 保留专有名词（人名/公司名/产品名）可括号补充英文
3) 不要添加原文没有的信息
4) 输出严格为JSON，键只有 zh_title 和 zh_summary

英文标题：
{title_en}

英文摘要/正文片段：
{summary_en}
""".strip()

    content = call_llm(prompt)

    zh_title, zh_summary = "", ""
    try:
        start = content.find("{")
        end = content.rfind("}")
        obj = json.loads(content[start:end + 1])
        zh_title = (obj.get("zh_title") or "").strip()
        zh_summary = (obj.get("zh_summary") or "").strip()
    except Exception:
        zh_summary = content.strip()

    state["translations"][cache_key] = {"zh_title": zh_title, "zh_summary": zh_summary}
    return zh_title, zh_summary

def make_bilingual_title(zh: str, en: str) -> str:
    zh, en = (zh or "").strip(), (en or "").strip()
    if zh and en:
        return f"{zh} / {en}"
    return zh or en

def make_bilingual_summary(zh: str, en: str) -> str:
    zh, en = (zh or "").strip(), (en or "").strip()
    if zh and en:
        return f"{zh}\n\n---\n\nOriginal:\n{en}"
    return zh or en

def read_existing_items(path: str):
    if not os.path.exists(path):
        return []
    d = feedparser.parse(path)
    items = []
    for e in d.entries:
        items.append({
            "id": e.get("id") or e.get("guid") or e.get("link") or hashlib.sha256((e.get("title","")+e.get("link","")).encode("utf-8")).hexdigest(),
            "title": e.get("title", ""),
            "summary": e.get("summary", ""),
            "link": e.get("link", ""),
            "published": parse_time(e),
        })
    return items

def write_feed(path: str, title: str, desc: str, items: list[dict]):
    fg = FeedGenerator()
    fg.id(path)
    fg.title(title)
    fg.link(href="https://github.com/", rel="alternate")
    fg.description(desc)

    for it in items:
        fe = fg.add_entry()
        fe.id(it["id"])
        fe.title(it["title"])
        fe.link(href=it["link"])
        fe.description(it["summary"])
        fe.pubDate(it["published"])

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fg.rss_file(path, pretty=True)

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

        # 单源输出路径
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
            summary_en = clip(strip_html(summary_raw), 700)

            link = (getattr(e, "link", "") or "").strip()
            published_dt = parse_time(e)

            zh_title, zh_summary = translate_bilingual(title_en, summary_en, state)

            out_title = make_bilingual_title(zh_title, title_en)
            out_summary = make_bilingual_summary(clip(zh_summary, 900), summary_en)

            item = {
                "id": uid,
                "title": out_title,
                "summary": out_summary,
                "link": link,
                "published": published_dt,
                "source": f["name"],
            }

            new_items.append(item)
            all_new_items.append(item)

            state["seen"][uid] = int(time.time())
            total_new += 1

            time.sleep(0.35)  # 轻微节流

        # 合并：新条目 + 旧条目（去重，保留最近 KEEP_ITEMS_PER_FEED 条）
        merged = new_items + existing
        dedup = {}
        for it in merged:
            dedup[it["id"]] = it
        final_items = sorted(dedup.values(), key=lambda x: x["published"], reverse=True)[:KEEP_ITEMS_PER_FEED]

        write_feed(
            out_path,
            title=f"Bilingual - {f['name']}",
            desc=f"Auto translated feed for {f['name']} (中文 + 原文)",
            items=final_items,
        )

        print(f"[OK] {f['name']}: new={len(new_items)} -> {out_path}")

    # 聚合 all.xml：标题加来源前缀，方便你刷未读
    existing_all = read_existing_items(ALL_PATH)
    for it in all_new_items:
        it["title"] = f"[{it['source']}] {it['title']}"

    merged_all = all_new_items + existing_all
    dedup_all = {}
    for it in merged_all:
        dedup_all[it["id"]] = it
    final_all = sorted(dedup_all.values(), key=lambda x: x["published"], reverse=True)[:KEEP_ITEMS_PER_FEED]

    write_feed(
        ALL_PATH,
        title="Bilingual RSS - All (中文 + 原文)",
        desc="Auto translated aggregated feed (中文 + 原文)",
        items=final_all,
    )

    # index.html（可选）
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for f in feeds:
        rows.append(f'<li><a href="feeds/{f["slug"]}.xml">{f["name"]}</a></li>')
    index = f"""<!doctype html>
<html><meta charset="utf-8"><title>Bilingual RSS Feeds</title>
<body>
<h1>Bilingual RSS Feeds</h1>
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
