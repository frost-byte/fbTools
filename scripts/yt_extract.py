#!/usr/bin/env python3
"""
yt_extract.py — YouTube video text extractor for LLM analysis.

Pulls metadata, transcript, and top comments from a YouTube video and
assembles them into a single clean markdown document ready to paste into
an LLM chat session.

Usage:
    python scripts/yt_extract.py <id-or-url> [options]
    python scripts/yt_extract.py dQw4w9WgXcQ --out rick.md --max-comments 30

    # One-time login to save YouTube session cookies (needed for transcripts):
    python scripts/yt_extract.py --login
    python scripts/yt_extract.py --login --browser /snap/brave/current/opt/brave.com/brave/brave

Requirements:
    pip install youtube-transcript-api
    pip install -e ".[scripts]"   # adds playwright (for --login)
    YOUTUBE_API_KEY env var (or .env file) — only needed for metadata + comments.

Cookie file (saved by --login):
    ~/.config/yt_extract/cookies.txt  (Netscape format, read by youtube-transcript-api)

Browser auto-detection order (for --login):
    1. --browser <path>  if supplied
    2. /snap/brave/current/opt/brave.com/brave/brave  (Brave snap)
    3. /usr/bin/brave-browser
    4. /usr/bin/brave
    5. Playwright's own installed Chromium (run: playwright install chromium)

Note: snap-confined browsers may refuse some Playwright flags. If --login fails
with a snap browser, run `playwright install chromium` and retry without --browser.

For personal research use only. See docs/youtube_extractor_action_plan.md §11.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


# ── .env loading ──────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    for candidate in [Path.cwd() / ".env", Path(__file__).parent.parent / ".env"]:
        if candidate.exists():
            with open(candidate) as fh:
                for line in fh:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    if key not in os.environ:
                        os.environ[key] = val
            break


# ── video ID extraction ────────────────────────────────────────────────────────

_VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")


def extract_video_id(raw: str) -> str:
    """Return the 11-char video ID from any YouTube URL form or a bare ID."""
    raw = raw.strip()
    parsed = urllib.parse.urlparse(raw)
    if parsed.scheme in ("http", "https"):
        host = parsed.netloc.lower().replace("www.", "")
        if host == "youtu.be":
            vid = parsed.path.lstrip("/").split("/")[0]
        elif "youtube.com" in host:
            qs = urllib.parse.parse_qs(parsed.query)
            vid = qs.get("v", [""])[0]
        else:
            raise ValueError(f"Unrecognised YouTube URL: {raw}")
        vid = vid.split("&")[0].split("?")[0]
    else:
        vid = raw
    if not _VIDEO_ID_RE.match(vid):
        raise ValueError(
            f"Could not extract a valid YouTube video ID from {raw!r}.\n"
            f"Got {vid!r} — expected 11 alphanumeric / dash / underscore characters."
        )
    return vid


# ── YouTube Data API v3 ───────────────────────────────────────────────────────

_API_BASE = "https://www.googleapis.com/youtube/v3"


def _api_get(endpoint: str, params: dict, api_key: str) -> dict:
    params = dict(params, key=api_key)
    url = f"{_API_BASE}/{endpoint}?" + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=20) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        try:
            err  = json.loads(body)
            code = err.get("error", {}).get("code", exc.code)
            msg  = err.get("error", {}).get("message", body)
        except Exception:
            code, msg = exc.code, body
        if code == 403 and "quota" in msg.lower():
            sys.exit(
                "YouTube Data API daily quota exceeded.\n"
                "Quotas reset at midnight Pacific Time.\n"
                "Usage dashboard: https://console.cloud.google.com/apis/dashboard"
            )
        if code in (401, 403):
            sys.exit(f"YouTube API auth error (HTTP {code}): {msg}\nCheck your YOUTUBE_API_KEY.")
        if code == 404:
            sys.exit("Video not found (HTTP 404). It may be private, deleted, or the ID is wrong.")
        raise RuntimeError(f"YouTube API error (HTTP {code}): {msg}")


# ── metadata ──────────────────────────────────────────────────────────────────

def _parse_duration(iso: str) -> str:
    m = re.match(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return iso or "?"
    h, mn, s = (int(x or 0) for x in m.groups())
    return f"{h}:{mn:02d}:{s:02d}" if h else f"{mn}:{s:02d}"


def fetch_metadata(video_id: str, api_key: str) -> dict:
    data  = _api_get("videos", {"part": "snippet,statistics,contentDetails", "id": video_id}, api_key)
    items = data.get("items", [])
    if not items:
        sys.exit(f"Video {video_id!r} not found. It may be private, deleted, or the ID is wrong.")
    item    = items[0]
    snippet = item.get("snippet", {})
    stats   = item.get("statistics", {})
    details = item.get("contentDetails", {})
    return {
        "title":         snippet.get("title", "(untitled)"),
        "channel":       snippet.get("channelTitle", ""),
        "published":     (snippet.get("publishedAt") or "")[:10],
        "description":   snippet.get("description", ""),
        "tags":          snippet.get("tags") or [],
        "duration":      _parse_duration(details.get("duration", "")),
        "views":         stats.get("viewCount", "?"),
        "likes":         stats.get("likeCount", "?"),
        "comment_count": stats.get("commentCount", "?"),
        "video_id":      video_id,
    }


# ── comments ──────────────────────────────────────────────────────────────────

def fetch_comments(
    video_id: str,
    api_key: str,
    max_comments: int,
    include_replies: bool,
    order: str,
) -> list[dict] | None:
    """Return comments sorted by like count, or None if comments are disabled."""
    comments:   list[dict] = []
    page_token: str | None = None
    fetch_target = max(max_comments * 3, 200)  # over-fetch, then re-sort + trim

    while len(comments) < fetch_target:
        params: dict = {
            "part":       "snippet",
            "videoId":    video_id,
            "maxResults": 100,
            "order":      order,
        }
        if page_token:
            params["pageToken"] = page_token

        try:
            data = _api_get("commentThreads", params, api_key)
        except RuntimeError as exc:
            msg = str(exc)
            if "commentsDisabled" in msg or ("403" in msg and "disabled" in msg.lower()):
                return None
            raise

        for item in data.get("items", []):
            top = item["snippet"]["topLevelComment"]["snippet"]
            entry: dict = {
                "author":     top.get("authorDisplayName", ""),
                "text":       top.get("textDisplay", "").strip(),
                "likes":      int(top.get("likeCount", 0)),
                "reply_count": item["snippet"].get("totalReplyCount", 0),
                "reply_list": [],
            }
            if include_replies and entry["reply_count"] > 0:
                try:
                    rdata = _api_get("comments", {
                        "part": "snippet", "parentId": item["id"], "maxResults": 20,
                    }, api_key)
                    entry["reply_list"] = [
                        {
                            "author": r["snippet"].get("authorDisplayName", ""),
                            "text":   r["snippet"].get("textDisplay", "").strip(),
                            "likes":  int(r["snippet"].get("likeCount", 0)),
                        }
                        for r in rdata.get("items", [])
                    ]
                except Exception:
                    pass
            comments.append(entry)

        page_token = data.get("nextPageToken")
        if not page_token:
            break
        time.sleep(0.15)

    # Re-sort by like count — more stable signal than API "relevance"
    comments.sort(key=lambda c: c["likes"], reverse=True)
    return comments[:max_comments]


# ── transcript ────────────────────────────────────────────────────────────────

_ARTIFACTS = re.compile(
    r"\[(?:Music|Applause|Laughter|Music\s+playing|Inaudible|__)\]",
    re.IGNORECASE,
)

_DEFAULT_COOKIE_FILE = Path.home() / ".config" / "yt_extract" / "cookies.txt"


def fetch_transcript(
    video_id: str,
    timestamps: bool,
    cookie_file: Path | None = None,
) -> str | None:
    """Fetch and assemble transcript. Returns None if unavailable."""
    try:
        from youtube_transcript_api import (  # type: ignore[import]
            NoTranscriptFound,
            TranscriptsDisabled,
            YouTubeTranscriptApi,
        )
    except ImportError:
        return (
            "[youtube-transcript-api not installed — "
            "run: pip install youtube-transcript-api]"
        )

    try:
        # Build an http_client session with cookies if available (API 1.x)
        http_client = None
        if cookie_file and cookie_file.exists():
            try:
                import http.cookiejar
                import requests as _requests
                session = _requests.Session()
                cj = http.cookiejar.MozillaCookieJar(str(cookie_file))
                cj.load(ignore_discard=True, ignore_expires=True)
                session.cookies = cj  # type: ignore[assignment]
                http_client = session
            except Exception:
                pass  # proceed without cookies if anything goes wrong

        fetcher_kwargs = {"http_client": http_client} if http_client is not None else {}
        fetcher = YouTubeTranscriptApi(**fetcher_kwargs)
        transcript_list = fetcher.list(video_id)
        try:
            t = transcript_list.find_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            t = next(iter(transcript_list))
        segments = t.fetch()
    except TranscriptsDisabled:
        return None
    except NoTranscriptFound:
        return None
    except Exception as exc:
        return f"[Transcript fetch failed: {exc}]"

    return _assemble_transcript(segments, timestamps)


def _assemble_transcript(segments: list, timestamps: bool) -> str:
    """
    Group caption segments into prose paragraphs.

    Paragraph boundaries are placed where the gap between the end of one
    segment and the start of the next is ≥ 1.5 s (a natural pause). A
    section header is inserted every 10 minutes of video time to help orient
    both reader and LLM in longer videos.
    """
    PAUSE_GAP        = 1.5   # seconds
    SECTION_INTERVAL = 600   # seconds between ### headers

    def _seg_get(seg: object, key: str, default: object = 0) -> object:
        """Access segment field whether the API returned dicts or objects."""
        if isinstance(seg, dict):
            return seg.get(key, default)
        return getattr(seg, key, default)

    # Clean and deduplicate segments
    cleaned: list[dict] = []
    for seg in segments:
        text = _ARTIFACTS.sub("", str(_seg_get(seg, "text", ""))).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        if cleaned and cleaned[-1]["text"].lower() == text.lower():
            continue
        cleaned.append({
            "text":     text,
            "start":    float(_seg_get(seg, "start", 0)),
            "duration": float(_seg_get(seg, "duration", 0)),
        })

    if not cleaned:
        return "(empty transcript)"

    # Group into paragraphs at pause boundaries
    paragraphs: list[tuple[float, str]] = []
    p_start  = cleaned[0]["start"]
    p_texts: list[str] = []

    for i, seg in enumerate(cleaned):
        p_texts.append(seg["text"])
        next_seg   = cleaned[i + 1] if i + 1 < len(cleaned) else None
        seg_end    = seg["start"] + seg["duration"]
        gap        = (next_seg["start"] - seg_end) if next_seg else 99.0
        if gap >= PAUSE_GAP or next_seg is None:
            paragraphs.append((p_start, " ".join(p_texts)))
            p_start = next_seg["start"] if next_seg else 0.0
            p_texts = []

    # Render
    lines:        list[str] = []
    last_section: float     = -SECTION_INTERVAL

    for start, text in paragraphs:
        # Section header every 10 minutes
        if start - last_section >= SECTION_INTERVAL:
            total_s = int(start)
            h, rem  = divmod(total_s, 3600)
            mn      = rem // 60
            label   = f"{h}:{mn:02d}" if h else f"{mn:02d}:00"
            lines.append(f"\n### ~[{label}]\n")
            last_section = start

        if timestamps:
            total_s   = int(start)
            h, rem    = divmod(total_s, 3600)
            mn, s     = divmod(rem, 60)
            ts        = f"[{h}:{mn:02d}:{s:02d}]" if h else f"[{mn:02d}:{s:02d}]"
            lines.append(f"{ts} {text}")
        else:
            lines.append(text)

    return "\n\n".join(lines)


# ── document assembly ─────────────────────────────────────────────────────────

def _fmt(n: object) -> str:
    try:
        return f"{int(n):,}"
    except (ValueError, TypeError):
        return str(n)


def build_document(
    meta:                 dict,
    comments:             list[dict] | None,
    transcript:           str | None,
    max_transcript_chars: int,
    include_replies:      bool,
) -> str:
    vid   = meta["video_id"]
    parts: list[str] = []

    # ── Header ────────────────────────────────────────────────────────────────
    tags_line = f"Tags: {', '.join(meta['tags'])}\n" if meta["tags"] else ""
    parts.append(
        f"# {meta['title']}\n"
        f"Channel: {meta['channel']} · Published: {meta['published']} · Duration: {meta['duration']}\n"
        f"URL: https://youtu.be/{vid}\n"
        f"Views: {_fmt(meta['views'])} · Likes: {_fmt(meta['likes'])} · "
        f"Comments: {_fmt(meta['comment_count'])}\n"
        + tags_line
    )

    # ── Description ───────────────────────────────────────────────────────────
    desc = (meta["description"] or "").strip()
    parts.append(f"## Description\n\n{desc}" if desc else "## Description\n\n_(none)_")

    # ── Transcript ────────────────────────────────────────────────────────────
    if transcript is None:
        parts.append("## Transcript\n\n_(unavailable — captions disabled or none published)_")
    else:
        body      = transcript
        truncated = False
        if max_transcript_chars and len(body) > max_transcript_chars:
            body      = body[:max_transcript_chars]
            truncated = True
        note = f"\n\n_[transcript truncated at ~{max_transcript_chars:,} chars]_" if truncated else ""
        parts.append(f"## Transcript\n\n{body}{note}")

    # ── Comments ──────────────────────────────────────────────────────────────
    if comments is None:
        parts.append("## Comments\n\n_(disabled on this video)_")
    elif not comments:
        parts.append("## Comments\n\n_(none retrieved)_")
    else:
        total = _fmt(meta["comment_count"])
        lines = [f"## Top Comments ({len(comments)} of {total}, sorted by likes)\n"]
        for c in comments:
            lines.append(f"- [{c['likes']:,}👍] **{c['author']}**: {c['text']}")
            if include_replies:
                for r in c.get("reply_list") or []:
                    lines.append(f"  - [{r['likes']:,}👍] **{r['author']}**: {r['text']}")
        parts.append("\n".join(lines))

    doc   = "\n\n---\n\n".join(parts)
    chars = len(doc)
    doc  += f"\n\n---\n\n_Document: ~{chars:,} chars / ~{chars // 4:,} estimated tokens_\n"
    return doc


# ── Playwright login ──────────────────────────────────────────────────────────

_BRAVE_SNAP = "/snap/brave/current/opt/brave.com/brave/brave"
_BRAVE_CANDIDATES = [
    _BRAVE_SNAP,
    "/usr/bin/brave-browser",
    "/usr/bin/brave",
]


def _find_browser() -> str | None:
    """Return the first browser executable found on disk, or None."""
    for path in _BRAVE_CANDIDATES:
        if Path(path).exists():
            return path
    return None


def _write_netscape_cookies(path: Path, cookies: list[dict]) -> None:
    """Write Playwright cookie dicts to a Netscape/Mozilla cookies.txt file."""
    lines = [
        "# Netscape HTTP Cookie File",
        "# Saved by yt_extract.py --login",
        "# https://curl.se/docs/http-cookies.html",
        "",
    ]
    for c in cookies:
        domain   = c.get("domain", "")
        flag     = "TRUE" if domain.startswith(".") else "FALSE"
        path_val = c.get("path", "/")
        secure   = "TRUE" if c.get("secure") else "FALSE"
        expires  = int(c.get("expires") or 0)
        if expires < 0:
            expires = 0
        name  = c.get("name", "")
        value = c.get("value", "")
        lines.append(f"{domain}\t{flag}\t{path_val}\t{secure}\t{expires}\t{name}\t{value}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def do_login(cookie_file: Path, browser_path: str | None) -> None:
    """Open a headed browser, let the user log in, save cookies."""
    try:
        from playwright.sync_api import sync_playwright  # type: ignore[import]
    except ImportError:
        sys.exit(
            "playwright is not installed.\n"
            "Install it with:  pip install -e '.[scripts]'\n"
            "  or:             pip install playwright"
        )

    exe = browser_path or _find_browser()

    print("Opening browser for YouTube login...", file=sys.stderr)
    if exe:
        print(f"  Browser: {exe}", file=sys.stderr)
    else:
        print(
            "  No system browser found. Using Playwright's Chromium.\n"
            "  If this fails, run: playwright install chromium",
            file=sys.stderr,
        )

    with sync_playwright() as pw:
        launch_kwargs: dict = {"headless": False}
        if exe:
            launch_kwargs["executable_path"] = exe

        try:
            browser = pw.chromium.launch(**launch_kwargs)
        except Exception as exc:
            msg = str(exc)
            hint = (
                "\n\nIf you are using a snap-confined browser, it may block Playwright's flags.\n"
                "Try installing Playwright's own Chromium instead:\n"
                "  playwright install chromium\n"
                "Then re-run --login without --browser."
            )
            sys.exit(f"Could not launch browser: {msg}{hint}")

        context = browser.new_context()
        page    = context.new_page()
        page.goto("https://www.youtube.com")

        print("\nBrowser is open at youtube.com.", file=sys.stderr)
        print("Sign in to your Google account, then come back here and press Enter.", file=sys.stderr)
        try:
            input()
        except EOFError:
            pass

        cookies = context.cookies()
        browser.close()

    cookie_file.parent.mkdir(parents=True, exist_ok=True)
    _write_netscape_cookies(cookie_file, cookies)
    yt_cookies = [c for c in cookies if "youtube" in c.get("domain", "") or "google" in c.get("domain", "")]
    print(
        f"\nSaved {len(cookies)} cookies ({len(yt_cookies)} YouTube/Google) → {cookie_file}",
        file=sys.stderr,
    )
    print("Run yt_extract.py normally — transcripts will now use this session.", file=sys.stderr)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="yt_extract",
        description=(
            "Extract YouTube video metadata, transcript, and top comments into\n"
            "a single markdown document for LLM analysis."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # One-time login (saves cookies for transcript access):\n"
            "  python scripts/yt_extract.py --login\n\n"
            "  # Extract a video:\n"
            "  python scripts/yt_extract.py https://youtu.be/dQw4w9WgXcQ --out rick.md\n\n"
            "Environment:\n"
            "  YOUTUBE_API_KEY   Required for metadata and comments.\n"
            "                    Set in the environment or a .env file.\n\n"
            "Cookie file (created by --login):\n"
            f"  {_DEFAULT_COOKIE_FILE}\n\n"
            "For personal research use only. See docs/youtube_extractor_action_plan.md §11."
        ),
    )

    parser.add_argument(
        "url", nargs="?",
        help="YouTube video ID or URL (any form). Omit when using --login.",
    )
    parser.add_argument(
        "--login", action="store_true",
        help=(
            "Open a browser, let you sign into YouTube, and save session cookies. "
            f"Cookies are stored at {_DEFAULT_COOKIE_FILE} and used automatically "
            "on subsequent runs to bypass transcript IP bans."
        ),
    )
    parser.add_argument(
        "--browser", metavar="PATH",
        help=(
            "Path to the browser executable for --login "
            f"(default: auto-detect Brave at {_BRAVE_SNAP}, then Playwright Chromium)."
        ),
    )
    parser.add_argument(
        "--cookie-file", metavar="FILE", default=str(_DEFAULT_COOKIE_FILE),
        help=f"Cookie file path (default: {_DEFAULT_COOKIE_FILE})",
    )
    parser.add_argument("--out", metavar="FILE",
                        help="Write output to FILE instead of stdout")
    parser.add_argument("--max-comments", type=int, default=50, metavar="N",
                        help="Top N comments to include after like-count sort (default: 50)")
    parser.add_argument("--max-transcript-chars", type=int, default=0, metavar="N",
                        help="Truncate transcript at N chars; 0 = no limit (default: 0)")
    parser.add_argument("--order", choices=["relevance", "time"], default="relevance",
                        help="Initial API fetch order before like-count re-sort (default: relevance)")
    parser.add_argument("--timestamps", action="store_true",
                        help="Prefix each transcript paragraph with a [MM:SS] timestamp")
    parser.add_argument("--include-replies", action="store_true",
                        help="Fetch and include reply threads under top comments")
    args = parser.parse_args()

    cookie_file = Path(args.cookie_file).expanduser()

    if args.login:
        do_login(cookie_file, args.browser)
        return

    if not args.url:
        parser.error("url is required unless --login is used")

    _load_dotenv()

    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        sys.exit(
            "YOUTUBE_API_KEY is not set.\n"
            "Add it to your environment or create a .env file:\n"
            "  YOUTUBE_API_KEY=your_key_here"
        )

    try:
        video_id = extract_video_id(args.url)
    except ValueError as exc:
        sys.exit(str(exc))

    print(f"Fetching: https://youtu.be/{video_id}", file=sys.stderr)

    print("  → metadata ...", file=sys.stderr)
    meta = fetch_metadata(video_id, api_key)
    print(f"     {meta['title']!r}", file=sys.stderr)

    print("  → comments ...", file=sys.stderr)
    try:
        comments = fetch_comments(
            video_id, api_key, args.max_comments, args.include_replies, args.order
        )
        status = "(disabled)" if comments is None else f"{len(comments)} fetched"
        print(f"     {status}", file=sys.stderr)
    except Exception as exc:
        print(f"     WARNING: comments failed — {exc}", file=sys.stderr)
        comments = []

    cookie_note = f" (cookies: {cookie_file})" if cookie_file.exists() else " (no cookie file — run --login for transcripts)"
    print(f"  → transcript ...{cookie_note}", file=sys.stderr)
    try:
        transcript = fetch_transcript(video_id, args.timestamps, cookie_file)
        status = "(unavailable)" if transcript is None else f"{len(transcript):,} chars"
        print(f"     {status}", file=sys.stderr)
    except Exception as exc:
        print(f"     WARNING: transcript failed — {exc}", file=sys.stderr)
        transcript = f"[Transcript fetch error: {exc}]"

    doc = build_document(
        meta=meta,
        comments=comments,
        transcript=transcript,
        max_transcript_chars=args.max_transcript_chars,
        include_replies=args.include_replies,
    )

    if args.out:
        Path(args.out).write_text(doc, encoding="utf-8")
        chars = len(doc)
        print(f"\nWritten to: {args.out}  (~{chars:,} chars / ~{chars // 4:,} tokens)", file=sys.stderr)
    else:
        sys.stdout.write(doc)


if __name__ == "__main__":
    main()
