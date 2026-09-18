# Action Plan — YouTube Video Extractor (for feeding video substance to an LLM)

**Location:** `scripts/yt_extract.py` — a standalone developer utility with no ComfyUI dependency.
Primary use: feed the output to an LLM (Claude, etc.) to help comprehend, plan around, or
summarize the video's content. Not integrated into fbTools nodes; run from the command line.

**Goal:** a small, reusable CLI/tool that takes a YouTube video ID or URL and produces a single
clean text/markdown document containing the video's **metadata + description + transcript +
selected comments**, formatted for pasting into an LLM chat for analysis.

Marker legend:
- **[INVARIANT]** — must hold.
- **[DECIDE] / [EXPLORE]** — judgment call; pick sensibly.
- **[VERIFY]** — confirm against current API/library behavior before relying on it.

---

## 1. Why this exists
Some source videos (e.g. local-LLM / agent-scaffolding walkthroughs) carry their real substance in
the **description and comments** (technical claims, corrections, caveats, alternative approaches) and
in the **spoken transcript** (methods, settings). Direct page-fetching is rate-limited/blocked, so the
tool pulls the text via APIs from the user's own environment, then formats it into one artifact the
user can hand to an LLM. The output — not the fetching — is the product.

---

## 2. What it produces (the output artifact — design this first)
A single UTF-8 markdown document, structured for LLM consumption, in this order:

```
# <video title>
Channel: <name> · Published: <date> · Duration: <hh:mm:ss>
URL: https://youtu.be/<id>
Views: <n> · Likes: <n> · Comments: <n>
Tags: <comma-separated, if present>

## Description
<full description text, verbatim>

## Transcript
<transcript text — see §5 for timestamp handling>

## Top Comments (<k> of <total>)
- [<likes>👍] <author>: <comment text>
- ...
```

Design constraints on the output:
- **[INVARIANT]** It is one self-contained document — no external references the LLM can't see.
- **[DECIDE] Length budget.** A long transcript + many comments can be large. Target a sensible cap
  (e.g. keep full description, full transcript up to N tokens/chars, and top K comments) and, if
  truncated, say so explicitly in the doc (`[transcript truncated at ~X tokens]`) rather than
  silently cutting. Make the cap a flag (`--max-comments`, `--max-transcript-chars`).
- Strip boilerplate from descriptions where obvious (affiliate link spam, repeated promo blocks) —
  **[EXPLORE]** a light heuristic, but never drop technical content; when unsure, keep it.
- Keep comments **verbatim** (they often contain the corrections/caveats that matter); don't summarize
  them in the tool — that's the LLM's job downstream.

---

## 3. Input handling
- Accept either a raw video ID or any YouTube URL form (`youtu.be/<id>`, `youtube.com/watch?v=<id>`,
  with extra query params, `&t=`, playlists, etc.). Parse out the 11-char video ID robustly.
- **[INVARIANT]** Validate the extracted ID (11 chars, expected charset) and fail with a clear message
  rather than calling the API with garbage.

---

## 4. Metadata + comments — YouTube Data API v3 (official)
Requires a Google Cloud project with **YouTube Data API v3 enabled** and an API key.
**This is the only API key the tool needs** — the transcript path (§5) uses an unofficial
library that requires no key. Read the key from env (`YOUTUBE_API_KEY`) or a `.env` file;
never hardcode it. Default quota is 10,000 units/day; the calls below are ~1 unit each, so
occasional personal use never approaches the limit.

- **Metadata:** `videos.list` with `part=snippet,statistics,contentDetails&id=<id>`.
  - snippet → title, description, channelTitle, publishedAt, tags
  - statistics → viewCount, likeCount, commentCount
  - contentDetails → duration (ISO 8601, e.g. `PT12M30S` — convert to hh:mm:ss)
- **Comments:** `commentThreads.list` with `part=snippet&videoId=<id>&maxResults=100&order=relevance`.
  - The goal is **salient community observations** — corrections, caveats, notable context —
    while preserving the ability to refer back to individual comments directly (verbatim text +
    author + like count).
  - Fetch with `order=relevance`, then **re-sort client-side by `likeCount` descending** before
    applying `--max-comments`. YouTube's "relevance" sort is opaque and changes over time;
    like count is a more stable proxy for "what the community found notable or correct".
    **[VERIFY]** that `likeCount` is available per-comment in the `snippet` response (it is in
    current API versions, but confirm before relying on it).
  - Keep comments **verbatim** with author and like count — summarization is the downstream
    LLM's job, not the tool's.
  - Pull top-level comments only; replies are usually noise — expose `--include-replies` for
    the cases where a thread contains a correction or extended discussion.
  - **[INVARIANT]** Read the API key from env (`YOUTUBE_API_KEY`) or a `.env`, never hardcode it.

---

## 5. Transcript — `youtube-transcript-api` (unofficial)
The official Data API `captions` endpoint only serves captions for videos you **own**, so it can't
fetch an arbitrary video's transcript. Use the `youtube-transcript-api` Python library instead
(**no API key needed**; scrapes the caption track — the only dependency-free path for third-party
videos).

- Call the transcript fetch for the video ID, preferring English (`languages=['en', ...]`), falling
  back to the video's default/auto-generated track if English isn't available.
- Returns a list of segments `{text, start, duration}`.
- **[DECIDE] Transcript assembly — prose vs timestamped.** Auto-generated captions are raw segments
  with no punctuation or paragraph structure. A naive join produces a wall of text. Two modes:

  - *Clean prose* (default): group segments into paragraphs using pause gaps as natural break
    points (a gap of ≥ 1.5 s between segments is a reasonable paragraph boundary; adjust if it
    over-fragments). Within each paragraph, join segment text with a space. Strip common caption
    artifacts: `[Music]`, `[Applause]`, `[Laughter]`, duplicated auto-caption lines. This produces
    text that is readable as prose and digestible for an LLM — better than a single flat wall, and
    better than one line per caption segment.

  - *Timestamped* (flag: `--timestamps`): insert a coarse `[MM:SS]` marker at each paragraph
    break (derived from the first segment in the group). Useful when you want the LLM to reason
    about *where* in the video something was said, or to give you a timestamp to look up.

  For very long videos consider also inserting a `### ~[HH:MM]` section header every 10 minutes
  of content so the LLM can orient itself in a long doc.

- **Dedup/cleanup:** before assembling, deduplicate consecutive segments with identical or near-
  identical text (auto-caption stutter), collapse repeated whitespace.
- **[INVARIANT]** Handle "no transcript" gracefully — captions disabled, or none exist. Emit the doc
  with `## Transcript\n_(unavailable — captions disabled or none published)_` rather than
  failing the whole run. Metadata + comments are still valuable without it.
- **[VERIFY]** This library is unofficial and periodically breaks when YouTube changes internals; pin a
  known-good version and fail with a clear, actionable message (not a stack trace) if the fetch breaks.

---

## 6. Error handling & edges [INVARIANT for each]
- **Comments disabled** on the video → emit `## Comments\n_(disabled)_`, continue.
- **Transcript unavailable** → note it, continue (see §5).
- **Video not found / private / deleted** → clear message, exit non-zero.
- **Quota exceeded** (Data API 403) → distinguish from other errors; tell the user it's a daily-quota
  issue, not a bug.
- **Rate/transient errors** → one or two retries with backoff, then fail clearly.
- Any single component failing should degrade to a partial document with an explicit note, not a total
  failure — the tool's value is assembling whatever text is available.

---

## 7. CLI shape (suggested)
```
yt-extract <id-or-url> [--out FILE] [--max-comments N] [--max-transcript-chars N]
                       [--order relevance|time] [--timestamps] [--include-replies]
```
- Default: write to stdout (so it pipes) and/or a `.md` file the user can open and copy.
- **[EXPLORE]** a `--json` mode emitting structured data instead of the formatted markdown, for
  programmatic reuse — but the markdown doc is the primary deliverable.

---

## 8. Wrinkles / considerations
- **[DECIDE] Token/length awareness** — if this feeds an LLM chat, the assembled doc shouldn't be
  enormous. Consider reporting an approximate token/char count at the top so the user knows before
  pasting, and honor the caps in §2.
- **Language** — non-English videos: pull the native transcript and note the language; don't attempt
  translation in the tool (leave that to the LLM).
- **Dedup/cleanup** — collapse repeated whitespace and caption artifacts (`[Music]`, `[Applause]`,
  duplicated auto-caption lines) in the transcript for readability; keep it light.
- **ToS note** — see §11 below.
- **Reusability** — this is a general "extract text substance from a video" tool, not specific to one
  video; keep the ID/URL as an argument so it works for anything the user is learning from.
- **Secrets** — `.env` / env var for the key; add `.env` to `.gitignore`.

---

## 9. Suggested sequence
1. Input parsing (ID/URL → validated video ID).
2. Data API metadata (`videos.list`) → header block. Verify the key/quota flow end-to-end first.
3. Data API comments (`commentThreads.list`) with pagination + cap.
4. Transcript via `youtube-transcript-api` with graceful fallback.
5. Assembler: compose the §2 markdown, apply caps + truncation notes, report approx length.
6. CLI flags + error/edge handling per §6.

## 10. Definition of done
- Given a video ID or any URL form, produces one clean markdown doc with metadata, full description,
  transcript (or an explicit unavailable note), and top-K comments (or a disabled note).
- Degrades to a partial doc — never a hard crash — when transcript or comments are missing.
- Caps are honored and any truncation is stated in the output.
- Key is read from env; nothing sensitive is hardcoded or committed.
- Output is immediately paste-ready into an LLM chat (self-contained, readable, length-aware).

---

## 11. Terms of Service / Disclaimer

**This tool is written for personal research use by a single user.**

| Component | Path | Status |
|---|---|---|
| YouTube Data API v3 | Metadata + comments | **Official.** Governed by the [YouTube API Services Terms of Service](https://developers.google.com/youtube/terms/api-services-tos) and [Google API ToS](https://developers.google.com/terms/). Personal, non-commercial, non-automated use well within permitted scope. |
| `youtube-transcript-api` | Transcript | **Unofficial / scraping-based.** YouTube's ToS (Section 5.H) prohibits scraping without written permission. The library is widely used for personal research; enforcement against individual non-commercial users is not documented. The transcript path is the fragile/gray one — which is why it degrades gracefully. |

**For any use beyond the author's own personal research** (sharing the tool, integrating it into a
service, running it at scale, or distributing output commercially), review:
- YouTube Terms of Service §5.H (scraping prohibition)
- YouTube API Services Terms §III.E (no circumventing access controls)
- The `youtube-transcript-api` library's own README re: terms

The output of this tool (the markdown document) is derived from YouTube content and subject to
YouTube's content policies when shared or published. For personal LLM analysis sessions, this is
not a concern.
