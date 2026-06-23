#!/usr/bin/env python3
"""
Offline transcription helper — tiny HTTP server.

Run:  python3 transcription_server.py
Open: http://localhost:8765   (opens automatically)

Serves audio clips from ./test_clips/ and saves transcriptions to
./transcriptions.json after every Save press.

No external dependencies — stdlib only.
"""

import csv
import html
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

PORT       = 8765
CLIPS_DIR  = Path("./test_clips")
CSV_PATH   = Path("./transcription_sheet.csv")
SAVE_PATH  = Path("./transcriptions.json")

SKIP_OPTIONS = ["Hindi", "music", "unclear", "other"]

# On-screen keyboard layout:
#   ("label", "text_to_insert", is_word)
#   is_word=True  → inserts text + space
#   is_word=False → inserts text as-is (characters/matras)
KB_ROWS = [
    [
        ("सै",     "सै",     True),
        ("कोन्या", "कोन्या", True),
        ("कोनी",   "कोनी",   True),
        ("म्ह",    "म्ह",    True),
        ("म्हारे", "म्हारे", True),
        ("घणी",    "घणी",    True),
        ("बखत",    "बखत",    True),
        ("गाम",    "गाम",    True),
    ],
    [
        ("इब",     "इब",     True),
        ("छोरा",   "छोरा",   True),
        ("छोरी",   "छोरी",   True),
        ("ताऊ",    "ताऊ",    True),
        ("मन्नै",  "मन्नै",  True),
        ("थारे",   "थारे",   True),
        ("आपणा",   "आपणा",   True),
        ("गेल्यां","गेल्यां",True),
    ],
    [
        ("ै",  "ै",  False),
        ("ण",  "ण",  False),
        ("्ह", "्ह", False),
        ("ख्", "ख्", False),
        ("ज्य","ज्य",False),
        ("व्", "व्", False),
        ("ड़", "ड़", False),
    ],
]


# ── data helpers ──────────────────────────────────────────────────────────────
def load_clips() -> list[dict]:
    if not CSV_PATH.exists():
        return []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        clips = []
        for r in reader:
            clips.append({
                "clip_file":    r["clip_file"],
                "duration_sec": float(r["duration_sec"]),
                "whisper_draft": r.get("whisper_draft", "").strip(),
            })
    return clips


def load_transcriptions() -> dict:
    if SAVE_PATH.exists():
        return json.loads(SAVE_PATH.read_text(encoding="utf-8"))
    return {}


def save_transcription(clip_file: str, transcribe_here: str, skip_reason: str) -> None:
    data = load_transcriptions()
    data[clip_file] = {
        "transcribe_here": transcribe_here,
        "skip_reason":     skip_reason,
    }
    SAVE_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ── keyboard HTML builder ─────────────────────────────────────────────────────
def _build_keyboard_html() -> str:
    rows_html = []
    for row in KB_ROWS:
        btns = []
        for label, text, is_word in row:
            # escape for JS string (single-quoted)
            js_text = text.replace("\\", "\\\\").replace("'", "\\'")
            insert  = f"'{js_text} '" if is_word else f"'{js_text}'"
            css_cls = "kb-word" if is_word else "kb-char"
            btns.append(
                f'<button class="kb-btn {css_cls}" '
                f'onmousedown="return false" '
                f'onclick="kbInsert({insert})">{html.escape(label)}</button>'
            )
        rows_html.append('<div class="kb-row">' + "".join(btns) + "</div>")
    return "\n".join(rows_html)


# ── HTML builder ──────────────────────────────────────────────────────────────
def _skip_options_html(current: str) -> str:
    lines = ['<option value="">-- skip reason --</option>']
    for opt in SKIP_OPTIONS:
        sel = " selected" if current == opt else ""
        lines.append(f'<option value="{opt}"{sel}>{opt}</option>')
    return "\n".join(lines)


def build_html(clips: list[dict], transcriptions: dict) -> str:
    clip_blocks = []
    for clip in clips:
        fn     = clip["clip_file"]
        dur    = clip["duration_sec"]
        draft  = clip["whisper_draft"]
        saved  = transcriptions.get(fn, {})
        trans  = saved.get("transcribe_here", "")
        skip   = saved.get("skip_reason", "")

        # Pre-fill: saved correction > whisper draft > empty
        textarea_val = trans if trans else draft
        status = "done" if (trans or skip) else "pending"

        clip_blocks.append(f"""
<div class="clip {status}" id="clip-{fn}" data-file="{fn}">
  <div class="clip-header">
    <span class="clip-name">{html.escape(fn)}</span>
    <span class="clip-dur">{dur:.1f}s</span>
  </div>
  <audio controls preload="none" src="/clips/{html.escape(fn)}"></audio>
  <div class="inputs">
    <textarea class="trans-input"
              placeholder="Bangru Devanagari text यहाँ लिखें…"
              rows="2">{html.escape(textarea_val)}</textarea>
    <div class="row2">
      <select class="skip-select">
        {_skip_options_html(skip)}
      </select>
      <button class="save-btn" onclick="saveClip('{fn}')">Save</button>
      <span class="save-status" id="status-{fn}"></span>
    </div>
  </div>
</div>""")

    clips_json   = json.dumps([c["clip_file"] for c in clips])
    keyboard_html = _build_keyboard_html()

    css = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Arial, sans-serif;
  background: #f0f2f5;
  padding: 20px 20px 160px;   /* bottom padding = keyboard height */
  max-width: 860px;
  margin: 0 auto;
}
h1 { color: #1a1a2e; margin-bottom: 4px; font-size: 1.3em; }
.subtitle { color: #555; font-size: 0.85em; margin-bottom: 14px; line-height: 1.5; }

/* progress */
.prog-wrap { background: #dde; border-radius: 6px; height: 18px; margin: 10px 0 4px; }
.prog-fill  { background: #4caf50; height: 100%; border-radius: 6px; transition: width .3s; }
.prog-text  { font-size: 0.82em; color: #555; margin-bottom: 14px; }

/* filters */
.filters { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 14px; }
.filter-btn {
  padding: 5px 14px; border: 1px solid #bbb; border-radius: 20px;
  cursor: pointer; background: white; font-size: 0.82em;
}
.filter-btn.active { background: #3f51b5; color: white; border-color: #3f51b5; }

/* clip card */
.clip {
  background: white; border: 2px solid #e0e0e0; border-radius: 10px;
  padding: 14px; margin-bottom: 10px;
}
.clip.done    { border-color: #4caf50; background: #f6fff6; }
.clip.pending { border-color: #ff9800; }
.clip-header  { display: flex; justify-content: space-between; margin-bottom: 6px; }
.clip-name    { font-weight: 700; color: #222; font-size: 0.9em; }
.clip-dur     { color: #888; font-size: 0.82em; }
audio         { width: 100%; margin: 6px 0 8px; }

/* inputs */
.inputs { display: flex; flex-direction: column; gap: 6px; }
textarea {
  width: 100%; font-size: 1.15em; font-family: inherit;
  border: 1px solid #ccc; border-radius: 6px; padding: 7px 10px;
  resize: vertical; line-height: 1.4;
}
textarea:focus { outline: 2px solid #3f51b5; border-color: transparent; }
.row2 { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
select {
  padding: 6px 10px; border: 1px solid #ccc; border-radius: 6px;
  font-size: 0.85em; background: white;
}
.save-btn {
  background: #3f51b5; color: white; border: none; padding: 7px 20px;
  border-radius: 6px; cursor: pointer; font-size: 0.88em; font-weight: 600;
}
.save-btn:hover { background: #303f9f; }
.save-status { font-size: 0.82em; color: #4caf50; }
.export-btn {
  display: inline-block; margin-top: 20px;
  background: #7b1fa2; color: white; border: none; padding: 9px 20px;
  border-radius: 6px; cursor: pointer; font-size: 0.88em; font-weight: 600;
}
.export-btn:hover { background: #6a1b9a; }
.kbd {
  display: inline-block; background: #eee; border: 1px solid #bbb;
  border-radius: 3px; padding: 1px 5px; font-size: 0.78em; font-family: monospace;
}

/* ── on-screen keyboard ── */
#keyboard {
  position: fixed;
  bottom: 0; left: 0; right: 0;
  background: #1a1a2e;
  border-top: 2px solid #3f51b5;
  padding: 8px 12px 10px;
  z-index: 200;
}
#keyboard .kb-label {
  color: #8899bb; font-size: 0.72em; margin-bottom: 4px;
}
.kb-row { display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 5px; }
.kb-row:last-child { margin-bottom: 0; }
.kb-btn {
  font-size: 1.0em; font-family: inherit;
  padding: 4px 10px; border-radius: 5px; border: none;
  cursor: pointer; user-select: none;
  transition: background 0.1s;
}
.kb-word {
  background: #3a3f6e; color: #e8eaf6;
}
.kb-word:hover  { background: #4f5a9e; }
.kb-word:active { background: #6573c3; }
.kb-char {
  background: #2d4a2d; color: #c8f0c8;
  font-size: 1.05em;
  min-width: 2.4em; text-align: center;
}
.kb-char:hover  { background: #3d6b3d; }
.kb-char:active { background: #4caf50; color: #000; }
#kb-hint {
  color: #556; font-size: 0.7em; margin-top: 3px;
  font-style: italic;
}
"""

    js = f"""
const allClips = {clips_json};

/* ── keyboard: track last focused textarea ── */
let activeTextarea = null;

document.querySelectorAll('.trans-input').forEach(ta => {{
  ta.addEventListener('focus', () => {{ activeTextarea = ta; }});
}});

function kbInsert(text) {{
  if (!activeTextarea) {{
    document.getElementById('kb-hint').textContent =
      '← Click a text box first, then press a key';
    return;
  }}
  document.getElementById('kb-hint').textContent = '';
  const ta    = activeTextarea;
  const start = ta.selectionStart;
  const end   = ta.selectionEnd;
  ta.value    = ta.value.substring(0, start) + text + ta.value.substring(end);
  ta.selectionStart = ta.selectionEnd = start + text.length;
  ta.focus();
}}

/* ── progress ── */
function updateProgress() {{
  const done  = document.querySelectorAll('.clip.done').length;
  const total = allClips.length;
  const pct   = total > 0 ? Math.round(done / total * 100) : 0;
  document.getElementById('prog-fill').style.width = pct + '%';
  document.getElementById('prog-text').textContent =
    done + ' / ' + total + ' clips done (' + pct + '%)';
}}

/* ── save ── */
async function saveClip(fn) {{
  const clipEl   = document.getElementById('clip-' + fn);
  const trans    = clipEl.querySelector('.trans-input').value.trim();
  const skip     = clipEl.querySelector('.skip-select').value;
  const statusEl = document.getElementById('status-' + fn);

  try {{
    const resp = await fetch('/save', {{
      method : 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body   : JSON.stringify({{ clip_file: fn, transcribe_here: trans, skip_reason: skip }}),
    }});
    if (resp.ok) {{
      clipEl.classList.remove('pending');
      clipEl.classList.add('done');
      statusEl.textContent = '✓ Saved';
      setTimeout(() => {{ statusEl.textContent = ''; }}, 1800);
      updateProgress();
    }} else {{
      statusEl.textContent = '✗ Server error';
    }}
  }} catch (e) {{
    statusEl.textContent = '✗ Connection error';
  }}
}}

/* ── filter buttons ── */
function filter(type, btn) {{
  document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('.clip').forEach(el => {{
    if      (type === 'all')     el.style.display = '';
    else if (type === 'done')    el.style.display = el.classList.contains('done')    ? '' : 'none';
    else if (type === 'pending') el.style.display = el.classList.contains('pending') ? '' : 'none';
  }});
}}

/* ── export ── */
async function exportJSON() {{
  try {{
    const resp = await fetch('/transcriptions');
    const data = await resp.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
    const a    = document.createElement('a');
    a.href     = URL.createObjectURL(blob);
    a.download = 'transcriptions.json';
    a.click();
  }} catch(e) {{
    alert('Export failed — is the server still running?');
  }}
}}

/* ── Enter = save + move to next visible clip ── */
document.querySelectorAll('.trans-input').forEach(ta => {{
  ta.addEventListener('keydown', e => {{
    if (e.key === 'Enter' && !e.shiftKey) {{
      e.preventDefault();
      const fn = ta.closest('.clip').dataset.file;
      saveClip(fn);
      const visible = [...document.querySelectorAll(
        '.clip:not([style*="display: none"]) .trans-input'
      )];
      const idx = visible.indexOf(ta);
      if (idx < visible.length - 1) {{
        visible[idx + 1].focus();
        visible[idx + 1].closest('.clip').scrollIntoView(
          {{behavior: 'smooth', block: 'center'}}
        );
      }}
    }}
  }});
}});

updateProgress();
"""

    clip_html = "\n".join(clip_blocks)

    return f"""<!DOCTYPE html>
<html lang="hi">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Haryanvi Transcription Helper</title>
<style>{css}</style>
</head>
<body>

<h1>Haryanvi / Bangru — Transcription Helper</h1>
<p class="subtitle">
  Click a text box, listen to the clip, correct the Whisper draft.
  Press <span class="kbd">Enter</span> to save and jump to next clip.
  <span class="kbd">Shift+Enter</span> for a line break.
  Use the keyboard bar at the bottom to insert dialect words/characters at the cursor.
</p>

<div class="prog-wrap"><div class="prog-fill" id="prog-fill" style="width:0%"></div></div>
<div class="prog-text" id="prog-text">Loading…</div>

<div class="filters">
  <button class="filter-btn active" onclick="filter('all', this)">All</button>
  <button class="filter-btn" onclick="filter('pending', this)">Pending</button>
  <button class="filter-btn" onclick="filter('done', this)">Done</button>
</div>

<div id="clips">
{clip_html}
</div>

<button class="export-btn" onclick="exportJSON()">Download transcriptions.json</button>

<!-- ── on-screen keyboard (fixed at bottom) ── -->
<div id="keyboard">
  <div class="kb-label">Common words — click to insert at cursor:</div>
  {keyboard_html}
  <div id="kb-hint"></div>
</div>

<script>{js}</script>
</body>
</html>"""


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            clips = load_clips()
            trans = load_transcriptions()
            body  = build_html(clips, trans).encode("utf-8")
            self._send(200, "text/html; charset=utf-8", body)

        elif path.startswith("/clips/"):
            clip_name = Path(path[7:]).name
            clip_path = CLIPS_DIR / clip_name
            if clip_path.suffix == ".wav" and clip_path.exists():
                self._send(200, "audio/wav", clip_path.read_bytes())
            else:
                self._send(404, "text/plain", b"not found")

        elif path == "/transcriptions":
            body = json.dumps(
                load_transcriptions(), ensure_ascii=False, indent=2
            ).encode("utf-8")
            self._send(200, "application/json", body)

        else:
            self._send(404, "text/plain", b"not found")

    def do_POST(self) -> None:
        if self.path != "/save":
            self._send(404, "text/plain", b"not found")
            return
        length = int(self.headers.get("Content-Length", 0))
        data   = json.loads(self.rfile.read(length))
        save_transcription(
            clip_file       = data.get("clip_file", ""),
            transcribe_here = data.get("transcribe_here", "").strip(),
            skip_reason     = data.get("skip_reason", ""),
        )
        self._send(200, "application/json", b'{"ok":true}')


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clips = load_clips()
    if not clips:
        print(f"⚠  {CSV_PATH} not found or empty — run chop_clips.py first.")
    else:
        drafts = sum(1 for c in clips if c["whisper_draft"])
        print(f"Loaded {len(clips)} clips  ({drafts} have a Whisper draft)")

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    url    = f"http://localhost:{PORT}"
    print(f"Transcription helper running at {url}")
    print("Press Ctrl+C to stop.\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
