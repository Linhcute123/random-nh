\
import os, json, random, zlib, base64, io
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, redirect, abort, Response

from PIL import Image
import numpy as np
import cv2

app = Flask(__name__)

# -------- Defaults --------
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "400"))
TIMEOUT = int(os.environ.get("TIMEOUT", "12"))
USER_AGENT = os.environ.get("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36 RandomImageBot/1.7")
GET_TRY_LIMIT = int(os.environ.get("GET_TRY_LIMIT", "50"))

# Heuristic default thresholds (used when filters are ON)
DEFAULT_EXCLUDE_KEYWORDS = [s.strip().lower() for s in os.environ.get(
    "EXCLUDE_KEYWORDS",
    "logo, icon, sprite, favicon, avatar, badge, banner, placeholder, ads, advert, tracking, pixel, og:image"
).split(",")]
DEFAULT_MIN_WIDTH = int(os.environ.get("MIN_WIDTH", "300"))
DEFAULT_MIN_HEIGHT = int(os.environ.get("MIN_HEIGHT", "300"))
DEFAULT_MIN_BYTES = int(os.environ.get("MIN_BYTES", "12000"))
DEFAULT_MAX_AR = float(os.environ.get("MAX_ASPECT_RATIO", "3.8"))
DEFAULT_REQUIRE_PERSON = os.environ.get("REQUIRE_PERSON", "1") == "1"

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

# OpenCV HOG person detector
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# -------- Token encode/decode --------
def to_token(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    comp = zlib.compress(data)
    b64 = base64.urlsafe_b64encode(comp).decode("ascii").rstrip("=")
    return b64

def from_token(token: str) -> dict:
    pad = "=" * (-len(token) % 4)
    comp = base64.urlsafe_b64decode(token + pad)
    data = zlib.decompress(comp)
    return json.loads(data.decode("utf-8"))

# -------- HTTP helpers --------
def _headers(page_url: str) -> dict:
    return {
        "User-Agent": USER_AGENT,
        "Referer": page_url,
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.8"
    }

def normalize_abs(page_url: str, candidate: str):
    if not candidate:
        return None
    c = candidate.strip()
    if not c or c.startswith("data:"):
        return None
    return urljoin(page_url, c)

def is_ext_image_url(u: str) -> bool:
    path = urlparse(u).path.lower()
    _, ext = os.path.splitext(path)
    return ext and ext in ALLOWED_EXTS

def fetch_image_bytes(page_url: str, u: str) -> bytes | None:
    try:
        r = requests.get(u, headers=_headers(page_url), timeout=TIMEOUT, allow_redirects=True, stream=True)
        r.raise_for_status()
        return r.content[:8*1024*1024]
    except requests.RequestException:
        return None

def has_person(img: Image.Image) -> bool:
    arr = np.array(img.convert("RGB"))
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    scale = 720 / max(h, w) if max(h, w) > 720 else 1.0
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(w*scale), int(h*scale)))
    rects, _ = _hog.detectMultiScale(bgr, winStride=(8,8), padding=(8,8), scale=1.05)
    return len(rects) > 0

def scrape_candidates(page_url: str) -> list[str]:
    r = requests.get(page_url, headers=_headers(page_url) | {"Accept":"text/html,application/xhtml+xml"}, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    found_abs = []

    def add(u):
        ab = normalize_abs(page_url, u)
        if ab:
            found_abs.append(ab)

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            add(img.get(attr))
        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add(p)

    for src in soup.find_all("source"):
        srcset = src.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add(p)

    # background-image in inline styles (basic)
    for el in soup.find_all(style=True):
        st = el.get("style", "")
        if "url(" in st:
            urlfrag = st.split("url(",1)[1].split(")",1)[0].strip(" '\"")
            add(urlfrag)

    # dedupe
    seen = set(); uniq = []
    for u in found_abs:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq[:MAX_IMAGES]

# -------- Filtering / Selection --------
def excluded_by_keyword(u: str, keywords: list[str]) -> bool:
    lu = u.lower()
    for k in keywords:
        if k and k in lu:
            return True
    if lu.endswith(".svg") or lu.endswith(".ico"):
        return True
    return False

def pick_any_image(page_url: str, cands: list[str]) -> str | None:
    """Minimal filter: first candidate that loads as image and has >= 120x120."""
    random.shuffle(cands)
    tries = 0
    for u in cands:
        if tries >= GET_TRY_LIMIT:
            break
        tries += 1
        data = fetch_image_bytes(page_url, u)
        if not data:
            continue
        try:
            im = Image.open(io.BytesIO(data))
            im.verify()
            im = Image.open(io.BytesIO(data))
            if im.width >= 120 and im.height >= 120:
                return u
        except Exception:
            continue
    return None

def pick_with_filters(page_url: str, cands: list[str], cfg: dict) -> str | None:
    # Build config
    merged_excl = DEFAULT_EXCLUDE_KEYWORDS[:]
    user_excl = [s.strip().lower() for s in cfg.get("exclude", []) if s.strip()]
    for s in user_excl:
        if s not in merged_excl:
            merged_excl.append(s)

    min_w = int(cfg.get("min_w", DEFAULT_MIN_WIDTH))
    min_h = int(cfg.get("min_h", DEFAULT_MIN_HEIGHT))
    min_bytes = int(cfg.get("min_bytes", DEFAULT_MIN_BYTES))
    max_ar = float(cfg.get("max_ar", DEFAULT_MAX_AR))
    require_person = bool(cfg.get("require_person", DEFAULT_REQUIRE_PERSON))
    smart_fallback = bool(cfg.get("smart_fallback", True))

    def dims_ok(img):
        w, h = img.size
        if w < min_w or h < min_h: return False
        ar = max(w, h)/max(1, min(w, h))
        return ar <= max_ar

    def try_pick(allow_person: bool, min_bytes_local: int, min_w_local: int, min_h_local: int, max_ar_local: float) -> str | None:
        random.shuffle(cands)
        tries = 0
        for u in cands:
            if excluded_by_keyword(u, merged_excl):
                continue
            if tries >= GET_TRY_LIMIT:
                break
            tries += 1
            data = fetch_image_bytes(page_url, u)
            if not data:
                continue
            try:
                im = Image.open(io.BytesIO(data))
                im.verify()
                im = Image.open(io.BytesIO(data)).convert("RGB")
            except Exception:
                continue
            if im.width < min_w_local or im.height < min_h_local:
                continue
            ar = max(im.width, im.height)/max(1, min(im.width, im.height))
            if ar > max_ar_local:
                continue
            if allow_person and not has_person(im):
                continue
            return u
        return None

    # Stage 1
    url = try_pick(require_person, min_bytes, min_w, min_h, max_ar)
    if url or not smart_fallback:
        return url
    # Stage 2 (looser bytes implicit by not checking, and keep person)
    url = try_pick(require_person, max(5000, min_bytes//2), min_w, min_h, max_ar)
    if url:
        return url
    # Stage 3 (drop person)
    url = try_pick(False, max(5000, min_bytes//2), min_w, min_h, max_ar)
    if url:
        return url
    # Stage 4 (looser dims/aspect)
    url = try_pick(False, 3000, max(200, min_w//2), max(200, min_h//2), max(max_ar, 5.0))
    return url

# -------- UI --------
ROOT_HTML = r"""<!doctype html>
<html><head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Random Image — Single Link (AI + Filters Toggle)</title>
<style>
:root{--bg:#0b0f19; --fg:#e6e8ef; --muted:#a0a7b4; --card:#111726; --border:#1c2437; --accent:#5b8cff;}
@media (prefers-color-scheme: light){:root{--bg:#f7f8fb; --fg:#0b1220; --muted:#5a6475; --card:#fff; --border:#e5e8ef; --accent:#3b6cff;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif}
.wrapper{max-width:960px;margin:0 auto;padding:24px}
.header{display:flex;align-items:center;justify-content:space-between;margin-bottom:16px}
.brand{font-weight:700;font-size:20px;letter-spacing:.2px}
.card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:16px;box-shadow:0 6px 24px rgba(0,0,0,.12);margin:16px 0}
label{font-size:13px;color:var(--muted);display:block;margin-bottom:6px}
input[type=url],textarea{width:100%;padding:12px 14px;border-radius:12px;border:1px solid var(--border);background:transparent;color:var(--fg);outline:none}
textarea{min-height:72px;resize:vertical}
.row{display:grid;grid-template-columns:1fr;gap:12px}
@media(min-width:720px){ .row.two{grid-template-columns:1fr 1fr;} }
.btn{display:inline-flex;gap:8px;align-items:center;padding:10px 14px;border-radius:12px;border:1px solid var(--border);background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.01));color:var(--fg);cursor:pointer}
.btn:hover{transform:translateY(-1px)}
.btn.primary{border-color:transparent;background:linear-gradient(180deg,var(--accent),#2f5fff);color:white}
.actions{display:flex;gap:10px;flex-wrap:wrap}
.kvs{display:grid;grid-template-columns:180px 1fr;gap:10px;align-items:center}
a{color:var(--accent);text-decoration:none}
.preview{margin-top:14px}
img{max-width:min(100%,820px);border-radius:12px;border:1px solid var(--border)}
.note{color:var(--muted);font-size:13px}
.switch{display:flex;align-items:center;gap:8px}
.switch input{width:18px;height:18px}
.toast{position:fixed;right:16px;bottom:16px;background:#1f2937;color:#fff;padding:10px 14px;border-radius:10px;opacity:.95}
</style>
</head>
<body>
<div class="wrapper">
  <div class="header">
    <div class="brand">Random Image — Single Link</div>
  </div>

  <div class="card">
    <div class="row">
      <div>
        <label>Link trang chứa ảnh</label>
        <input id="pageUrl" type="url" placeholder="https://...">
      </div>
    </div>
    <div class="row two">
      <div>
        <label>Từ khoá loại trừ (phân tách bằng dấu phẩy)</label>
        <textarea id="exclude" placeholder="ví dụ: logo, watermark, banner"></textarea>
        <div class="note">Có sẵn bộ lọc mặc định (logo, icon, avatar, banner, favicon, ...). Bạn có thể thêm tại đây.</div>
      </div>
      <div>
        <label class="switch"><input id="requirePerson" type="checkbox" checked> Chỉ lấy ảnh có người (AI)</label>
        <label class="switch"><input id="smartFallback" type="checkbox" checked> Tự nới lỏng nếu không tìm được</label>
        <label class="switch"><input id="disableFilters" type="checkbox"> <b>Tắt mọi bộ lọc</b> (chấp nhận mọi ảnh hợp lệ)</label>
        <div class="note">Khi bật “Tắt mọi bộ lọc”, công cụ chỉ kiểm tra ảnh có tải được và kích thước ≥ 120×120.</div>
      </div>
    </div>
    <div class="actions">
      <button class="btn primary" id="make">Tạo link mặc định</button>
      <button class="btn" id="open">Mở ảnh ngẫu nhiên</button>
    </div>
  </div>

  <div id="resultCard" class="card" style="display:none;">
    <div class="kvs">
      <div>Shareable link:</div>
      <div><a id="share" target="_blank" rel="noopener"></a> <button class="btn" id="copyShare">Copy</button></div>
      <div>JSON link:</div>
      <div><a id="jsonl" target="_blank" rel="noopener"></a> <button class="btn" id="copyJson">Copy</button></div>
    </div>
    <div class="preview"><img id="preview" alt="Preview"></div>
  </div>
</div>

<div id="toast" class="toast" style="display:none;"></div>

<script>
function showToast(msg){ const t=document.getElementById('toast'); t.textContent=msg; t.style.display='block'; setTimeout(()=>t.style.display='none',1500); }
async function copyText(text){
  try{
    if(navigator.clipboard && window.isSecureContext){
      await navigator.clipboard.writeText(text);
    }else{
      const ta=document.createElement('textarea'); ta.value=text; document.body.appendChild(ta);
      ta.style.position='fixed'; ta.style.left='-9999px'; ta.select(); document.execCommand('copy'); document.body.removeChild(ta);
    }
    showToast('Đã copy');
  }catch(e){ alert('Copy thất bại: '+e.message); }
}
async function makeLink(){
  const url = document.getElementById('pageUrl').value.trim();
  if(!url){ alert('Dán link trang vào trước đã.'); return; }
  const exclude = document.getElementById('exclude').value.trim();
  const rp = document.getElementById('requirePerson').checked ? 1 : 0;
  const sf = document.getElementById('smartFallback').checked ? 1 : 0;
  const df = document.getElementById('disableFilters').checked ? 1 : 0;
  const r = await fetch(`/make?url=${encodeURIComponent(url)}&exclude=${encodeURIComponent(exclude)}&require_person=${rp}&smart_fallback=${sf}&disable_filters=${df}`);
  if(!r.ok){ const t = await r.text(); throw new Error(t || ('HTTP '+r.status)); }
  const d = await r.json();
  document.getElementById('resultCard').style.display='block';
  const s=document.getElementById('share'); s.href=d.shareable_random_link; s.textContent=d.shareable_random_link;
  const j=document.getElementById('jsonl'); j.href=d.json_link; j.textContent=d.json_link;
  const r2=await fetch(d.json_link); if(r2.ok){ const d2=await r2.json(); document.getElementById('preview').src=d2.random_image_url; }
}
document.getElementById('make').addEventListener('click', async ()=>{
  const btn=document.getElementById('make'); btn.disabled=true; btn.textContent='Đang xử lý...';
  try{ await makeLink(); } catch(e){ alert('Lỗi: '+e.message); }
  finally{ btn.disabled=false; btn.textContent='Tạo link mặc định'; }
});
document.getElementById('open').addEventListener('click', async ()=>{
  try{ await makeLink(); window.open(document.getElementById('share').href,'_blank'); }catch(e){ alert('Lỗi: '+e.message); }
});
document.getElementById('copyShare').addEventListener('click', ()=>copyText(document.getElementById('share').href));
document.getElementById('copyJson').addEventListener('click', ()=>copyText(document.getElementById('jsonl').href));
</script>
</body></html>
"""

def parse_exclude(s: str) -> list[str]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def build_payload(page_url: str, exclude_kw: str, require_person_flag: str, smart_fallback_flag: str, disable_filters_flag: str) -> dict:
    payload = {"url": page_url}
    ex = parse_exclude(exclude_kw)
    if ex: payload["exclude"] = ex
    rp = 1 if require_person_flag in ("1","true","True","on") else 0
    sf = 1 if smart_fallback_flag in ("1","true","True","on") else 0
    df = 1 if disable_filters_flag in ("1","true","True","on") else 0
    payload["require_person"] = bool(rp)
    payload["smart_fallback"] = bool(sf)
    payload["disable_filters"] = bool(df)
    return payload

@app.get("/")
def root_ui():
    return Response(ROOT_HTML, mimetype="text/html")

@app.get("/make")
def make_link():
    page_url = request.args.get("url", "").strip()
    if not page_url:
        abort(400, description="Missing ?url=")
    if not page_url.startswith(("http://", "https://")):
        abort(400, description="url must start with http(s)://")
    exclude_kw = request.args.get("exclude", "").strip()
    require_person_flag = request.args.get("require_person", "1").strip()
    smart_fallback_flag = request.args.get("smart_fallback", "1").strip()
    disable_filters_flag = request.args.get("disable_filters", "0").strip()
    payload = build_payload(page_url, exclude_kw, require_person_flag, smart_fallback_flag, disable_filters_flag)
    token = to_token(payload)
    base = request.url_root.rstrip("/")
    return jsonify({
        "source_url": page_url,
        "token": token,
        "shareable_random_link": f"{base}/r/{token}",
        "json_link": f"{base}/j/{token}"
    })

def scrape_candidates(page_url: str) -> list[str]:
    r = requests.get(page_url, headers=_headers(page_url) | {"Accept":"text/html,application/xhtml+xml"}, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    found_abs = []
    def add(u):
        ab = normalize_abs(page_url, u)
        if ab: found_abs.append(ab)
    for img in soup.find_all("img"):
        for attr in ("src","data-src","data-original","data-lazy","data-image"):
            add(img.get(attr))
        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts: add(p)
    for src in soup.find_all("source"):
        srcset = src.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts: add(p)
    for el in soup.find_all(style=True):
        st = el.get("style", "")
        if "url(" in st:
            urlfrag = st.split("url(",1)[1].split(")",1)[0].strip(" '\"")
            add(urlfrag)
    seen=set(); uniq=[]
    for u in found_abs:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq[:MAX_IMAGES]

from flask import abort

@app.get("/r/<token>")
def open_random(token):
    try:
        payload = from_token(token)
    except Exception:
        abort(400, description="Bad token")
    page_url = payload.get("url", "")
    if not page_url:
        abort(400, description="Token missing url")
    try:
        cands = scrape_candidates(page_url)
    except requests.RequestException as e:
        abort(502, description=f"Failed to fetch page: {e}")
    if payload.get("disable_filters", False):
        url = pick_any_image(page_url, cands)
    else:
        url = pick_with_filters(page_url, cands, cfg=payload)
    if not url:
        abort(404, description="No suitable image found after filtering")
    return no_cache(redirect(url, code=302))

@app.get("/j/<token>")
def json_random(token):
    try:
        payload = from_token(token)
    except Exception:
        abort(400, description="Bad token")
    page_url = payload.get("url", "")
    if not page_url:
        abort(400, description="Token missing url")
    try:
        cands = scrape_candidates(page_url)
    except requests.RequestException as e:
        abort(502, description=f"Failed to fetch page: {e}")
    if payload.get("disable_filters", False):
        url = pick_any_image(page_url, cands)
    else:
        url = pick_with_filters(page_url, cands, cfg=payload)
    if not url:
        abort(404, description="No suitable image found after filtering")
    base = request.url_root.rstrip("/")
    return no_cache(jsonify({
        "source_url": page_url,
        "filters": {k:payload.get(k) for k in ("exclude","require_person","smart_fallback","disable_filters") if k in payload},
        "random_image_url": url,
        "shareable_random_link": f"{base}/r/{token}"
    }))

@app.get("/health")
def health():
    return "ok"
