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
HEAD_CHECK_MAX = int(os.environ.get("HEAD_CHECK_MAX", "12"))
USER_AGENT = os.environ.get("USER_AGENT", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36 RandomImageBot/1.6")

DEFAULT_EXCLUDE_KEYWORDS = [s.strip().lower() for s in os.environ.get(
    "EXCLUDE_KEYWORDS",
    "logo, icon, sprite, favicon, avatar, badge, banner, placeholder, ads, advert, tracking, pixel, og:image"
).split(",")]

DEFAULT_MIN_WIDTH = int(os.environ.get("MIN_WIDTH", "300"))
DEFAULT_MIN_HEIGHT = int(os.environ.get("MIN_HEIGHT", "300"))
DEFAULT_MIN_BYTES = int(os.environ.get("MIN_BYTES", "12000"))
DEFAULT_MAX_AR = float(os.environ.get("MAX_ASPECT_RATIO", "3.8"))
DEFAULT_REQUIRE_PERSON = os.environ.get("REQUIRE_PERSON", "1") == "1"
GET_TRY_LIMIT = int(os.environ.get("GET_TRY_LIMIT", "40"))

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
        "Referer": page_url,  # some CDNs require this
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

def head_info(page_url: str, u: str):
    try:
        r = requests.head(u, headers=_headers(page_url), timeout=TIMEOUT, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")
        clen = 0
        try:
            clen = int(r.headers.get("Content-Length", "0"))
        except ValueError:
            clen = 0
        return (ct.startswith("image/"), clen, True)
    except requests.RequestException:
        return (False, 0, False)

def fetch_image_bytes(page_url: str, u: str) -> bytes | None:
    try:
        r = requests.get(u, headers=_headers(page_url), timeout=TIMEOUT, allow_redirects=True, stream=True)
        r.raise_for_status()
        return r.content[:8*1024*1024]
    except requests.RequestException:
        return None

def img_dims_ok(img: Image.Image, min_w: int, min_h: int, max_ar: float) -> bool:
    w, h = img.size
    if w < min_w or h < min_h:
        return False
    ar = max(w, h) / max(1.0, min(w, h))
    if ar > max_ar:
        return False
    return True

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

    # <img> sources
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            add(img.get(attr))
        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add(p)

    # <source srcset> in <picture>
    for src in soup.find_all("source"):
        srcset = src.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add(p)

    # background-image in inline style (basic)
    for el in soup.find_all(style=True):
        st = el.get("style", "")
        # crude url() extractor
        for part in st.split("url(")[1:]:
            urlfrag = part.split(")", 1)[0].strip(" '\"")
            add(urlfrag)

    # dedupe
    seen = set(); uniq = []
    for u in found_abs:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq[:MAX_IMAGES]

def excluded_by_keyword(u: str, keywords: list[str]) -> bool:
    lu = u.lower()
    for k in keywords:
        if k and k in lu:
            return True
    if lu.endswith(".svg") or lu.endswith(".ico"):
        return True
    return False

def try_pick(page_url, cands, merged_excl, min_w, min_h, min_bytes, max_ar, require_person, get_try_limit, debug=False):
    tries = 0
    debug_rows = []
    for u in cands:
        row = {"url": u, "steps": []}
        if excluded_by_keyword(u, merged_excl):
            row["steps"].append("excluded: keyword")
            if debug: debug_rows.append(row)
            continue
        ext_ok = is_ext_image_url(u)
        img_hdr, clen, head_ok = head_info(page_url, u)
        row["steps"].append(f"HEAD: img={img_hdr} len={clen} ok={head_ok}")
        if img_hdr and clen and clen < min_bytes:
            row["steps"].append("skip: tiny Content-Length")
            if debug: debug_rows.append(row)
            continue

        data = None
        if tries < get_try_limit:
            tries += 1
            data = fetch_image_bytes(page_url, u)
        else:
            row["steps"].append("skip: GET budget exceeded")
            if debug: debug_rows.append(row)
            continue

        if not data:
            row["steps"].append("skip: GET failed/empty")
            if debug: debug_rows.append(row)
            continue

        try:
            im = Image.open(io.BytesIO(data))
            im.verify()
            im = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception:
            row["steps"].append("skip: PIL not image")
            if debug: debug_rows.append(row)
            continue
        if not img_dims_ok(im, min_w=min_w, min_h=min_h, max_ar=max_ar):
            row["steps"].append("skip: dims/aspect")
            if debug: debug_rows.append(row)
            continue
        if require_person and not has_person(im):
            row["steps"].append("skip: no person")
            if debug: debug_rows.append(row)
            continue

        row["steps"].append("ACCEPT")
        if debug: debug_rows.append(row)
        return u, debug_rows

        # else continue
    return None, debug_rows

def pick_random_valid(page_url: str, cands: list[str], cfg: dict, debug=False):
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

    random.shuffle(cands)
    url, dbg = try_pick(page_url, cands, merged_excl, min_w, min_h, min_bytes, max_ar, require_person, GET_TRY_LIMIT, debug=debug)
    if url or not smart_fallback:
        return url, dbg

    # fallback 1
    url, dbg2 = try_pick(page_url, cands, merged_excl, min_w, min_h, max(5000, min_bytes//2), max_ar, require_person, GET_TRY_LIMIT, debug=debug)
    dbg += dbg2
    if url:
        return url, dbg

    # fallback 2
    url, dbg3 = try_pick(page_url, cands, merged_excl, min_w, min_h, max(5000, min_bytes//2), max_ar, False, GET_TRY_LIMIT, debug=debug)
    dbg += dbg3
    if url:
        return url, dbg

    # fallback 3
    url, dbg4 = try_pick(page_url, cands, merged_excl, max(200, min_w//2), max(200, min_h//2), 3000, max(max_ar, 5.0), False, GET_TRY_LIMIT, debug=debug)
    dbg += dbg4
    return url, dbg

# -------- UI --------
ROOT_HTML = r"""<!doctype html>
<html><head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Random Image — Single Link (AI + Keyword Filters)</title>
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
pre{white-space:pre-wrap;background:#0f172a;color:#cbd5e1;border-radius:12px;padding:12px;max-height:340px;overflow:auto;border:1px solid var(--border)}
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
        <div class="note">Có sẵn bộ lọc mặc định: logo, icon, avatar, banner, favicon, ... (cộng thêm từ khoá bạn nhập)</div>
      </div>
      <div>
        <label class="switch"><input id="requirePerson" type="checkbox" checked> Chỉ lấy ảnh có người (AI)</label>
        <label class="switch"><input id="smartFallback" type="checkbox" checked> Tự nới lỏng nếu không tìm được</label>
        <div class="note">Fallback sẽ thử lỏng dần (bytes/AI/kích thước) để tránh 404.</div>
      </div>
    </div>
    <div class="actions">
      <button class="btn primary" id="make">Tạo link mặc định</button>
      <button class="btn" id="open">Mở ảnh ngẫu nhiên</button>
      <button class="btn" id="test">Kiểm tra (debug)</button>
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

  <div id="debugCard" class="card" style="display:none;">
    <div class="note">Kết quả kiểm tra (chỉ 1 phần, giúp hiểu vì sao bị lọc):</div>
    <pre id="debugOut"></pre>
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
  const r = await fetch(`/make?url=${encodeURIComponent(url)}&exclude=${encodeURIComponent(exclude)}&require_person=${rp}&smart_fallback=${sf}`);
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

document.getElementById('test').addEventListener('click', async ()=>{
  const url = document.getElementById('pageUrl').value.trim();
  if(!url){ alert('Dán link trang vào trước đã.'); return; }
  const exclude = document.getElementById('exclude').value.trim();
  const rp = document.getElementById('requirePerson').checked ? 1 : 0;
  const sf = document.getElementById('smartFallback').checked ? 1 : 0;
  const r = await fetch(`/debug?url=${encodeURIComponent(url)}&exclude=${encodeURIComponent(exclude)}&require_person=${rp}&smart_fallback=${sf}`);
  const out = document.getElementById('debugOut'); document.getElementById('debugCard').style.display='block';
  if(!r.ok){ out.textContent = await r.text(); return; }
  const d = await r.json();
  out.textContent = JSON.stringify(d, null, 2);
});
</script>
</body></html>
"""

def parse_exclude(s: str) -> list[str]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",")]
    return [p for p in parts if p]

def build_payload(page_url: str, exclude_kw: str, require_person_flag: str, smart_fallback_flag: str) -> dict:
    payload = {"url": page_url}
    ex = parse_exclude(exclude_kw)
    if ex: payload["exclude"] = ex
    rp = 1 if require_person_flag in ("1","true","True","on") else 0
    payload["require_person"] = bool(rp)
    sf = 1 if smart_fallback_flag in ("1","true","True","on") else 0
    payload["smart_fallback"] = bool(sf)
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
    payload = build_payload(page_url, exclude_kw, require_person_flag, smart_fallback_flag)
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
            after = st.split("url(",1)[1]
            urlfrag = after.split(")",1)[0].strip(" '\"")
            add(urlfrag)
    seen=set(); uniq=[]
    for u in found_abs:
        if u not in seen:
            uniq.append(u); seen.add(u)
    return uniq[:MAX_IMAGES]

def debug_scan(page_url, cfg):
    cands = scrape_candidates(page_url)
    url, dbg = pick_random_valid(page_url, cands, cfg, debug=True)
    return {"candidates": len(cands), "accepted": url or None, "trace": dbg[:60]}  # limit trace

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
    url, _ = pick_random_valid(page_url, cands, cfg=payload, debug=False)
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
    url, _ = pick_random_valid(page_url, cands, cfg=payload, debug=False)
    if not url:
        abort(404, description="No suitable image found after filtering")
    base = request.url_root.rstrip("/")
    return no_cache(jsonify({
        "source_url": page_url,
        "filters": {k:payload.get(k) for k in ("exclude","require_person","smart_fallback") if k in payload},
        "random_image_url": url,
        "shareable_random_link": f"{base}/r/{token}"
    }))

@app.get("/debug")
def debug_endpoint():
    page_url = request.args.get("url","").strip()
    if not page_url: abort(400, description="Missing ?url=")
    if not page_url.startswith(("http://","https://")): abort(400, description="url must start with http(s)://")
    exclude_kw = request.args.get("exclude","").strip()
    require_person_flag = request.args.get("require_person","1").strip()
    smart_fallback_flag = request.args.get("smart_fallback","1").strip()
    payload = build_payload(page_url, exclude_kw, require_person_flag, smart_fallback_flag)
    try:
        result = debug_scan(page_url, payload)
    except requests.RequestException as e:
        abort(502, description=f"Failed to fetch page: {e}")
    return no_cache(jsonify(result))

@app.get("/health")
def health():
    return "ok"
