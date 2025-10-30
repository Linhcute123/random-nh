\
import os, re, json, random, mimetypes, zlib, base64
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, redirect, abort, make_response, Response, send_file

app = Flask(__name__)

# Config
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "300"))
TIMEOUT = int(os.environ.get("TIMEOUT", "10"))
ALLOW_ANY_DOMAIN = os.environ.get("ALLOW_ANY_DOMAIN", "1") == "1"
HEAD_CHECK_MAX = int(os.environ.get("HEAD_CHECK_MAX", "12"))
USER_AGENT = os.environ.get("USER_AGENT", "Mozilla/5.0 (compatible; RandomImageBot/1.2)")
ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

# -------- Token encode/decode (deterministic share link) --------
def url_to_token(u: str) -> str:
    data = zlib.compress(u.encode("utf-8"))
    b = base64.urlsafe_b64encode(data).decode("ascii")
    return b.rstrip("=")

def token_to_url(token: str) -> str:
    pad = "=" * (-len(token) % 4)
    data = base64.urlsafe_b64decode(token + pad)
    return zlib.decompress(data).decode("utf-8")

def normalize_abs(page_url: str, candidate: str):
    if not candidate: return None
    c = candidate.strip()
    if not c or c.startswith("data:"): return None
    return urljoin(page_url, c)

def is_ext_image_url(u: str) -> bool:
    path = urlparse(u).path.lower()
    _, ext = os.path.splitext(path)
    return ext and ext in ALLOWED_EXTS

def head_is_image(u: str) -> bool:
    try:
        r = requests.head(u, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, allow_redirects=True)
        ct = r.headers.get("Content-Type", "")
        return ct.startswith("image/")
    except requests.RequestException:
        return False

def scrape_images(page_url: str) -> list[str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"}
    r = requests.get(page_url, headers=headers, timeout=TIMEOUT, allow_redirects=True)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")
    found_abs = []

    def add(u):
        ab = normalize_abs(page_url, u)
        if ab: found_abs.append(ab)

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
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

    seen = set(); uniq = []
    for u in found_abs:
        if u not in seen:
            uniq.append(u); seen.add(u)

    exts = [u for u in uniq if is_ext_image_url(u)]
    if len(exts) < MAX_IMAGES and HEAD_CHECK_MAX > 0:
        candidates = [u for u in uniq if not is_ext_image_url(u)]
        for u in candidates[:HEAD_CHECK_MAX]:
            if head_is_image(u):
                exts.append(u)
    return exts[:MAX_IMAGES]

# ---------------- Routes ----------------
INLINE_UI = r"""<!doctype html>
<html><head>
<meta charset='utf-8'/>
<meta name='viewport' content='width=device-width, initial-scale=1'/>
<title>Random Image — Single Link</title>
<style>
 body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;margin:24px}
 .card{border:1px solid #ddd;border-radius:12px;padding:16px;margin:16px 0}
 .row{display:flex;gap:8px;flex-wrap:wrap}
 input[type=url]{flex:1 1 520px;padding:10px;border-radius:8px;border:1px solid #ccc}
 button{padding:10px 14px;border-radius:8px;border:1px solid #aaa;cursor:pointer}
 .kvs{display:grid;grid-template-columns:170px 1fr;gap:8px}
 img{max-width:min(100%,800px);border-radius:10px;border:1px solid #ddd;margin-top:10px}
 .muted{color:#666;font-size:.9rem}
 code{background:#f5f5f5;padding:2px 6px;border-radius:6px}
</style></head><body>
<h2>Random Image — Single Link</h2>
<div class='card'>
  <div class='row'>
    <input id='pageUrl' type='url' placeholder='Dán link trang chứa ảnh (https://...)'>
    <button id='make'>Tạo link mặc định</button>
    <button id='open'>Mở ảnh random</button>
  </div>
  <div class='muted'>Link sẽ có dạng <code>/r/&lt;token&gt;</code> — luôn là <b>1 link cố định</b> cho trang bạn nhập.</div>
</div>

<div class='card' id='result' style='display:none'>
  <div class='kvs'>
    <div>Shareable link:</div>
    <div><a id='share' target='_blank' rel='noopener'></a> <button id='copyShare'>Copy</button></div>
    <div>JSON link:</div>
    <div><a id='jsonl' target='_blank' rel='noopener'></a> <button id='copyJson'>Copy</button></div>
  </div>
  <img id='preview' alt='Preview'/>
</div>

<script>
async function makeLink(){
  const url=document.getElementById('pageUrl').value.trim();
  if(!url){alert('Dán link trang vào trước đã.');return;}
  const r=await fetch('/make?url='+encodeURIComponent(url));
  if(!r.ok){const t=await r.text();throw new Error(t||('HTTP '+r.status));}
  const d=await r.json();
  document.getElementById('result').style.display='block';
  const s=document.getElementById('share'); s.href=d.shareable_random_link; s.textContent=d.shareable_random_link;
  const j=document.getElementById('jsonl'); j.href=d.json_link; j.textContent=d.json_link;
  const r2=await fetch(d.json_link); if(r2.ok){ const d2=await r2.json(); document.getElementById('preview').src=d2.random_image_url; }
}
document.getElementById('make').addEventListener('click', async ()=>{
  const btn=document.getElementById('make'); btn.disabled=true; btn.textContent='Đang xử lý...';
  try{ await makeLink(); } catch(e){ alert('Lỗi: '+e.message); }
  finally{ btn.disabled=false; btn.textContent='Tạo link mặc định'; }
});
document.getElementById('open').addEventListener('click', ()=>{
  const url=document.getElementById('pageUrl').value.trim();
  if(!url){alert('Dán link trang vào trước đã.');return;}
  makeLink().then(()=>{ window.open(document.getElementById('share').href,'_blank'); });
});
document.getElementById('copyShare').addEventListener('click',()=>{navigator.clipboard.writeText(document.getElementById('share').href).catch(()=>{});});
document.getElementById('copyJson').addEventListener('click',()=>{navigator.clipboard.writeText(document.getElementById('jsonl').href).catch(()=>{});});
</script>
</body></html>
"""

@app.get("/")
def root_ui():
    # If a static index exists, serve it; else inline UI.
    static_index = os.path.join(os.getcwd(), "static", "index.html")
    if os.path.isfile(static_index):
        return send_file(static_index)
    return Response(INLINE_UI, mimetype="text/html")

@app.get("/make")
def make_link():
    page_url = request.args.get("url", "").strip()
    if not page_url: abort(400, description="Missing ?url=")
    if not page_url.startswith(("http://", "https://")):
        abort(400, description="url must start with http(s)://")
    token = url_to_token(page_url)
    base = request.url_root.rstrip("/")
    return jsonify({
        "source_url": page_url,
        "token": token,
        "shareable_random_link": f"{base}/r/{token}",
        "json_link": f"{base}/j/{token}"
    })

@app.get("/r/<token>")
def open_random(token):
    try:
        page_url = token_to_url(token)
    except Exception:
        abort(400, description="Bad token")
    try:
        images = scrape_images(page_url)
    except requests.RequestException as e:
        abort(502, description=f"Failed to fetch page: {e}")
    if not images:
        abort(404, description="No images found on that page")
    url = random.choice(images)
    return no_cache(redirect(url, code=302))

@app.get("/j/<token>")
def json_random(token):
    try:
        page_url = token_to_url(token)
    except Exception:
        abort(400, description="Bad token")
    try:
        images = scrape_images(page_url)
    except requests.RequestException as e:
        abort(502, description=f"Failed to fetch page: {e}")
    if not images:
        abort(404, description="No images found on that page")
    url = random.choice(images)
    base = request.url_root.rstrip("/")
    return no_cache(jsonify({
        "source_url": page_url,
        "random_image_url": url,
        "shareable_random_link": f"{base}/r/{token}"
    }))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
