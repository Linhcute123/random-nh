\
import os, re, json, secrets, string, random, mimetypes
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, redirect, abort, make_response, send_file, Response

app = Flask(__name__)

BASE_DIR = os.path.abspath(os.environ.get("BASE_DIR", "."))

# Config
MAX_IMAGES = int(os.environ.get("MAX_IMAGES", "300"))
TIMEOUT = int(os.environ.get("TIMEOUT", "10"))
ALLOW_ANY_DOMAIN = os.environ.get("ALLOW_ANY_DOMAIN", "1") == "1"  # default ON
HEAD_CHECK_MAX = int(os.environ.get("HEAD_CHECK_MAX", "12"))  # number of extensionless URLs to probe via HEAD

USER_AGENT = os.environ.get("USER_AGENT", "Mozilla/5.0 (compatible; RandomImageBot/1.1)")

ALLOWED_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}
SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

def no_cache(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

def domain_allowed(url: str) -> bool:
    return True if ALLOW_ANY_DOMAIN else False

def normalize_abs(page_url: str, candidate: str) -> str | None:
    if not candidate:
        return None
    cand = candidate.strip()
    if not cand or cand.startswith("data:"):
        return None
    return urljoin(page_url, cand)

def is_ext_image_url(u: str) -> bool:
    path = urlparse(u).path.lower()
    ext = os.path.splitext(path)[1]
    return bool(ext and ext in ALLOWED_EXTS)

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
    found_abs: list[str] = []

    def add_img(u: str):
        absu = normalize_abs(page_url, u)
        if absu:
            found_abs.append(absu)

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            add_img(img.get(attr))
        srcset = img.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add_img(p)

    for src in soup.find_all("source"):
        srcset = src.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add_img(p)

    seen = set()
    uniq = []
    for u in found_abs:
        if u not in seen:
            uniq.append(u)
            seen.add(u)

    exts = [u for u in uniq if is_ext_image_url(u)]

    if len(exts) < MAX_IMAGES and HEAD_CHECK_MAX > 0:
        candidates = [u for u in uniq if not is_ext_image_url(u)]
        head_ok = []
        for u in candidates[:HEAD_CHECK_MAX]:
            if head_is_image(u):
                head_ok.append(u)
        exts.extend(head_ok)

    return exts[:MAX_IMAGES]

# ----------- Routes -----------
@app.get("/")
def home():
    return redirect("/ui", code=302)

@app.get("/config")
def config_info():
    return jsonify({
        "allow_any_domain": ALLOW_ANY_DOMAIN,
        "max_images": MAX_IMAGES,
        "timeout": TIMEOUT,
        "head_check_max": HEAD_CHECK_MAX,
        "base_dir": BASE_DIR
    })

@app.get("/rand.json")
def rand_json():
    page_url = request.args.get("url", "").strip()
    if not page_url:
        abort(400, description="Missing ?url=")
    if not page_url.startswith(("http://", "https://")):
        abort(400, description="url must start with http(s)://")
    if not domain_allowed(page_url):
        abort(403, description="Domain not allowed")
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
        "shareable_random_link": f"{base}/rand?url={page_url}"
    }))

@app.get("/rand")
def one_shot_redirect():
    page_url = request.args.get("url", "").strip()
    if not page_url:
        abort(400, description="Missing ?url=")
    if not page_url.startswith(("http://", "https://")):
        abort(400, description="url must start with http(s)://")
    if not domain_allowed(page_url):
        abort(403, description="Domain not allowed")
    try:
        images = scrape_images(page_url)
    except requests.RequestException as e:
        abort(502, description=f"Failed to fetch page: {e}")
    if not images:
        abort(404, description="No images found on that page")
    url = random.choice(images)
    return no_cache(redirect(url, code=302))

# ---------- UI (with fallback) ----------
INLINE_UI = r"""<!doctype html>
<html>
<head>
  <meta charset='utf-8' />
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <title>One-shot Random Image</title>
  <style>
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 24px; }
    .card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin: 16px 0; }
    .row { display: flex; gap: 8px; flex-wrap: wrap; }
    input[type='url'] { flex: 1 1 520px; padding: 10px; border-radius: 8px; border: 1px solid #ccc; }
    button { padding: 10px 14px; border-radius: 8px; border: 1px solid #aaa; cursor: pointer; }
    .muted { color: #666; font-size: 0.9rem; }
    .kvs { display: grid; grid-template-columns: 170px 1fr; gap: 8px; }
    .preview { margin-top: 12px; }
    img { max-width: min(100%, 800px); border-radius: 10px; border: 1px solid #ddd; }
    code { background: #f5f5f5; padding: 2px 6px; border-radius: 6px; }
  </style>
</head>
<body>
  <h2>One-shot Random Image</h2>
  <div class='card'>
    <div class='row'>
      <input id='pageUrl' type='url' placeholder='Dán link trang chứa ảnh (https://...)' />
      <button id='runBtn'>Chạy & hiển thị link</button>
      <button id='openBtn'>Mở ảnh random</button>
    </div>
    <div class='muted' id='cfg'></div>
  </div>

  <div class='card' id='result' style='display:none;'>
    <h3>Kết quả</h3>
    <div class='kvs'>
      <div>Shareable random link:</div>
      <div>
        <a id='shareLink' target='_blank' rel='noopener'></a>
        <button id='copyShare'>Copy</button>
      </div>
      <div>Random image URL:</div>
      <div>
        <a id='imgUrl' target='_blank' rel='noopener'></a>
        <button id='copyImg'>Copy</button>
      </div>
    </div>
    <div class='preview'>
      <h4>Preview</h4>
      <img id='preview' alt='Random preview' />
    </div>
  </div>

<script>
async function fetchConfig() {
  try {
    const r = await fetch('/config');
    if (!r.ok) throw new Error('config failed');
    const cfg = await r.json();
    document.getElementById('cfg').textContent =
      `Allow any domain: ${cfg.allow_any_domain ? 'ON' : 'OFF'} | MAX_IMAGES=${cfg.max_images} | HEAD_CHECK_MAX=${cfg.head_check_max}`;
  } catch (e) {
    document.getElementById('cfg').textContent = 'Không đọc được cấu hình server.';
  }
}

async function runOnce() {
  const url = document.getElementById('pageUrl').value.trim();
  if (!url) { alert('Dán link trang vào trước đã.'); return; }
  const r = await fetch('/rand.json?url=' + encodeURIComponent(url));
  if (!r.ok) { const t = await r.text(); throw new Error(t || ('HTTP ' + r.status)); }
  const data = await r.json();
  document.getElementById('result').style.display = 'block';
  document.getElementById('shareLink').href = data.shareable_random_link;
  document.getElementById('shareLink').textContent = data.shareable_random_link;
  document.getElementById('imgUrl').href = data.random_image_url;
  document.getElementById('imgUrl').textContent = data.random_image_url;
  document.getElementById('preview').src = data.random_image_url;
}

document.getElementById('runBtn').addEventListener('click', async () => {
  const btn = document.getElementById('runBtn');
  btn.disabled = true; btn.textContent = 'Đang lấy ảnh...';
  try { await runOnce(); } catch (e) { alert('Lỗi: ' + e.message); }
  finally { btn.disabled = false; btn.textContent = 'Chạy & hiển thị link'; }
});

document.getElementById('openBtn').addEventListener('click', () => {
  const url = document.getElementById('pageUrl').value.trim();
  if (!url) { alert('Dán link trang vào trước đã.'); return; }
  window.open('/rand?url=' + encodeURIComponent(url), '_blank');
});

document.getElementById('copyShare').addEventListener('click', () => {
  const t = document.getElementById('shareLink').href;
  navigator.clipboard.writeText(t).catch(()=>{});
});
document.getElementById('copyImg').addEventListener('click', () => {
  const t = document.getElementById('imgUrl').href;
  navigator.clipboard.writeText(t).catch(()=>{});
});

fetchConfig();
</script>
</body>
</html>"""

@app.get("/ui")
def ui_page():
    html_path = os.path.join(BASE_DIR, "static", "ui", "index.html")
    if os.path.isfile(html_path):
        return send_file(html_path)
    # Fallback inline UI if the file is missing
    return Response(INLINE_UI, mimetype="text/html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
