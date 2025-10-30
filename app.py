\
import os, re, json, secrets, string, random, mimetypes
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, redirect, abort, make_response, send_file

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
    # no domain limits when ALLOW_ANY_DOMAIN is True
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

    # Collect from <img> and common lazy attrs
    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-original", "data-lazy", "data-image"):
            add_img(img.get(attr))
        srcset = img.get("srcset")
        if srcset:
            # take all candidates from srcset (more robust)
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add_img(p)

    # <source srcset> in picture
    for src in soup.find_all("source"):
        srcset = src.get("srcset")
        if srcset:
            parts = [p.strip().split(" ")[0] for p in srcset.split(",") if p.strip()]
            for p in parts:
                add_img(p)

    # Deduplicate, then filter
    seen = set()
    uniq = []
    for u in found_abs:
        if u not in seen:
            uniq.append(u)
            seen.add(u)

    # First keep URLs with explicit image extensions
    exts = [u for u in uniq if is_ext_image_url(u)]

    # For some sites images have no extension; probe a limited number via HEAD
    if len(exts) < MAX_IMAGES and HEAD_CHECK_MAX > 0:
        candidates = [u for u in uniq if not is_ext_image_url(u)]
        head_ok = []
        for u in candidates[:HEAD_CHECK_MAX]:
            if head_is_image(u):
                head_ok.append(u)
        exts.extend(head_ok)

    # Trim to MAX_IMAGES
    return exts[:MAX_IMAGES]

# ----------- Routes -----------
@app.get("/")
def home():
    return ("Page → Random Images service\n"
            "UI: /ui\n"
            "One-shot JSON: /rand.json?url=<page_url>\n"
            "One-shot redirect: /rand?url=<page_url>\n")

@app.get("/config")
def config_info():
    return jsonify({
        "allow_any_domain": ALLOW_ANY_DOMAIN,
        "max_images": MAX_IMAGES,
        "timeout": TIMEOUT,
        "head_check_max": HEAD_CHECK_MAX,
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
    choice = random.choice(images)
    base = request.url_root.rstrip("/")
    return no_cache(jsonify({
        "source_url": page_url,
        "random_image_url": choice,
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

# ---------- UI ----------
@app.get("/ui")
def ui_page():
    html_path = os.path.join(BASE_DIR, "static", "ui", "index.html")
    if os.path.isfile(html_path):
        return send_file(html_path)
    return abort(404, description="UI file missing")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
