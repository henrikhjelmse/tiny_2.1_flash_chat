 ?  """Flask chat for TINY 2.1 (https://github.com/WollyMonjac/Tiny-2.1)

 Run:
    pip install flask numpy
     python app.py                      # searches for dos/TINY.BIN or TINY.BIN
     python app.py --bin path/TINY.BIN --port 5000

 The model is loaded directly from TINY.BIN (version 2) and run with NumPy,
 so PyTorch is not needed. It reads and writes one character at a time
 (ASCII 32..126 plus newline) and understands English only.
"""
import argparse
import os
import struct
import threading

import numpy as np
from flask import Flask, Response, jsonify, render_template_string, request

VOCAB, CTX, MAXQ = 97, 128, 72


# ------------------------------------------------- model
class Tiny:
    def __init__(self, path):
        b = open(path, "rb").read()
        if b[:4] != b"TNY2":
            raise ValueError("Not a TINY.BIN version 2 (missing 'TNY2' in header)")
        self.D, self.L, self.H, self.HID = struct.unpack_from("<4h", b, 4)
        off = 12

        def table(rows, cols):
            nonlocal off
            w = np.empty((rows, cols), np.float32)
            for r in range(rows):
                s = struct.unpack_from("<f", b, off)[0]
                q = np.frombuffer(b, np.int8, cols, off + 4)
                w[r] = q.astype(np.float32) * s
                off += 4 + cols
            return w

        def gains():
            nonlocal off
            g = np.frombuffer(b, "<f4", self.D, off).copy()
            off += 4 * self.D
            return g

        D, HID = self.D, self.HID
        self.emb = table(VOCAB, D)
        self.pos = table(CTX, D)
        self.blocks = []
        for _ in range(self.L):
            blk = {"n1": gains()}
            for n in "qkvo":
                blk[n] = table(D, D)
            blk["n2"] = gains()
            blk["f1"] = table(HID, D)
            blk["f2"] = table(D, HID)
            self.blocks.append(blk)
        self.nf = gains()
        if off != len(b):
            raise ValueError("File size does not match header")

    @staticmethod
    def _norm(x, g):
        return x / np.sqrt(np.mean(x * x) + 1e-5) * g

    def new_cache(self):
        return [([], []) for _ in range(self.L)]

    def step(self, tok, pos, cache):
        """One character in -> logits (scores) for next character."""
        D, H = self.D, self.H
        dh = D // H
        x = self.emb[tok] + self.pos[pos]
        for blk, (ks, vs) in zip(self.blocks, cache):
            h = self._norm(x, blk["n1"])
            ks.append(blk["k"] @ h)
            vs.append(blk["v"] @ h)
            q = (blk["q"] @ h).reshape(H, dh)
            K = np.stack(ks).reshape(len(ks), H, dh)
            V = np.stack(vs).reshape(len(vs), H, dh)
            att = np.einsum("hd,thd->ht", q, K) / np.sqrt(dh)
            att = np.exp(att - att.max(axis=1, keepdims=True))
            att /= att.sum(axis=1, keepdims=True)
            a = np.einsum("ht,thd->hd", att, V).reshape(D)
            x = x + blk["o"] @ a
            h = self._norm(x, blk["n2"])
            x = x + blk["f2"] @ np.maximum(blk["f1"] @ h, 0)
        return self._norm(x, self.nf) @ self.emb.T


def clean(line):
    """Same cleaning as TINY.EXE: only printable characters, max 72,
    no trailing spaces, ALL CAPS becomes lowercase."""
    q = "".join(c for c in line if 32 <= ord(c) <= 126)[:MAXQ].rstrip()
    if any(c.isupper() for c in q) and not any(c.islower() for c in q):
        q = q.lower()
    return q


def pick(score, creativity, rng):
    best = int(np.argmax(score))
    if creativity <= 0:
        return best
    t = creativity / 5.0
    p = np.exp((score - score[best]) / t)
    p[p < 0.15] = 0.0  # never pick unlikely characters (like in TINY.C)
    return int(rng.choice(VOCAB, p=p / p.sum()))


def answer_stream(model, question, creativity=5):
    """Generates the answer one character at a time."""
    rng = np.random.default_rng()
    cache = model.new_cache()
    p = 0
    score = None
    for ch in question:
        score = model.step(ord(ch) - 30, p, cache)
        p += 1
    score = model.step(1, p, cache)  # newline = end of question
    p += 1
    while p < CTX:
        tok = pick(score, creativity, rng)
        if tok <= 1:
            break
        yield chr(tok + 30)
        score = model.step(tok, p, cache)
        p += 1


# ------------------------------------------------- app
app = Flask(__name__)
MODEL = None
LOCK = threading.Lock()  # model is small, but one question at a time is fine

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TINY 2.1</title>
<style>
  :root { color-scheme: dark; --bg:#0b0f14; --panel:#121923; --fg:#d7e3f0;
          --accent:#39ff9c; --me:#1d2b3d; --dim:#6b7f95; }
  * { box-sizing: border-box; }
  body { margin:0; height:100vh; display:flex; flex-direction:column;
         background:var(--bg); color:var(--fg);
         font:15px/1.5 ui-monospace, Menlo, Consolas, monospace; }
  header { padding:12px 16px; border-bottom:1px solid #1c2735;
           display:flex; gap:16px; align-items:center; flex-wrap:wrap; }
  header h1 { margin:0; font-size:16px; color:var(--accent); letter-spacing:2px; }
  header small { color:var(--dim); }
  .spacer { flex:1; }
  label { color:var(--dim); font-size:13px; }
  input[type=range] { vertical-align:middle; accent-color:var(--accent); }
  #log { flex:1; overflow-y:auto; padding:16px; display:flex;
         flex-direction:column; gap:10px; }
  .msg { max-width:min(680px,85%); padding:8px 12px; border-radius:10px;
         white-space:pre-wrap; word-break:break-word; }
  .me  { align-self:flex-end; background:var(--me); }
  .bot { align-self:flex-start; background:var(--panel); border:1px solid #1c2735; }
  .bot.typing::after { content:"▌"; color:var(--accent); animation:b 1s steps(1) infinite; }
  @keyframes b { 50% { opacity:0; } }
  .hint { color:var(--dim); font-size:13px; align-self:center; text-align:center; }
  .hint span { cursor:pointer; color:var(--accent); margin:0 6px; white-space:nowrap; }
  form { display:flex; gap:8px; padding:12px 16px; border-top:1px solid #1c2735; }
  #q { flex:1; padding:10px 12px; border-radius:8px; border:1px solid #1c2735;
       background:var(--panel); color:var(--fg); font:inherit; }
  #q:focus { outline:1px solid var(--accent); }
  button { padding:10px 16px; border-radius:8px; border:0; cursor:pointer;
           background:var(--accent); color:#04130b; font:inherit; font-weight:700; }
  button:disabled { opacity:.4; cursor:default; }
  button.ghost { background:transparent; color:var(--dim); border:1px solid #1c2735;
                 padding:6px 10px; font-weight:400; }
</style>
</head>
<body>
<header>
  <h1>TINY 2.1</h1>
  <small>367,000 parameters · answers in English</small>
  <span class="spacer"></span>
  <label>Creativity <input id="cr" type="range" min="0" max="9" value="5">
    <b id="crv">5</b></label>
  <button class="ghost" id="clear" type="button">Clear</button>
</header>

<div id="log">
  <div class="hint">TINY knows about ~3,500 facts (countries, elements, animals, basic math…)
    and confidently makes things up when it doesn't know.<br>Try:
    <span>what is the capital of peru</span>
    <span>what is 7 times 8</span>
    <span>is a whale a fish</span>
    <span>spell elephant</span>
    <span>opposite of hot</span>
  </div>
</div>

<form id="f" autocomplete="off">
  <input id="q" maxlength="72" placeholder="Ask a question in English …" autofocus>
  <button id="go">Send</button>
</form>

<script>
const log = document.getElementById('log'), q = document.getElementById('q'),
      go = document.getElementById('go'), cr = document.getElementById('cr'),
      crv = document.getElementById('crv');
const history = [];
let hi = -1;

cr.oninput = () => crv.textContent = cr.value;
document.getElementById('clear').onclick = () => {
  log.querySelectorAll('.msg').forEach(e => e.remove());
};
document.querySelectorAll('.hint span').forEach(s =>
  s.onclick = () => { q.value = s.textContent; q.focus(); });

function add(cls, text) {
  const d = document.createElement('div');
  d.className = 'msg ' + cls;
  d.textContent = text;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

q.addEventListener('keydown', e => {          // arrow up/down = previous questions
  if (e.key === 'ArrowUp' && history.length) {
    hi = Math.min(hi + 1, history.length - 1);
    q.value = history[history.length - 1 - hi]; e.preventDefault();
  } else if (e.key === 'ArrowDown') {
    hi = Math.max(hi - 1, -1);
    q.value = hi < 0 ? '' : history[history.length - 1 - hi]; e.preventDefault();
  }
});

document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const text = q.value.trim();
  if (!text) return;
  history.push(text); hi = -1; q.value = '';
  add('me', text);
  const bot = add('bot typing', '');
  go.disabled = true;
  try {
    const r = await fetch('/ask', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({q: text, creativity: +cr.value})
    });
    const reader = r.body.getReader(), dec = new TextDecoder();
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      bot.textContent += dec.decode(value, {stream: true});
      log.scrollTop = log.scrollHeight;
    }
    if (!bot.textContent) bot.textContent = '(no response)';
  } catch (err) {
    bot.textContent = 'Error: ' + err;
  }
  bot.classList.remove('typing');
  go.disabled = false; q.focus();
};
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE)


@app.post("/ask")
def ask():
    data = request.get_json(silent=True) or {}
    question = clean(str(data.get("q", "")))
    try:
        creativity = max(0, min(9, int(data.get("creativity", 5))))
    except (TypeError, ValueError):
        creativity = 5
    if not question:
        return Response("", mimetype="text/plain")

    def generate():
        with LOCK:
            for ch in answer_stream(MODEL, question, creativity):
                yield ch

    return Response(generate(), mimetype="text/plain; charset=utf-8",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"})


@app.get("/info")
def info():
    return jsonify(D=MODEL.D, layers=MODEL.L, heads=MODEL.H, hidden=MODEL.HID)


def find_bin(explicit):
    here = os.path.dirname(os.path.abspath(__file__))
    for p in ([explicit] if explicit else []) + [
        os.path.join(here, "TINY.BIN"),
        os.path.join(here, "dos", "TINY.BIN"),
        os.path.join(here, "Tiny-2.1", "dos", "TINY.BIN"),
    ]:
        if p and os.path.isfile(p):
            return p
    raise SystemExit("Cannot find TINY.BIN – specify path with --bin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", help="path to TINY.BIN")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5014)
    args = ap.parse_args()
    path = find_bin(args.bin)
    MODEL = Tiny(path)
    print(f"Loaded {path}: D={MODEL.D} L={MODEL.L} H={MODEL.H} HID={MODEL.HID}")
    app.run(host=args.host, port=args.port, threaded=True)