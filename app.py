"""Flask chat for TINY 2.1 (https://github.com/WollyMonjac/Tiny-2.1)

Run:
    pip install flask numpy
    python app.py                        # looks for dos/TINY.BIN or TINY.BIN
    python app.py --bin path/TINY.BIN --port 5000

The model is loaded directly from TINY.BIN (version 2) and runs with NumPy,
so PyTorch is not required. It reads and writes one character at a time
(ASCII 32 through 126 plus newline) and understands only English.
"""
import argparse
import os
import struct
import threading

import numpy as np
from flask import Flask, Response, jsonify, render_template, request

VOCAB, CTX, MAXQ = 97, 128, 72


# ------------------------------------------------------------------- model
class Tiny:
    def __init__(self, path):
        b = open(path, "rb").read()
        if b[:4] != b"TNY2":
            raise ValueError("Not a version 2 TINY.BIN ('TNY2' header is missing)")
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
            raise ValueError("The file size does not match the header")

    @staticmethod
    def _norm(x, g):
        return x / np.sqrt(np.mean(x * x) + 1e-5) * g

    def new_cache(self):
        return [([], []) for _ in range(self.L)]

    def step(self, tok, pos, cache):
        """Return next-character logits from one input character."""
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
    """Match TINY.EXE cleaning: printable characters, at most 72,
    no trailing spaces, and uppercase-only input becomes lowercase."""
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
    p[p < 0.15] = 0.0  # Never select unlikely characters, as in TINY.C.
    return int(rng.choice(VOCAB, p=p / p.sum()))


def answer_stream(model, question, creativity=5):
    """Generate the answer one character at a time."""
    rng = np.random.default_rng()
    cache = model.new_cache()
    p = 0
    score = None
    for ch in question:
        score = model.step(ord(ch) - 30, p, cache)
        p += 1
    score = model.step(1, p, cache)  # Newline marks the end of the question.
    p += 1
    while p < CTX:
        tok = pick(score, creativity, rng)
        if tok <= 1:
            break
        yield chr(tok + 30)
        score = model.step(tok, p, cache)
        p += 1


# --------------------------------------------------------------------- app
app = Flask(__name__)
MODEL = None
LOCK = threading.Lock()  # The model is small, but one request at a time is enough.


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/docs")
def api_docs():
    return render_template("api_docs.html")


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
    raise SystemExit("TINY.BIN was not found; provide its path with --bin")


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
