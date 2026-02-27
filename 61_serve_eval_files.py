import argparse
import html
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _build_index(source_dir: Path) -> None:
    meta = source_dir / "ab_metadata.jsonl"
    rows = []
    if meta.exists():
        with meta.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))

    wavs = sorted(source_dir.glob("*.wav"))
    wav_names = {w.name for w in wavs}

    lines = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'><title>Qwen3-TTS Eval Files</title></head><body>",
        "<h2>Qwen3-TTS Eval Files</h2>",
        f"<p>Directory: {html.escape(str(source_dir))}</p>",
    ]

    if rows:
        lines.append("<h3>A/B Metadata</h3>")
        lines.append("<table border='1' cellpadding='6' cellspacing='0'>")
        lines.append(
            "<tr><th>id</th><th>text</th><th>teacher</th><th>student</th><th>teacher_ms</th><th>student_ms</th></tr>"
        )
        for r in rows:
            tname = Path(r["teacher_wav"]).name
            sname = Path(r["student_wav"]).name
            tlink = f"<a href='{html.escape(tname)}'>{html.escape(tname)}</a>" if tname in wav_names else html.escape(tname)
            slink = f"<a href='{html.escape(sname)}'>{html.escape(sname)}</a>" if sname in wav_names else html.escape(sname)
            lines.append(
                "<tr>"
                f"<td>{html.escape(str(r.get('id', '')))}</td>"
                f"<td>{html.escape(str(r.get('text', '')))}</td>"
                f"<td>{tlink}</td>"
                f"<td>{slink}</td>"
                f"<td>{html.escape(str(r.get('teacher_ms', '')))}</td>"
                f"<td>{html.escape(str(r.get('student_ms', '')))}</td>"
                "</tr>"
            )
        lines.append("</table>")

    lines.append("<h3>All WAV Files</h3><ul>")
    for w in wavs:
        lines.append(f"<li><a href='{html.escape(w.name)}'>{html.escape(w.name)}</a></li>")
    lines.append("</ul>")
    lines.append("</body></html>")

    (source_dir / "index.html").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="Serve eval wav files over HTTP (for RunPod download).")
    p.add_argument("--source-dir", default="/workspace/dist/eval_ab")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    source_dir = Path(args.source_dir).resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    _build_index(source_dir)

    print(f"[INFO] serving: {source_dir}")
    print(f"[INFO] open: http://{args.host}:{args.port}/")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *hargs, **hkwargs):
            super().__init__(*hargs, directory=str(source_dir), **hkwargs)

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
