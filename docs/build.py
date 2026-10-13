"""Build the article from article.src.html and data.json.

article.html is the page body (the artifact host adds its own skeleton);
index.html wraps the same body in a full document for GitHub Pages."""
import json
from pathlib import Path

here = Path(__file__).parent
src = (here / "article.src.html").read_text(encoding="utf-8")
data = json.dumps(json.loads((here / "data.json").read_text(encoding="utf-8")), separators=(",", ":"))
body = src.replace("/*DATA*/", data.replace("</", "<" + chr(92) + "/"))  # keep "</" out of the inline script
(here / "article.html").write_text(body, encoding="utf-8", newline="\n")
head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
        '<style>html{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n')
title_end = body.index("</title>") + len("</title>")
page = head + body[:title_end] + "\n" + body[title_end:].replace("<main", "</head>\n<body>\n<main", 1) + "\n</body>\n</html>\n"
(here / "index.html").write_text(page, encoding="utf-8", newline="\n")
print("built", len(body), "bytes")
