"""Two trivial references that read no meaning, only surface features. A
detector that does not clearly beat them is picking up the same shortcut, not
checking the answer against the documents.

- `baseline-length`: the response's length in words. Longer answers carry more
  statements, so more chances for one to be unsupported; some datasets also
  build their hallucinated answers longer or shorter than the faithful ones.
- `baseline-novel-words`: the share of the response's content words that never
  appear in the documents (lowercased, stopwords dropped, no stemming).

Both are written as "longer / more novel = more likely hallucinated", mapped
to [0, 1] monotonically. The grader then picks each dataset's direction on its
dev rows (metrics._orient_on_dev), since a surface cue can point either way.
They are not probabilities and are graded like RAGAS/TruLens, without
calibration.
"""
from __future__ import annotations

import re

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result

_WORD = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*")
_STOPWORDS = frozenset("""
a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having
he her here hers him his how i if in into is it its itself just me more most my no nor not of off on once
only or other our ours out over own same she should so some such than that the their theirs them then
there these they this those through to too under until up very was we were what when where which while
who whom why will with would you your yours
""".split())


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Length(Detector):
    name = "baseline-length"
    workers = 1000

    def score_one(self, ex: Example) -> Result:
        n = len(_words(ex.response))
        return Result(id=ex.id, score=n / (n + 100), raw={"words": n})


class NovelWords(Detector):
    name = "baseline-novel-words"
    workers = 1000

    def score_one(self, ex: Example) -> Result:
        seen = set(_words(" ".join(ex.documents)))
        content = [w for w in _words(ex.response) if w not in _STOPWORDS]
        if not content:
            return Result(id=ex.id, score=0.0, raw={"content_words": 0})
        novel = sum(1 for w in content if w not in seen)
        return Result(id=ex.id, score=novel / len(content), raw={"content_words": len(content), "novel": novel})
