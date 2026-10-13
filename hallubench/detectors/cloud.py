"""Managed grounding checks sold by the two largest clouds: what a team gets
without hosting or prompting anything.

- `vertex-check-grounding`: Google Cloud's Check Grounding API (Vertex AI
  Search / Agent Search, Discovery Engine). Input: the response as the
  "answer candidate" and the documents as "facts" (at most 10k characters
  each, so longer documents are cut at sentence boundaries). It does not take
  the question. Output: a support score in [0, 1]. Score = 1 - support.
  Credentials: a service-account key with the Discovery Engine User role,
  path in GOOGLE_APPLICATION_CREDENTIALS (default `gcp-key.json`), project
  in GCP_PROJECT.

- `bedrock-contextual-grounding`: AWS Bedrock Guardrails' contextual
  grounding check, called on its own through the ApplyGuardrail API (no model
  invocation). Input: the documents as the grounding source, the question as
  the query, the response as the content to guard. Output: a grounding score
  in [0, 1]. Score = 1 - grounding. The guardrail it applies holds only that
  check; it is created on first use under the name in `_GUARDRAIL_NAME`, in
  the region AWS_REGION (default us-east-1). Credentials: AWS_ACCESS_KEY_ID
  and AWS_SECRET_ACCESS_KEY. Billed per 1,000 characters of source, query
  and response together; the units AWS reports are kept as the row's cost.

Neither service exposes a model version: both can change under the same name,
which is why their raw outputs are published with the results.
"""
from __future__ import annotations

import os
import random
import threading
import time

import httpx

from hallubench.data import Example
from hallubench.detectors.base import Detector, Result
from hallubench.text import render_documents, split_sentences

_FACT_CHARS = 10_000
_MAX_FACTS = 200


def _facts(documents: list[str]) -> list[str]:
    """Each document as one fact, or as consecutive sentence-aligned pieces
    under the per-fact limit when it is longer."""
    out: list[str] = []
    for doc in documents:
        if len(doc) <= _FACT_CHARS:
            out.append(doc)
            continue
        piece = ""
        for sent in split_sentences(doc):
            while len(sent) > _FACT_CHARS:  # a "sentence" with no boundary for 10k characters
                if piece:
                    out.append(piece)
                    piece = ""
                out.append(sent[:_FACT_CHARS])
                sent = sent[_FACT_CHARS:]
            if piece and len(piece) + 1 + len(sent) > _FACT_CHARS:
                out.append(piece)
                piece = sent
            else:
                piece = f"{piece} {sent}" if piece else sent
        if piece:
            out.append(piece)
    if len(out) > _MAX_FACTS:
        raise ValueError(f"{len(out)} facts, over the API's limit of {_MAX_FACTS}")
    return out


class VertexCheckGrounding(Detector):
    name = "vertex-check-grounding"
    workers = 8

    def setup(self) -> None:
        from dotenv import load_dotenv
        from google.oauth2 import service_account

        load_dotenv()
        key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "gcp-key.json")
        self.project = os.environ.get("GCP_PROJECT")
        if not self.project or not os.path.exists(key):
            raise SystemExit("set GCP_PROJECT and a service-account key (GOOGLE_APPLICATION_CREDENTIALS) in .env")
        self.creds = service_account.Credentials.from_service_account_file(
            key, scopes=["https://www.googleapis.com/auth/cloud-platform"])
        self.lock = threading.Lock()
        self.http = httpx.Client(timeout=120)
        self.url = (f"https://discoveryengine.googleapis.com/v1/projects/{self.project}/locations/global/"
                    "groundingConfigs/default_grounding_config:check")

    def _token(self) -> str:
        from google.auth.transport.requests import Request

        with self.lock:  # one refresh for all worker threads
            if not self.creds.valid:
                self.creds.refresh(Request())
            return self.creds.token

    def score_one(self, ex: Example) -> Result:
        body = {
            "answerCandidate": ex.response,
            "facts": [{"factText": f} for f in _facts(ex.documents)],
        }
        delay = 2.0
        for attempt in range(6):
            t0 = time.monotonic()
            r = self.http.post(self.url, json=body, headers={
                "Authorization": f"Bearer {self._token()}", "X-Goog-User-Project": self.project})
            latency = (time.monotonic() - t0) * 1000
            if r.status_code == 200:
                break
            if attempt == 5 or not (r.status_code == 429 or r.status_code >= 500):
                raise RuntimeError(f"{r.status_code} {r.text[:300]}")
            time.sleep(delay * (1 + random.random()))
            delay *= 2
        data = r.json()
        claims = data.get("claims", [])
        checkable = [c for c in claims if c.get("groundingCheckRequired", True)]
        if "supportScore" in data:
            support = float(data["supportScore"])
        elif claims and not checkable:
            # The service found nothing to check ("True.", "The context does
            # not say."), returns no score and so flags nothing: in production
            # the answer goes through, which is what it is graded as here.
            support = 1.0
        else:
            raise ValueError(f"no supportScore in response: {str(data)[:300]}")
        return Result(id=ex.id, score=1.0 - support, latency_ms=latency,
                      raw={"support": support, "scored_by_service": "supportScore" in data,
                           "facts": len(body["facts"]), "claims": len(claims)})


_GUARDRAIL_NAME = "rag-hallu-bench-grounding"
_USD_PER_UNIT = 0.10 / 1000  # contextual grounding, us-east-1 list price


class BedrockContextualGrounding(Detector):
    name = "bedrock-contextual-grounding"
    workers = 8

    def setup(self) -> None:
        import boto3
        from dotenv import load_dotenv

        load_dotenv()
        key, secret = os.environ.get("AWS_ACCESS_KEY_ID"), os.environ.get("AWS_SECRET_ACCESS_KEY")
        if not key or not secret:
            # never fall back to whatever profile the machine has configured
            raise SystemExit("set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env")
        session = boto3.Session(aws_access_key_id=key, aws_secret_access_key=secret,
                                region_name=os.environ.get("AWS_REGION", "us-east-1"))
        control = session.client("bedrock")
        self.runtime = session.client("bedrock-runtime")
        self.guardrail_id = self._guardrail(control)

    @staticmethod
    def _guardrail(control) -> str:
        """The benchmark's guardrail, created once. Its threshold only decides
        whether AWS would block; the score is returned either way."""
        for page in control.get_paginator("list_guardrails").paginate():
            for g in page["guardrails"]:
                if g["name"] == _GUARDRAIL_NAME:
                    return g["id"]
        created = control.create_guardrail(
            name=_GUARDRAIL_NAME,
            description="Contextual grounding only, for the rag-hallu-bench benchmark",
            contextualGroundingPolicyConfig={"filtersConfig": [{"type": "GROUNDING", "threshold": 0.5}]},
            blockedInputMessaging="blocked",
            blockedOutputsMessaging="blocked",
        )
        return created["guardrailId"]

    def score_one(self, ex: Example) -> Result:
        content = [
            {"text": {"text": render_documents(ex.documents), "qualifiers": ["grounding_source"]}},
            {"text": {"text": ex.question, "qualifiers": ["query"]}},
            {"text": {"text": ex.response, "qualifiers": ["guard_content"]}},
        ]
        delay = 2.0
        for attempt in range(6):
            t0 = time.monotonic()
            try:
                resp = self.runtime.apply_guardrail(guardrailIdentifier=self.guardrail_id, guardrailVersion="DRAFT",
                                                    source="OUTPUT", content=content, outputScope="FULL")
                break
            except self.runtime.exceptions.ThrottlingException:
                if attempt == 5:
                    raise
                time.sleep(delay * (1 + random.random()))
                delay *= 2
        latency = (time.monotonic() - t0) * 1000
        filters = [f for a in resp.get("assessments", [])
                   for f in a.get("contextualGroundingPolicy", {}).get("filters", []) if f["type"] == "GROUNDING"]
        if not filters:
            raise ValueError(f"no grounding score in response: {str(resp.get('assessments'))[:300]}")
        grounding = float(filters[0]["score"])
        units = resp.get("usage", {}).get("contextualGroundingPolicyUnits")
        return Result(id=ex.id, score=1.0 - grounding, latency_ms=latency,
                      cost_usd=None if units is None else units * _USD_PER_UNIT,
                      raw={"grounding": grounding, "action": filters[0].get("action"), "units": units})
