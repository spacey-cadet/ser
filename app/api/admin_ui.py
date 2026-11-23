"""
app/api/admin_ui.py

Adds one server-rendered HTML page to the existing FastAPI app for the
solo human labeler: list pending review-queue items, play the audio via
a presigned S3 URL, submit a label. No separate frontend build or host
-- it's mounted into the same app that already serves /predict.

Wire this in by including its router in main.py:

    from app.api.admin_ui import router as admin_ui_router
    app.include_router(admin_ui_router)

Protected by a single shared secret (checked via a query param). The
secret is resolved lazily, in this order:
  1. ADMIN_UI_SECRET env var          (HF Spaces / local docker: set it as a Space secret)
  2. SSM parameter named by ADMIN_UI_SECRET_PARAM  (AWS branch)
Fine for a one-person labeling workflow, not meant to scale past that.

Importing this module never touches AWS: boto3 clients are created on
first use, so the app boots on hosts with no AWS region or credentials.
"""
import hmac
import os
from functools import lru_cache
from html import escape
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, RedirectResponse

router = APIRouter(prefix="/admin", tags=["admin"])

# Must match scripts/eval_report.py's IDX_TO_EMOTION values exactly
# (lowercase; "fearful"/"surprised", not "fear"/"surprise").
VALID_LABELS = ("angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised")


def _region() -> str | None:
    return os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")


@lru_cache(maxsize=1)
def _get_ssm():
    import boto3

    region = _region()
    if not region:
        raise RuntimeError("AWS region not set (AWS_REGION / AWS_DEFAULT_REGION)")
    return boto3.client("ssm", region_name=region)


@lru_cache(maxsize=1)
def _get_s3():
    import boto3

    region = _region()
    if not region:
        raise RuntimeError("AWS region not set (AWS_REGION / AWS_DEFAULT_REGION)")
    return boto3.client("s3", region_name=region)


_secret_cache: str | None = None


def _get_admin_secret() -> str:
    global _secret_cache
    if _secret_cache is not None:
        return _secret_cache

    direct = os.environ.get("ADMIN_UI_SECRET")
    if direct:
        _secret_cache = direct
        return _secret_cache

    param_name = os.environ.get("ADMIN_UI_SECRET_PARAM")
    if not param_name:
        raise RuntimeError("Set ADMIN_UI_SECRET (or ADMIN_UI_SECRET_PARAM for SSM)")
    resp = _get_ssm().get_parameter(Name=param_name, WithDecryption=True)
    _secret_cache = resp["Parameter"]["Value"]
    return _secret_cache


def _require_auth(token: str = Query(...)):
    try:
        secret = _get_admin_secret()
    except Exception as e:  # not configured / SSM unreachable -> clear 503, not a 500 stack trace
        raise HTTPException(status_code=503, detail=f"admin UI not configured: {e}")
    if not hmac.compare_digest(token.encode(), secret.encode()):
        raise HTTPException(status_code=403, detail="bad token")


def _audio_url(item: dict) -> str | None:
    bucket = os.environ.get("AUDIO_BUCKET")
    key = item.get("s3_audio_key")
    if not bucket or not key:
        return None
    try:
        return _get_s3().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key},
            ExpiresIn=3600,
        )
    except Exception:
        return None


@router.get("/label", response_class=HTMLResponse)
def label_page(request: Request, token: str = Query(...), _=Depends(_require_auth)):
    # list_pending() comes from the active review-queue backend
    from app.logging_pipeline.review_queue import get_review_queue_backend

    backend = get_review_queue_backend()
    items = backend.list_pending(limit=20)
    token_q = quote(token, safe="")

    options = "".join(
        f'<option value="{lab}"{" selected" if lab == "neutral" else ""}>{lab.capitalize()}</option>'
        for lab in VALID_LABELS
    )

    rows = []
    for item in items:
        rid = escape(str(item["request_id"]))
        pred = item.get("prediction") or {}
        label = escape(str(pred.get("label", "?"))) if isinstance(pred, dict) else escape(str(pred))
        conf = escape(str(item.get("confidence", "?")))
        url = _audio_url(item)
        audio = (
            f'<audio controls src="{escape(url, quote=True)}"></audio>'
            if url
            else "<p><i>No playable audio for this item.</i></p>"
        )
        rows.append(f"""
        <div class="item" data-request-id="{rid}">
          <p><b>{rid}</b> &mdash; model said <code>{label}</code> (confidence {conf})</p>
          {audio}
          <form method="post" action="/admin/label?token={token_q}">
            <input type="hidden" name="request_id" value="{rid}">
            <select name="label">{options}</select>
            <button type="submit">Submit label</button>
          </form>
        </div>
        """)

    html = f"""
    <html><head><title>SER review queue</title>
    <style>body{{font-family:sans-serif;max-width:600px;margin:2rem auto}}
    .item{{border:1px solid #ccc;padding:1rem;margin-bottom:1rem;border-radius:8px}}</style>
    </head><body>
    <h1>Pending review ({len(items)})</h1>
    {''.join(rows) if rows else '<p>Nothing pending.</p>'}
    </body></html>
    """
    return HTMLResponse(html)


@router.post("/label")
async def submit_label(request: Request, token: str = Query(...), _=Depends(_require_auth)):
    from app.logging_pipeline.review_queue import get_review_queue_backend

    form = await request.form()
    request_id = form.get("request_id")
    label = form.get("label")
    if not request_id or label not in VALID_LABELS:
        raise HTTPException(status_code=400, detail=f"label must be one of {VALID_LABELS}")

    get_review_queue_backend().mark_labeled(request_id, label)
    return RedirectResponse(url=f"/admin/label?token={quote(token, safe='')}", status_code=303)