"""Protocol layer: the A2A 1.0 HTTP+JSON surface.

Run:  uvicorn main:app --host 0.0.0.0 --port $PORT      (ONE worker only)
"""
import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, Request, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

import brain
import store

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("a2a")

# The EXACT base URL you submit on the assignment page, e.g. https://my-agent.onrender.com/a2a/
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000/a2a/")
PREFIX = urlparse(BASE_URL).path.rstrip("/")
# Optional allow-list. Leave EMPTY for the grader: it uses several different tokens (owner + outsider).
ALLOWED_TOKENS = {t.strip() for t in os.getenv("AUTH_TOKENS", "").split(",") if t.strip()}

A2A_MT = "application/a2a+json"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"

S_INPUT = "TASK_STATE_INPUT_REQUIRED"
S_DONE = "TASK_STATE_COMPLETED"
S_CANCELED = "TASK_STATE_CANCELED"

MAX_BODY = 3_000_000
LIST_BUDGET = 400_000          # keep list responses well under 512 KiB

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None, redirect_slashes=False)
router = APIRouter(prefix=PREFIX)
INFLIGHT = {}                  # (principal, messageId) -> (hash, asyncio task)  => concurrent dedupe


# ------------------------------------------------------------------ helpers
def now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def reply(obj, status=200, media_type=A2A_MT):
    body = json.dumps(obj, separators=(",", ":"), ensure_ascii=False)
    if len(body) > 500_000:
        log.warning("large response: %d bytes", len(body))
    return Response(body, status_code=status, media_type=media_type)


class ApiError(Exception):
    def __init__(self, status, code, message, headers=None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def error_response(status, code, message, headers=None):
    body = {"error": {"code": status, "status": code, "message": message}, "code": code}
    return Response(json.dumps(body), status_code=status, media_type=A2A_MT, headers=headers)


@app.exception_handler(ApiError)
async def _api_error(_req, exc):
    return error_response(exc.status, exc.code, exc.message, exc.headers)


@app.exception_handler(StarletteHTTPException)
async def _http_error(_req, exc):
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "ERROR")
    return error_response(exc.status_code, code, "Request cannot be served.")


@app.exception_handler(Exception)
async def _unhandled(_req, exc):
    log.exception("unhandled error")
    return error_response(500, "INTERNAL", "Internal error.")


def guard(request: Request) -> str:
    """Auth + version. Returns the principal (a hash of the token, so we never store raw tokens)."""
    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    token = token.strip()
    if scheme != "Bearer" or not token or (ALLOWED_TOKENS and token not in ALLOWED_TOKENS):
        raise ApiError(401, "UNAUTHENTICATED", "Authentication required.", {"WWW-Authenticate": "Bearer"})
    if request.headers.get("a2a-version") != "1.0":
        raise ApiError(400, "VERSION_NOT_SUPPORTED", "Only A2A-Version 1.0 is supported.")
    return hashlib.sha256(token.encode()).hexdigest()[:32]


def check_content_type(request: Request):
    ct = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if ct not in (A2A_MT, "application/json"):
        raise ApiError(415, "UNSUPPORTED_MEDIA_TYPE", "Use Content-Type: application/a2a+json.")


async def read_json(request: Request) -> dict:
    raw = await request.body()
    if len(raw) > MAX_BODY:
        raise ApiError(413, "PAYLOAD_TOO_LARGE", "Request body too large.")
    try:
        data = json.loads(raw)
    except ValueError:
        raise ApiError(400, "INVALID_ARGUMENT", "Malformed JSON.")
    if not isinstance(data, dict):
        raise ApiError(400, "INVALID_ARGUMENT", "JSON object expected.")
    return data


def bad(msg):
    return ApiError(400, "INVALID_ARGUMENT", msg)


def conflict():
    return ApiError(409, "IDEMPOTENCY_CONFLICT", "IDEMPOTENCY_CONFLICT: messageId already used with different content.")


def read_message(body: dict) -> dict:
    m = body.get("message")
    if not isinstance(m, dict):
        raise bad("message is required.")
    mid = m.get("messageId")
    if not isinstance(mid, str) or not mid.strip() or len(mid) > 256:
        raise bad("messageId is required.")
    if m.get("role") != "ROLE_USER":
        raise bad("role must be ROLE_USER.")
    parts = m.get("parts")
    if not isinstance(parts, list) or not parts or not all(isinstance(p, dict) for p in parts):
        raise bad("parts is required.")
    return m


def one_part_data(message: dict, media_type: str) -> dict:
    hits = [p for p in message["parts"] if p.get("mediaType") == media_type]
    if not hits:
        raise ApiError(415, "UNSUPPORTED_CONTENT_TYPE", "No supported part in message.")
    if len(hits) != 1 or not isinstance(hits[0].get("data"), dict):
        raise bad("Expected exactly one part with object data.")
    return hits[0]["data"]


def is_terminal(task):
    return task["status"]["state"] != S_INPUT


# ------------------------------------------------------------------ agent card (public)
@app.get("/.well-known/agent-card.json")
async def agent_card():
    card = {
        "name": "Invoice Action Agent",
        "description": "Reads invoice packages, proposes one safe business action per invoice with cited evidence, "
                       "and only records actions the grader accepted.",
        "version": "1.0.0",
        "supportedInterfaces": [{"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}],
        "capabilities": {"streaming": False, "pushNotifications": False},
        "defaultInputModes": [BATCH_MT],
        "defaultOutputModes": [PROPOSALS_MT, RECEIPTS_MT],
        "skills": [{
            "id": "invoice_action_agent",
            "name": "Invoice action agent",
            "description": "Chooses settle, approve, hold, reject-duplicate or exception for each invoice package.",
            "tags": ["invoice", "accounts-payable", "approval", "duplicate-detection"],
            "inputModes": [BATCH_MT],
            "outputModes": [PROPOSALS_MT, RECEIPTS_MT],
        }],
    }
    return reply(card, media_type="application/json")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


# ------------------------------------------------------------------ message:send
@router.post("/message:send")
async def message_send(request: Request):
    principal = guard(request)
    check_content_type(request)
    message = read_message(await read_json(request))
    mid, h = message["messageId"], store.sha(message)     # hash of `message` ONLY (configuration ignored)
    key = (principal, mid)

    prior = store.get_message(principal, mid)             # already finished before?
    if prior:
        if prior[0] != h:
            raise conflict()
        task = store.get_task(principal, prior[1])
        if task is None:
            raise ApiError(404, "NOT_FOUND", "Task not found.")
        return reply({"task": task})

    flight = INFLIGHT.get(key)                            # same message still being processed?
    if flight:
        if flight[0] != h:
            raise conflict()
        return reply({"task": await asyncio.shield(flight[1])})

    if message.get("taskId"):                             # a result continuation (no awaits => atomic)
        return reply({"task": continue_task(principal, message, h)})

    fut = asyncio.ensure_future(create_task(principal, message, h))
    INFLIGHT[key] = (h, fut)
    fut.add_done_callback(lambda _f, k=key: INFLIGHT.pop(k, None))
    return reply({"task": await asyncio.shield(fut)})


async def create_task(principal, message, msg_hash):
    data = one_part_data(message, BATCH_MT)
    batch_id, packages = data.get("batchId"), data.get("packages")
    if not isinstance(batch_id, str) or not batch_id:
        raise bad("batchId is required.")
    if not isinstance(packages, list) or not packages or len(packages) > 200:
        raise bad("packages must be a non-empty list.")
    ids = []
    for p in packages:
        if not isinstance(p, dict) or not isinstance(p.get("packageId"), str) or not p["packageId"]:
            raise bad("every package needs a packageId.")
        ids.append(p["packageId"])
    if len(set(ids)) != len(ids):
        raise bad("packageId values must be unique.")

    try:
        decisions = await brain.decide(packages, str(data.get("policyRevision", "")))
    except brain.BrainError as e:
        log.error("brain failed: %s", e)
        raise ApiError(503, "DECISION_UNAVAILABLE", "Decision service unavailable, please retry.")

    task_id = str(uuid.uuid4())
    context_id = message.get("contextId") or str(uuid.uuid4())
    proposals = []
    for pkg, dec in zip(packages, decisions):
        action_id = "act-" + hashlib.sha256(
            f"{batch_id}|{pkg['packageId']}|{brain.content_key(pkg)}".encode()).hexdigest()[:24]
        proposals.append({
            "packageId": pkg["packageId"], "actionId": action_id, "action": dec["action"],
            "facts": dec["facts"], "evidenceRefs": brain.public_refs(dec), "rationale": brain.rationale(dec),
        })

    stored_msg = dict(message, taskId=task_id, contextId=context_id)
    task = {
        "id": task_id, "contextId": context_id,
        "status": {"state": S_INPUT, "timestamp": now()},
        "artifacts": [{
            "artifactId": f"proposals-{task_id}", "name": "invoice-action-proposals",
            "parts": [{"mediaType": PROPOSALS_MT, "data": {"batchId": batch_id, "proposals": proposals}}],
        }],
        "history": [stored_msg],
    }
    store.insert_task(principal, task, batch_id, message["messageId"], msg_hash)   # save BEFORE replying
    return task


# ------------------------------------------------------------------ continuation (grader's results)
def continue_task(principal, message, msg_hash):
    row = store.get_task(principal, str(message["taskId"]))     # only this user's tasks
    if row is None:
        raise ApiError(404, "NOT_FOUND", "Task not found.")
    if message.get("contextId") != row["contextId"]:
        raise bad("Context does not match.")
    if is_terminal(row):
        raise ApiError(409, "TASK_NOT_ACTIVE", "Task can no longer accept results.")

    data = one_part_data(message, RESULTS_MT)
    prop_data = row["artifacts"][0]["parts"][0]["data"]
    if data.get("batchId") != prop_data["batchId"]:
        raise bad("Batch does not match.")
    results = data.get("results")
    if not isinstance(results, list):
        raise bad("results must be a list.")

    by_pkg = {p["packageId"]: p for p in prop_data["proposals"]}
    order = {p["packageId"]: i for i, p in enumerate(prop_data["proposals"])}
    seen, executions = set(), []
    for r in results:
        if not isinstance(r, dict):
            raise bad("Invalid result entry.")
        pid = r.get("packageId")
        prop = by_pkg.get(pid) if isinstance(pid, str) else None
        if prop is None or pid in seen:
            raise bad("Unknown or repeated package.")
        seen.add(pid)
        if r.get("actionId") != prop["actionId"] or r.get("action") != prop["action"]:
            raise bad("Action does not match the stored proposal.")
        outcome = r.get("outcome")
        if outcome not in ("ACCEPTED", "REJECTED"):
            raise bad("Invalid outcome.")
        if outcome == "ACCEPTED":
            nonce = r.get("receiptNonce")
            if not isinstance(nonce, str) or not nonce:
                raise bad("Accepted result needs a receiptNonce.")
            executions.append({                       # facts + evidence come from OUR stored proposal
                "packageId": pid, "actionId": prop["actionId"], "action": prop["action"],
                "receiptNonce": nonce, "facts": prop["facts"], "evidenceRefs": prop["evidenceRefs"],
            })
    executions.sort(key=lambda e: order[e["packageId"]])

    stored_msg = dict(message, taskId=row["id"], contextId=row["contextId"])
    row["history"].append(stored_msg)
    row["artifacts"].append({
        "artifactId": f"receipts-{row['id']}", "name": "invoice-action-receipts",
        "parts": [{"mediaType": RECEIPTS_MT, "data": {"batchId": prop_data["batchId"], "executions": executions}}],
    })
    row["status"] = {"state": S_DONE, "timestamp": now()}
    try:
        store.transition(principal, row, executions, message["messageId"], msg_hash)
    except store.StateConflict:
        raise ApiError(409, "TASK_NOT_ACTIVE", "Task can no longer accept results.")
    return row


# ------------------------------------------------------------------ reads, list, cancel
@router.get("/tasks")
async def list_tasks(request: Request):
    principal = guard(request)
    q = request.query_params
    try:
        size = max(1, min(int(q.get("pageSize") or 50), 100))
        offset = max(0, int(q.get("pageToken") or 0))
        hist = max(0, int(q.get("historyLength") or 0))
    except ValueError:
        raise bad("Invalid paging parameter.")
    with_artifacts = (q.get("includeArtifacts") or "").lower() == "true"
    rows, total = store.list_tasks(principal, size, offset)
    out, used = [], 0
    for t in rows:
        t = dict(t)
        if not with_artifacts:
            t.pop("artifacts", None)
        if hist > 0:
            t["history"] = t["history"][-hist:]
        else:
            t.pop("history", None)
        n = len(json.dumps(t))
        if out and used + n > LIST_BUDGET:
            break
        out.append(t)
        used += n
    nxt = offset + len(out)
    return reply({"tasks": out, "nextPageToken": str(nxt) if nxt < total else "",
                  "pageSize": len(out), "totalSize": total})


@router.post("/tasks/{task_id}:cancel")
async def cancel_task(task_id: str, request: Request):
    principal = guard(request)
    row = store.get_task(principal, task_id)
    if row is None:
        raise ApiError(404, "NOT_FOUND", "Task not found.")
    if is_terminal(row):
        raise ApiError(409, "TASK_NOT_CANCELABLE", "Task is already finished.")
    row["status"] = {"state": S_CANCELED, "timestamp": now()}
    try:
        store.transition(principal, row)
    except store.StateConflict:
        raise ApiError(409, "TASK_NOT_CANCELABLE", "Task is already finished.")
    return reply(row)


@router.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request):
    principal = guard(request)
    row = store.get_task(principal, task_id)
    if row is None:
        raise ApiError(404, "NOT_FOUND", "Task not found.")
    hl = request.query_params.get("historyLength")
    if hl is not None:
        try:
            n = max(0, int(hl))
        except ValueError:
            raise bad("Invalid historyLength.")
        row["history"] = row["history"][-n:] if n else []
    return reply(row)


app.include_router(router)
