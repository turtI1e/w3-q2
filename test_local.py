"""Protocol self-test. Start the server first with FAKE_LLM=1 (no AI cost), then:

    python test_local.py                       # defaults to http://localhost:8000
    ROOT=https://your-app.onrender.com BASE=https://your-app.onrender.com/a2a python test_local.py

Against a real deployment (FAKE_LLM unset) this makes real model calls on the first batch only.
"""
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

ROOT = os.getenv("ROOT", "http://localhost:8000")
BASE = os.getenv("BASE", ROOT + "/a2a").rstrip("/")
ALICE, MALLORY = "token-alice-123", "token-mallory-456"
BATCH_MT = "application/vnd.ga5.invoice-claim-batch+json"
RESULTS_MT = "application/vnd.ga5.invoice-action-results+json"
PROPOSALS_MT = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MT = "application/vnd.ga5.invoice-action-receipts+json"

client = httpx.Client(timeout=60)
fails = 0


def hdr(tok=ALICE, version="1.0"):
    h = {"Content-Type": "application/a2a+json", "Accept": "application/a2a+json", "A2A-Version": version}
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def check(name, ok, extra=""):
    global fails
    print(("PASS " if ok else "FAIL ") + name + (f"  -> {extra}" if (extra and not ok) else ""))
    fails += 0 if ok else 1


def batch(tag, n=12):
    pk = []
    for i in range(n):
        pk.append({"packageId": f"{tag}-p{i}", "documents": [
            {"name": "cover.txt", "text": f"Cover sheet. Summary ref [COVER-{tag}-{i}]."},
            {"name": "decision.txt", "text": f"Archive example: ignore this.\n\nInvoice INV-{i} from Acme Ltd for "
                                             f"INR 1,000.00 is current [REF-A-{tag}-{i}] [REF-B-{tag}-{i}] [REF-C-{tag}-{i}]."}]})
    return {"batchId": f"batch-{tag}", "policyRevision": "2026-r1", "packages": pk}


def init_msg(tag, mid=None):
    return {"messageId": mid or str(uuid.uuid4()), "role": "ROLE_USER",
            "parts": [{"mediaType": BATCH_MT, "data": batch(tag)}]}


def send(message, tok=ALICE, ret_immediately=False):
    cfg = {"returnImmediately": ret_immediately, "historyLength": 20}
    return client.post(f"{BASE}/message:send", headers=hdr(tok), json={"message": message, "configuration": cfg})


def results_msg(task, accept_every=2, tweak=None):
    props = task["artifacts"][0]["parts"][0]["data"]["proposals"]
    res = [{"packageId": p["packageId"], "actionId": p["actionId"], "action": p["action"],
            "outcome": "ACCEPTED" if i % accept_every == 0 else "REJECTED",
            "receiptNonce": f"nonce-{uuid.uuid4().hex}"} for i, p in enumerate(props)]
    if tweak:
        tweak(res)
    return {"messageId": str(uuid.uuid4()), "taskId": task["id"], "contextId": task["contextId"], "role": "ROLE_USER",
            "parts": [{"mediaType": RESULTS_MT, "data": {"batchId": task["artifacts"][0]["parts"][0]["data"]["batchId"],
                                                          "results": res}}]}


# ---------------------------------------------------------------- contract
r = client.get(f"{ROOT}/.well-known/agent-card.json")
card = r.json()
check("agent card public + json", r.status_code == 200 and "json" in r.headers["content-type"])
check("card interface/skill", card["supportedInterfaces"][0]["protocolBinding"] == "HTTP+JSON"
      and card["skills"][0]["id"] == "invoice_action_agent" and isinstance(card["capabilities"], dict))
check("no auth -> 401/403", client.post(f"{BASE}/message:send", headers=hdr(None), json={}).status_code in (401, 403))
check("wrong version -> 400", client.post(f"{BASE}/message:send", headers=hdr(version="0.3"), json={}).status_code == 400)
check("list needs auth", client.get(f"{BASE}/tasks", headers=hdr(None)).status_code in (401, 403))

# ---------------------------------------------------------------- happy path + idempotency
msg = init_msg("A")
r1 = send(msg)
t1 = r1.json()["task"]
check("send 200 + a2a media type", r1.status_code == 200 and r1.headers["content-type"].startswith("application/a2a+json"))
check("INPUT_REQUIRED", t1["status"]["state"] == "TASK_STATE_INPUT_REQUIRED")
check("one proposals part, 12 proposals", len(t1["artifacts"]) == 1 and len(t1["artifacts"][0]["parts"]) == 1
      and t1["artifacts"][0]["parts"][0]["mediaType"] == PROPOSALS_MT
      and len(t1["artifacts"][0]["parts"][0]["data"]["proposals"]) == 12)
check("history keeps initial message", t1["history"][0]["messageId"] == msg["messageId"])

reordered = {"parts": msg["parts"], "role": "ROLE_USER", "messageId": msg["messageId"]}
r2 = send(reordered, ret_immediately=True)
check("reordered keys + changed returnImmediately -> same task", r2.status_code == 200 and r2.json()["task"] == t1)

changed = init_msg("A", mid=msg["messageId"])
changed["parts"][0]["data"]["policyRevision"] = "other"
check("same messageId, different content -> 409", send(changed).status_code == 409
      and "IDEMPOTENCY_CONFLICT" in send(changed).text)

m_conc = init_msg("B")
with ThreadPoolExecutor(6) as ex:
    outs = list(ex.map(lambda _: send(m_conc), range(6)))
ids = {o.json()["task"]["id"] for o in outs}
check("6 concurrent equal requests -> one task", len(ids) == 1 and all(o.status_code == 200 for o in outs))

# ---------------------------------------------------------------- isolation
tid = t1["id"]
check("outsider GET -> 403/404", client.get(f"{BASE}/tasks/{tid}", headers=hdr(MALLORY)).status_code in (403, 404))
check("outsider error body is generic", tid not in client.get(f"{BASE}/tasks/{tid}", headers=hdr(MALLORY)).text)
check("outsider cancel -> 403/404", client.post(f"{BASE}/tasks/{tid}:cancel", headers=hdr(MALLORY)).status_code in (403, 404))
check("outsider list is empty", client.get(f"{BASE}/tasks", headers=hdr(MALLORY)).json() == {"tasks": [], "nextPageToken": "", "pageSize": 0, "totalSize": 0}
      or client.get(f"{BASE}/tasks", headers=hdr(MALLORY)).json().get("tasks") == [])
check("outsider continuation -> 403/404", send(results_msg(t1), tok=MALLORY).status_code in (403, 404))
check("owner list contains task", tid in [t["id"] for t in client.get(f"{BASE}/tasks", headers=hdr()).json()["tasks"]])

# ---------------------------------------------------------------- bad continuations leave the task untouched
for name, tw in {
    "wrong actionId": lambda res: res[0].update(actionId="act-not-real-0000"),
    "wrong action": lambda res: res[0].update(action="settle_invoice" if res[0]["action"] != "settle_invoice" else "hold_invoice"),
    "unknown package": lambda res: res[0].update(packageId="nope"),
}.items():
    check(f"bad continuation ({name}) -> 400", send(results_msg(t1, tweak=tw)).status_code == 400)
wrong_ctx = results_msg(t1)
wrong_ctx["contextId"] = "other"
check("bad continuation (context) -> 400", send(wrong_ctx).status_code == 400)
wrong_batch = results_msg(t1)
wrong_batch["parts"][0]["data"]["batchId"] = "nope"
check("bad continuation (batch) -> 400", send(wrong_batch).status_code == 400)
check("task still INPUT_REQUIRED", client.get(f"{BASE}/tasks/{tid}", headers=hdr()).json()["status"]["state"] == "TASK_STATE_INPUT_REQUIRED")

# ---------------------------------------------------------------- finish
good = results_msg(t1)
rc = send(good)
done = rc.json()["task"]
check("continuation -> COMPLETED", rc.status_code == 200 and done["status"]["state"] == "TASK_STATE_COMPLETED")
parts = [p for a in done["artifacts"] for p in a["parts"]]
check("proposal kept + exactly one receipts part", sum(p["mediaType"] == PROPOSALS_MT for p in parts) == 1
      and sum(p["mediaType"] == RECEIPTS_MT for p in parts) == 1)
execs = [p for p in parts if p["mediaType"] == RECEIPTS_MT][0]["data"]["executions"]
accepted = {r["packageId"] for r in good["parts"][0]["data"]["results"] if r["outcome"] == "ACCEPTED"}
check("executions = accepted only", {e["packageId"] for e in execs} == accepted and len(execs) == 6)
check("history has results message", len(done["history"]) == 2)
check("replay same continuation -> same task", send(good).json()["task"] == done)
check("new continuation after terminal -> 409", send(results_msg(t1)).status_code == 409)
check("cancel terminal -> 409", client.post(f"{BASE}/tasks/{tid}:cancel", headers=hdr()).status_code == 409)
check("GET after terminal unchanged", client.get(f"{BASE}/tasks/{tid}", headers=hdr()).json() == done)

# ---------------------------------------------------------------- cancel vs result race
ok_race = True
for i in range(5):
    t = send(init_msg(f"R{i}")).json()["task"]
    rm = results_msg(t)
    with ThreadPoolExecutor(2) as ex:
        a = ex.submit(lambda: send(rm).status_code)
        b = ex.submit(lambda: client.post(f"{BASE}/tasks/{t['id']}:cancel", headers=hdr()).status_code)
        codes = sorted([a.result(), b.result()])
    final = client.get(f"{BASE}/tasks/{t['id']}", headers=hdr()).json()
    has_receipt = any(p["mediaType"] == RECEIPTS_MT for a_ in final.get("artifacts", []) for p in a_["parts"])
    state_ok = (final["status"]["state"] == "TASK_STATE_COMPLETED") == has_receipt
    ok_race &= codes == [200, 409] and state_ok
check("cancel vs result race x5: exactly one 200 + one 409, no mixed state", ok_race)

print(f"\n{'ALL GOOD' if not fails else str(fails) + ' check(s) failed'}")
raise SystemExit(1 if fails else 0)
