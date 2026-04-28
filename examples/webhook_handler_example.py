"""verify_webhook example: validate Standard Webhooks signatures.

This script demonstrates the verification flow without standing up a
server. For an end-to-end FastAPI handler, the snippet at the bottom
shows the wiring.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time

from langchain_parallel import verify_webhook

# Set your webhook secret: export PARALLEL_WEBHOOK_SECRET="..."


def sign(payload: bytes, webhook_id: str, ts: str, secret: str) -> str:
    """Standard Webhooks: HMAC-SHA256 over `<id>.<ts>.<body>`, base64-encoded."""
    signed = f"{webhook_id}.{ts}.{payload.decode()}"
    digest = hmac.new(secret.encode(), signed.encode(), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()


def round_trip_verification() -> None:
    print("=== verify_webhook: synthetic round-trip ===")
    secret = "test_secret"  # noqa: S105 - demo
    body = b'{"run_id":"trun_abc","status":"completed"}'
    webhook_id = "msg_demo_001"
    ts = str(int(time.time()))
    sig = sign(body, webhook_id, ts, secret)

    ok = verify_webhook(
        body,
        webhook_id=webhook_id,
        webhook_timestamp=ts,
        webhook_signature=f"v1,{sig}",
        secret=secret,
    )
    print("verified:", ok)


def replay_attack_rejected() -> None:
    print("\n=== verify_webhook: rejects stale timestamp ===")
    secret = "test_secret"  # noqa: S105 - demo
    body = b"{}"
    old_ts = "1000000000"  # year 2001
    sig = sign(body, "msg", old_ts, secret)
    ok = verify_webhook(
        body,
        webhook_id="msg",
        webhook_timestamp=old_ts,
        webhook_signature=f"v1,{sig}",
        secret=secret,
    )
    print("verified:", ok, "(expected False)")


# --- FastAPI wiring (illustrative; uncomment + run with `uvicorn`) -----------
#
# from fastapi import FastAPI, Header, HTTPException, Request
#
# app = FastAPI()
#
# @app.post("/parallel/webhook")
# async def webhook(
#     request: Request,
#     webhook_id: str = Header(..., alias="webhook-id"),
#     webhook_timestamp: str = Header(..., alias="webhook-timestamp"),
#     webhook_signature: str = Header(..., alias="webhook-signature"),
# ) -> dict:
#     body = await request.body()
#     if not verify_webhook(
#         body,
#         webhook_id=webhook_id,
#         webhook_timestamp=webhook_timestamp,
#         webhook_signature=webhook_signature,
#         secret=os.environ["PARALLEL_WEBHOOK_SECRET"],
#     ):
#         raise HTTPException(status_code=401, detail="invalid signature")
#     # ... process the event
#     return {"ok": True}


def main() -> None:
    round_trip_verification()
    replay_attack_rejected()


if __name__ == "__main__":
    main()
