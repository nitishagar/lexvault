#!/usr/bin/env bash
# End-to-end smoke test: boot the fake backend + the LiteLLM proxy with the
# lexvault guardrail, then assert mask-upstream + restore-to-client (CS-D3).
#
# This is the gate that retroactively validates the guardrail config SHAPE
# (CS-D1) and the quickstart wiring. `litellm --config` crashes on a malformed
# guardrail block at startup, so a clean boot alone proves CS-D1; the curl
# assertions prove the round-trip.
#
# Usage: examples/ci_smoke.sh  (from repo root)
# Exits non-zero on any failure. Cleans up both background processes.

set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

PORT_BACKEND=8000
PORT_PROXY=4000
TMPDIR_WORK="$(mktemp -d)"
trap 'kill $(jobs -p) 2>/dev/null || true; rm -rf "$TMPDIR_WORK"' EXIT

PYTHON="${PYTHON:-python3}"

echo ":: Booting fake backend on :${PORT_BACKEND}"
"$PYTHON" examples/fake_backend.py >"$TMPDIR_WORK/backend.log" 2>&1 &

# Wait for the backend to accept connections.
for _ in $(seq 1 40); do
  if curl -sf -o /dev/null "http://127.0.0.1:${PORT_BACKEND}/" -X POST \
        -H "Content-Type: application/json" -d '{}' 2>/dev/null; then
    break
  fi
  sleep 0.25
done

echo ":: Booting LiteLLM proxy on :${PORT_PROXY}"
# Run litellm from the examples dir so the shim resolves next to the config.
(cd examples && litellm --config ci-smoke-config.yaml --port "$PORT_PROXY" \
    --num_workers 1 >"$TMPDIR_WORK/proxy.log" 2>&1 &
)
PROXY_SUBPID=$!

# Wait for the proxy to be ready (litellm boots the guardrail at startup — a
# malformed guardrail block crashes here, which is exactly what we're gating on).
echo ":: Waiting for proxy to become ready ..."
PROXY_READY=0
for _ in $(seq 1 120); do
  if curl -sf -o /dev/null "http://127.0.0.1:${PORT_PROXY}/health/liveliness" 2>/dev/null; then
    PROXY_READY=1
    break
  fi
  # Fail fast if the proxy process died (startup crash = config rejected).
  if ! kill -0 "$PROXY_SUBPID" 2>/dev/null && ! pgrep -f "litellm.*ci-smoke-config" >/dev/null 2>&1; then
    echo "ERROR: proxy process exited during startup. Proxy log:" >&2
    cat "$TMPDIR_WORK/proxy.log" >&2
    exit 1
  fi
  sleep 0.5
done
if [ "$PROXY_READY" -ne 1 ]; then
  echo "ERROR: proxy did not become ready. Proxy log:" >&2
  cat "$TMPDIR_WORK/proxy.log" >&2
  exit 1
fi

echo ":: Sending chat completion mentioning a dictionary term ..."
RESP="$(curl -sf "http://127.0.0.1:${PORT_PROXY}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"fake","messages":[{"role":"user","content":"What is Project Titan and EMP-123456?"}]}')"

echo ":: Client response:"
echo "$RESP"

# Assertion 1: the CLIENT sees the originals restored (no placeholders leaked).
if echo "$RESP" | grep -q '\[LEX-'; then
  echo "FAIL: client response contains an unrestored [LEX-...] placeholder" >&2
  exit 1
fi
if ! echo "$RESP" | grep -q 'Project Titan'; then
  echo "FAIL: client response missing restored original 'Project Titan'" >&2
  exit 1
fi

# Assertion 2: the UPSTREAM (fake backend) saw the masked placeholder — proving
# the original never left the boundary on the way up. The echo reports the count.
if ! echo "$RESP" | grep -q 'upstream saw.*placeholder'; then
  echo "FAIL: upstream did not report a masked placeholder (mask-up failed)" >&2
  exit 1
fi

echo ""
echo ":: PASS — mask-upstream + restore-to-client round-trip verified."
