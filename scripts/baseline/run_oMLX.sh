#!/usr/bin/env bash
# Baseline measurement using the oMLX Python server.
#
# Usage:
#   ./run_oMLX.sh <model_dir> <prompt_file> <max_tokens> [cpu|gpu]
#
# Appends one row to metrics/baseline.csv (cross-backend schema).
#
# Strategy:
#   1. Install oMLX into a temporary venv if not already available.
#   2. Start the oMLX server with the model, wait up to 60 s for readiness.
#   3. Send a chat-completion request, capture timing + output.
#   4. Record peak RSS of the server process.
#   5. Stop the server; write the CSV row.
#
# If oMLX fails to start within 60 s, writes OMLX_LAUNCH_FAILED row.

set -euo pipefail

MODEL_DIR="${1:?model_dir required}"
PROMPT_FILE="${2:?prompt_file required}"
MAX_TOKENS="${3:?max_tokens required}"
DEVICE="${4:-gpu}"

CSV="metrics/baseline.csv"
BACKEND="oMLX"
PROMPT_LABEL="$(basename "$PROMPT_FILE")"
MODEL_BASENAME="$(basename "$MODEL_DIR")"
OMLX_PORT=8099   # use a non-default port to avoid clash with user's oMLX
OMLX_DIR="${OMLX_ROOT:-../oMLX}"
OMLX_AUTH="Bearer 1234"
MLX_LM_PYTHON="${MLX_LM_ROOT:-../mlx-lm}/.venv/bin/python3"

mkdir -p metrics
if [ ! -s "$CSV" ]; then
    printf 'run_id,timestamp_utc,backend,model_basename,quantization_type,context_size,prompt,device,prompt_tokens,load_ms,ttft_ms,tps,peak_rss_mb,output_first_50\n' >> "$CSV"
fi

python3 - "$MODEL_DIR" "$PROMPT_FILE" "$MAX_TOKENS" "$DEVICE" "$CSV" "$BACKEND" "$PROMPT_LABEL" "$MODEL_BASENAME" "$OMLX_DIR" "$OMLX_PORT" "$OMLX_AUTH" "$MLX_LM_PYTHON" << 'PYEOF'
import csv, io, json, os, subprocess, sys, time, signal
from datetime import datetime, timezone

(model_dir, prompt_file, max_tokens, device, csv_path, backend,
 prompt_label, model_basename, omlx_dir, omlx_port, omlx_auth,
 mlx_lm_python) = sys.argv[1:]
omlx_port = int(omlx_port)

def ts_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')

run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-omlx'

def get_quant_type(md):
    try:
        cfg = json.load(open(os.path.join(md, 'config.json')))
    except Exception:
        return ''
    q = cfg.get('quantization', {})
    mode, bits, gs = q.get('mode',''), q.get('bits',''), q.get('group_size','')
    if mode:
        return f'{mode} g{gs} b{bits}'
    elif bits:
        return f'affine g{gs} b{bits}'
    return ''

def get_context_size(md):
    try:
        cfg = json.load(open(os.path.join(md, 'config.json')))
    except Exception:
        return 0
    return cfg.get('text_config', {}).get('max_position_embeddings', 0)

quant_type = get_quant_type(model_dir)
context_size = get_context_size(model_dir)

def append_row(**kw):
    fields = ['run_id','timestamp_utc','backend','model_basename','quantization_type',
              'context_size','prompt','device','prompt_tokens','load_ms','ttft_ms',
              'tps','peak_rss_mb','output_first_50']
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    w.writerow(kw)
    with open(csv_path, 'a') as f:
        f.write(buf.getvalue())

def fail(reason):
    append_row(
        run_id=run_id, timestamp_utc=ts_utc(), backend=backend,
        model_basename=model_basename, quantization_type=quant_type,
        context_size=context_size, prompt=prompt_label, device=device,
        prompt_tokens=0, load_ms=0, ttft_ms=0, tps='0.000',
        peak_rss_mb=0.0, output_first_50=reason,
    )
    print(f'{backend}: {model_basename} {reason}')
    sys.exit(0)

# Find python with oMLX available
import importlib.util, shutil

# Try to find oMLX-capable python: mlx-lm venv likely has mlx
# mlx_lm_python is resolved by the shell (MLX_LM_ROOT:-../mlx-lm) and passed as argv.
omlx_python = None
for py in [mlx_lm_python, shutil.which('python3') or 'python3']:
    if not py or not os.path.exists(py):
        continue
    # Check if omlx is importable from this python
    r = subprocess.run([py, '-c', f'import sys; sys.path.insert(0, "{omlx_dir}"); import omlx'], capture_output=True)
    if r.returncode == 0:
        omlx_python = py
        break

if omlx_python is None:
    fail('OMLX_LAUNCH_FAILED: no python with oMLX importable found')

# Start oMLX server
import tempfile
log_file = tempfile.NamedTemporaryFile(suffix='.log', delete=False, mode='w')
log_file.close()

env = os.environ.copy()
env['PYTHONPATH'] = omlx_dir + ':' + env.get('PYTHONPATH', '')
# Use model_dir as the model-dir (oMLX scans a directory of models)
model_parent = os.path.dirname(os.path.abspath(model_dir))

t_server_start = time.time()
server_proc = subprocess.Popen(
    [omlx_python, '-m', 'omlx.cli', 'serve',
     '--model-dir', model_parent,
     '--port', str(omlx_port),
     '--host', '127.0.0.1',
     '--api-key', '1234'],
    stdout=open(log_file.name, 'a'),
    stderr=subprocess.STDOUT,
    env=env,
)

# Poll /v1/models for up to 60 s
import urllib.request, urllib.error
url = f'http://127.0.0.1:{omlx_port}/v1/models'
headers = {'Authorization': omlx_auth}
ready = False
for _ in range(60):
    time.sleep(1.0)
    if server_proc.poll() is not None:
        # Server crashed
        server_proc.wait()
        break
    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=2)
        if resp.status == 200:
            ready = True
            break
    except Exception:
        pass

if not ready:
    try:
        server_proc.terminate()
        server_proc.wait(timeout=5)
    except Exception:
        pass
    fail('OMLX_LAUNCH_FAILED')

load_ms = round((time.time() - t_server_start) * 1000)
prompt_text = open(prompt_file).read().strip()

# Send chat completion request
import urllib.parse
req_body = json.dumps({
    'model': model_basename,
    'messages': [{'role': 'user', 'content': prompt_text}],
    'max_tokens': int(max_tokens),
    'temperature': 0.0,
    'stream': False,
}).encode()

t_req_start = time.time()
try:
    req = urllib.request.Request(
        f'http://127.0.0.1:{omlx_port}/v1/chat/completions',
        data=req_body,
        headers={
            'Content-Type': 'application/json',
            'Authorization': omlx_auth,
        },
        method='POST',
    )
    resp = urllib.request.urlopen(req, timeout=300)
    resp_body = json.loads(resp.read())
    t_req_end = time.time()
except Exception as e:
    try:
        server_proc.terminate(); server_proc.wait(timeout=5)
    except Exception:
        pass
    fail(f'OMLX_REQUEST_FAILED: {e}')

req_ms = round((t_req_end - t_req_start) * 1000)

# Parse response
usage = resp_body.get('usage', {})
prompt_tokens = usage.get('prompt_tokens', 0)
completion_tokens = usage.get('completion_tokens', 0)
content = ''
for ch in resp_body.get('choices', []):
    content += ch.get('message', {}).get('content', '') or ''

ttft_ms = req_ms  # wall-clock of the full request (no per-token timing from non-stream)
tps = round(completion_tokens / (req_ms / 1000), 3) if req_ms > 0 and completion_tokens > 0 else 0.0

# Peak RSS of the server process
try:
    rss_result = subprocess.run(['ps', '-o', 'rss=', '-p', str(server_proc.pid)], capture_output=True, text=True)
    peak_rss_mb = round(int(rss_result.stdout.strip()) / 1024, 1)
except Exception:
    peak_rss_mb = 0.0

# Stop server
try:
    server_proc.terminate()
    server_proc.wait(timeout=10)
except Exception:
    pass

output_first_50 = content.strip()[:200].replace('\n', ' ')

append_row(
    run_id=run_id, timestamp_utc=ts_utc(), backend=backend,
    model_basename=model_basename, quantization_type=quant_type,
    context_size=context_size, prompt=prompt_label, device=device,
    prompt_tokens=prompt_tokens, load_ms=load_ms, ttft_ms=ttft_ms,
    tps=f'{tps:.3f}', peak_rss_mb=peak_rss_mb,
    output_first_50=output_first_50,
)
print(f'{backend}: {model_basename}  load={load_ms}ms  ttft={ttft_ms}ms  tps={tps:.3f}  rss={peak_rss_mb}MB')
PYEOF
