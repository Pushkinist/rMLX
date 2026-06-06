#!/usr/bin/env bash
# Baseline measurement using Apple's stock mlx-lm loader.
#
# Usage:
#   ./run_mlx-lm.sh <model_dir> <prompt_file> <max_tokens> [cpu|gpu]
#
# Appends one row to metrics/baseline.csv (cross-backend schema).
# max_tokens=8 recommended for the cross-backend matrix.
# Time budget: 300 s per run; row written as TIMEOUT if exceeded.

set -euo pipefail

MODEL_DIR="${1:?model_dir required}"
PROMPT_FILE="${2:?prompt_file required}"
MAX_TOKENS="${3:?max_tokens required}"
DEVICE="${4:-gpu}"

VENV_GENERATE="${MLX_LM_ROOT:-../mlx-lm}/.venv/bin/mlx_lm.generate"
CSV="metrics/baseline.csv"
BACKEND="mlx-lm"
PROMPT_LABEL="$(basename "$PROMPT_FILE")"
MODEL_BASENAME="$(basename "$MODEL_DIR")"

mkdir -p metrics
if [ ! -s "$CSV" ]; then
    printf 'run_id,timestamp_utc,backend,model_basename,quantization_type,context_size,prompt,device,prompt_tokens,load_ms,ttft_ms,tps,peak_rss_mb,output_first_50\n' >> "$CSV"
fi

# Run via Python so we get clean timing and CSV writing without escaping issues.
python3 - "$MODEL_DIR" "$PROMPT_FILE" "$MAX_TOKENS" "$DEVICE" "$CSV" "$BACKEND" "$PROMPT_LABEL" "$MODEL_BASENAME" "$VENV_GENERATE" << 'PYEOF'
import csv, io, json, os, subprocess, sys, time
from datetime import datetime, timezone

model_dir, prompt_file, max_tokens, device, csv_path, backend, prompt_label, model_basename, venv_generate = sys.argv[1:]

def ts_utc():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z')

run_id = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-mlx-lm'

# Read quant metadata from config.json
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

def csv_escape(s):
    s = str(s)
    if any(c in s for c in [',', '"', '\n', '\r']):
        return '"' + s.replace('"', '""') + '"'
    return s

def append_row(**kw):
    fields = ['run_id','timestamp_utc','backend','model_basename','quantization_type',
              'context_size','prompt','device','prompt_tokens','load_ms','ttft_ms',
              'tps','peak_rss_mb','output_first_50']
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator='\n')
    w.writerow(kw)
    with open(csv_path, 'a') as f:
        f.write(buf.getvalue())

prompt_text = open(prompt_file).read()

t_start = time.time()
try:
    result = subprocess.run(
        [venv_generate,
         '--model', model_dir,
         '--prompt', prompt_text,
         '--max-tokens', max_tokens,
         '--temp', '0.0',
         '--seed', '0',
         '--verbose', 'true'],
        capture_output=True,
        text=True,
        timeout=300,
    )
    t_end = time.time()
    wall_ms = int((t_end - t_start) * 1000)
    combined = result.stdout + result.stderr
except subprocess.TimeoutExpired as e:
    combined = (e.stdout or '') + (e.stderr or '')
    append_row(
        run_id=run_id, timestamp_utc=ts_utc(), backend=backend,
        model_basename=model_basename, quantization_type=quant_type,
        context_size=context_size, prompt=prompt_label, device=device,
        prompt_tokens=0, load_ms=300000, ttft_ms=0, tps='0.000',
        peak_rss_mb=0.0, output_first_50='TIMEOUT',
    )
    print(f'mlx-lm: {model_basename} TIMEOUT (300s)')
    sys.exit(0)

# Check for arch not supported
if 'not supported' in combined.lower() or 'valueerror' in combined.lower() and 'not supported' in combined.lower():
    append_row(
        run_id=run_id, timestamp_utc=ts_utc(), backend=backend,
        model_basename=model_basename, quantization_type=quant_type,
        context_size=context_size, prompt=prompt_label, device=device,
        prompt_tokens=0, load_ms=0, ttft_ms=0, tps='0.000',
        peak_rss_mb=0.0, output_first_50='ARCH_NOT_SUPPORTED',
    )
    print(f'mlx-lm: {model_basename} ARCH_NOT_SUPPORTED')
    sys.exit(0)

# Parse verbose lines
import re
def find(pattern, text, default='0'):
    m = re.search(pattern, text)
    return m.group(1) if m else default

prompt_tokens = int(find(r'Prompt:\s*(\d+)\s*tokens', combined))
prompt_tps    = float(find(r'Prompt:\s*\d+\s*tokens,\s*([\d.]+)\s*tokens-per-sec', combined, '0'))
gen_tokens    = int(find(r'Generation:\s*(\d+)\s*tokens', combined))
gen_tps       = float(find(r'Generation:\s*\d+\s*tokens,\s*([\d.]+)\s*tokens-per-sec', combined, '0'))
peak_gb       = float(find(r'Peak memory:\s*([\d.]+)\s*GB', combined, '0'))

peak_rss_mb = round(peak_gb * 1024, 1)
gen_ms  = (gen_tokens / gen_tps * 1000) if gen_tps > 0 else 0
prompt_ms = (prompt_tokens / prompt_tps * 1000) if prompt_tps > 0 else 0
load_ms = max(0, round(wall_ms - prompt_ms - gen_ms))
ttft_ms = round(prompt_ms)

# Extract output between ========== markers
parts = re.split(r'^=+$', combined, flags=re.MULTILINE)
output_text = parts[1].strip() if len(parts) >= 3 else ''
output_first_50 = output_text[:200].replace('\n', ' ')

append_row(
    run_id=run_id, timestamp_utc=ts_utc(), backend=backend,
    model_basename=model_basename, quantization_type=quant_type,
    context_size=context_size, prompt=prompt_label, device=device,
    prompt_tokens=prompt_tokens, load_ms=load_ms, ttft_ms=ttft_ms,
    tps=f'{gen_tps:.3f}', peak_rss_mb=peak_rss_mb,
    output_first_50=output_first_50,
)
print(f'mlx-lm: {model_basename}  load={load_ms}ms  ttft={ttft_ms}ms  tps={gen_tps:.3f}  rss={peak_rss_mb}MB')
PYEOF
