"""Poll a running station and append JSONL evidence. Does not fabricate 24-hour results."""
import argparse
import json
import time
import urllib.request

p = argparse.ArgumentParser()
p.add_argument('--url', default='http://localhost:8000')
p.add_argument('--hours', type=float, default=24)
p.add_argument('--interval', type=float, default=60)
p.add_argument('--output', default='soak.jsonl')
a = p.parse_args()
if a.interval <= 0 or a.hours <= 0:
    p.error('hours and interval must be positive')
end = time.monotonic() + a.hours * 3600
while time.monotonic() < end:
    row = {'timestamp': time.time()}
    try:
        with urllib.request.urlopen(a.url + '/api/status', timeout=15) as response:
            row['status'] = json.load(response)
        row['ok'] = True
    except Exception as exc:
        row.update(ok=False, error=str(exc))
    with open(a.output, 'a') as f:
        f.write(json.dumps(row) + '\n')
    time.sleep(min(a.interval, max(0, end-time.monotonic())))
