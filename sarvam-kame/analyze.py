import json, statistics
from pathlib import Path

logs = []
for line in Path("latency_log.jsonl").read_text().strip().splitlines():
    try: logs.append(json.loads(line))
    except: pass

cascaded = [l for l in logs if l["mode"]=="cascaded" and l.get("total_latency")]
kame     = [l for l in logs if l["mode"]=="kame"     and l.get("time_to_first_audio")]

print(f"\n{'═'*50}")
print(f"  SARVAM-KAME LATENCY ANALYSIS")
print(f"  Total sessions: {len(logs)}")
print(f"{'═'*50}")

if cascaded:
    tots = [l["total_latency"] for l in cascaded]
    print(f"\n  CASCADED (baseline) — {len(cascaded)} sessions")
    print(f"  median total latency : {statistics.median(tots):.3f}s")
    print(f"  mean   total latency : {statistics.mean(tots):.3f}s")
    print(f"  best                 : {min(tots):.3f}s")
    print(f"  worst                : {max(tots):.3f}s")

if kame:
    firsts = [l["time_to_first_audio"] for l in kame]
    print(f"\n  KAME (tandem) — {len(kame)} sessions")
    print(f"  median first audio   : {statistics.median(firsts):.3f}s")
    print(f"  mean   first audio   : {statistics.mean(firsts):.3f}s")
    print(f"  best                 : {min(firsts):.3f}s")
    print(f"  worst                : {max(firsts):.3f}s")

if cascaded and kame:
    cas_med = statistics.median([l["total_latency"]      for l in cascaded])
    kme_med = statistics.median([l["time_to_first_audio"] for l in kame])
    pct = (cas_med - kme_med) / cas_med * 100
    print(f"\n  IMPROVEMENT")
    print(f"  {cas_med:.3f}s  →  {kme_med:.3f}s")
    print(f"  {pct:.1f}% reduction in time-to-first-audio")

print(f"\n{'═'*50}\n")