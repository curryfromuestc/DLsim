#!/usr/bin/env python3
"""Print a simulated InferenceX row next to the measured snapshot row: compare_row.py sim.json [sim2.json ...]"""
import json, sys
rows = {int(r['id']): r for r in json.load(open('traces/inferencex/benchmarks_DeepSeek-V4-Pro_2026-09-20.json'))}
der = json.load(open('traces/inferencex/derived-agentic-metrics_2026-09-20.json'))
KEYS = ['tput_per_gpu', 'input_tput_per_gpu', 'output_tput_per_gpu', 'mean_ttft', 'median_ttft', 'p90_ttft', 'mean_itl', 'p90_itl',
        'p90_intvty', 'mean_e2el', 'p90_e2el', 'mean_input_tokens', 'mean_output_tokens_actual', 'total_requests_completed', 'duration_seconds', 'cache_hit_rate', 'theoretical_cache_hit_rate']
for path in sys.argv[1:]:
    r = json.load(open(path))
    id = int(path.rsplit('_', 1)[1].split('.')[0])
    m, mm, d = r['metrics'], rows[id]['metrics'], der.get(str(id), {})
    print(f"== {id} {rows[id]['hardware']} {rows[id]['framework']} conc {rows[id]['conc']}   {'metric':26}{'sim':>12}{'meas':>12}{'err':>8}")
    for k in KEYS + ['p90_e2e_norm_intvty', 'p75_e2e_norm_intvty']:
        a = m.get(k); b = mm.get(k, d.get(k))
        e = (a / b - 1) * 100 if a and b else float('nan')
        fa = round(a, 4) if isinstance(a, float) else a; fb = round(b, 4) if isinstance(b, float) else b
        print(f"   {k:26}{str(fa):>12}{str(fb):>12}{e:>7.0f}%")
    x = r['dlsim']
    print('   time_share', {k: round(v, 3) for k, v in x['time_share'].items()}, 'unsupported', x['extra'].get('unsupported_passes'),
          'preemptions', x['extra'].get('preemptions'), 'tier_hits', x['tier_hits'], 'wall', round(x['extra'].get('wall_s', 0), 1))
