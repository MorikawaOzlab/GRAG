import json
from pathlib import Path

root = Path('output/cppo')
summaries = []
for path in sorted(root.glob('diagnosis*.json')):
    data = json.loads(path.read_text())
    for result in data['results']:
        last = result['diagnostics'][-1]
        summaries.append({'file': path.name, **{key: last[key] for key in
                         ['variant', 'seed', 'episode', 'successes', 'visited', 'wall_entropy']},
                         'last_100_success_rate': sum(r['success'] for r in result['history'][-100:]) / min(100, len(result['history']))})
print(json.dumps(summaries, indent=2))
(root / 'investigation_summary.json').write_text(json.dumps(summaries, indent=2))
