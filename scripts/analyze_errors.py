import json
from pathlib import Path
from collections import Counter
import difflib
from zalia_ruta.bench.srt_io import parse_srt
from zalia_ruta.bench.score import filter_cues_and_regions, normalize_word
from zalia_ruta.bench.systems import deserialize_words

manifest = json.loads(Path('benchmark/manifest.json').read_text(encoding='utf-8'))
refs_dir = Path('benchmark/references')
hyp_dir = Path('benchmark/hyp/paprika_ct2_acoustic')

substitutions = Counter()
deletions = Counter()
insertions = Counter()

for c in manifest['clips']:
    cid = c['id']
    ref_file = refs_dir / f'{cid}.srt'
    hyp_file = hyp_dir / f'{cid}.json'
    if not ref_file.exists() or not hyp_file.exists():
        continue
    ref_cues = parse_srt(ref_file.read_text(encoding='utf-8'))
    clean_refs, skip_intervals = filter_cues_and_regions(ref_cues)
    hyp_words = deserialize_words(json.loads(hyp_file.read_text(encoding='utf-8')))
    
    clean_hyps = [h for h in hyp_words if normalize_word(h.word)]
    
    ref_toks = [c.text for c in clean_refs]
    hyp_toks = [normalize_word(h.word) for h in clean_hyps]
    
    matcher = difflib.SequenceMatcher(None, ref_toks, hyp_toks)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            r_chunk = ' '.join(ref_toks[i1:i2])
            h_chunk = ' '.join(hyp_toks[j1:j2])
            substitutions[(r_chunk, h_chunk)] += 1
        elif tag == 'delete':
            for w in ref_toks[i1:i2]:
                deletions[w] += 1
        elif tag == 'insert':
            for w in hyp_toks[j1:j2]:
                insertions[w] += 1

print('=== TOP 25 SUBSTITUTIONS (Ref -> Hyp) ===')
for (r, h), count in substitutions.most_common(25):
    print(f'{count:2d}x:  "{r}"  ->  "{h}"')

print('\n=== TOP 10 DELETIONS (Missed words) ===')
for w, count in deletions.most_common(10):
    print(f'{count:2d}x:  "{w}"')

print('\n=== TOP 10 INSERTIONS (Hallucinated / extra words) ===')
for w, count in insertions.most_common(10):
    print(f'{count:2d}x:  "{w}"')

