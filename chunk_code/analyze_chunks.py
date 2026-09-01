import json

with open("output_chunks/all_chunks_chunks.json", encoding="utf-8") as f:
    chunks = json.load(f)

lens = [c["chunk_len"] for c in chunks]
print(f"Total chunks: {len(chunks)}")
print(f"Total chars: {sum(lens)}")
print(f"Average chunk length: {sum(lens)/len(lens):.1f}")
print(f"Min: {min(lens)}, Max: {max(lens)}")
print(f"Total docs (lines): 10451")
print(f"Compression ratio: {len(chunks)/10451:.2f}x (chunks per line)")

from collections import Counter
lens_bucket = Counter()
for l in lens:
    if l < 200:
        lens_bucket["<200"] += 1
    elif l < 400:
        lens_bucket["200-400"] += 1
    elif l < 600:
        lens_bucket["400-600"] += 1
    elif l < 800:
        lens_bucket["600-800"] += 1
    else:
        lens_bucket[">=800"] += 1

print("\nChunk length distribution:")
for bucket, count in sorted(lens_bucket.items()):
    print(f"  {bucket}: {count} ({count*100/len(lens):.1f}%)")

print("\nSample chunks:")
for i in range(min(5, len(chunks))):
    c = chunks[i]
    text = c["chunk_text"]
    print(f"  [{i}] len={c['chunk_len']}, text={text[:100]}")

print("\n\nJSON chunks with metadata sample (first 3):")
for c in chunks[:3]:
    print(json.dumps(c, ensure_ascii=False, indent=2)[:300])
    print("---")
