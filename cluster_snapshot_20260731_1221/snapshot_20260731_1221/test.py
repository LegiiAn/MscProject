import json
d = json.load(open("data/processed_dataset/annotations.json"))
ids = {s: {int(r["source_file"].split("-")[0]) for r in d[s]} for s in d}
near = sum(1 for v in ids["val"] if any(v+k in ids["train"] for k in (-2,-1,1,2)))
print(f"val sources with a train neighbour within ±2 frames: {near}/{len(ids['val'])}")
