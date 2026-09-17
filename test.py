import json

with open("val_pairs_individual.json") as f:
    val = json.load(f)

print("All keys in val entry:")
print(list(val[0].keys()))
print("\nSample entry:")
print(json.dumps(val[0], indent=2))
