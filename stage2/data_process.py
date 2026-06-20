import csv
from pathlib import Path
from kb import cur_exmp

inp = "data/dataset.csv"
eval = "stage2/eval_set.csv"
pref_len = 80


def normalize_label(label: str) -> str:
    label = label.strip().lower()
    if label.startswith("fact"):
        return "Fact"
    if label.startswith("opin"):
        return "Opinion"
    raise ValueError(f"Unknown label: {label}")


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip("\"'\u201c\u201d\u2018\u2019")


def load_dataset(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        return [
            {
                "text": row["Content"].strip(),
                "label": normalize_label(row["Classification"]),
            }
            for row in reader
            if row["Content"].strip()
        ]


def deduplicate(rows):
    seen = {}
    deduped = []
    conflicts = []

    for row in rows:
        key = normalize_text(row["text"])

        if key not in seen:
            seen[key] = row["label"]
            deduped.append(row)

        elif seen[key] != row["label"]:
            conflicts.append((key[:80], seen[key], row["label"]))

    if conflicts:
        print(f"\nWarning: {len(conflicts)} label conflicts found:")
        for text, label1, label2 in conflicts:
            print(f"  '{text}...' -> {label1} vs {label2}")

    return deduped, len(rows) - len(deduped)


def main():
    rows = load_dataset(inp)

    print(f"Loaded {len(rows)} rows")

    rows, removed = deduplicate(rows)

    fact_count = sum(r["label"] == "Fact" for r in rows)
    opinion_count = len(rows) - fact_count

    print(f"Removed {removed} duplicates")
    print(f"Fact: {fact_count}, Opinion: {opinion_count}")

    kb_prefixes = {
        normalize_text(example["text"])[:pref_len]
        for example in cur_exmp
    }

    eval_rows = [
        row
        for row in rows
        if normalize_text(row["text"])[:pref_len] not in kb_prefixes
    ]

    excluded = len(rows) - len(eval_rows)

    if excluded != len(kb_prefixes):
        print(
            f"Warning: expected {len(kb_prefixes)} exclusions, "
            f"but got {excluded}"
        )

    with open(eval, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()
        writer.writerows(eval_rows)

    print(f"Excluded {excluded} few-shot examples")
    print(f"Saved {len(eval_rows)} rows to {eval}")


if __name__ == "__main__":
    main()