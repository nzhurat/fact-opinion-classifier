import csv


def load_results(path: str):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))



def compute_metrics(rows, true_field="true_label", pred_field="predicted_label"):
    classes = ["Fact", "Opinion"]

    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    correct = 0
    total = len(rows)
    unknown = 0

    for r in rows:
        true = r[true_field]
        pred = r[pred_field]

        if pred == "Unknown":
            unknown += 1
            continue

        if pred == true:
            correct += 1
            tp[true] += 1
        else:
            fn[true] += 1
            if pred in classes:
                fp[pred] += 1

    metrics = {
        "accuracy": correct / total if total else 0,
        "total": total,
        "unknown": unknown,
    }

    for c in classes:
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0
        )

        metrics[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics



def confusion_matrix(rows, true_field="true_label", pred_field="predicted_label"):
    classes = ["Fact", "Opinion", "Unknown"]

    matrix = {
        "Fact": {"Fact": 0, "Opinion": 0, "Unknown": 0},
        "Opinion": {"Fact": 0, "Opinion": 0, "Unknown": 0},
    }

    for r in rows:
        true = r[true_field]
        pred = r[pred_field] if r[pred_field] in classes else "Unknown"

        if true in matrix:
            matrix[true][pred] += 1

    return matrix



def print_report(name: str, rows):
    metrics = compute_metrics(rows)
    matrix = confusion_matrix(rows)

    print("\n" + "=" * 60)
    print(f"{name}")
    print("=" * 60)

    print(
        f"Accuracy: {metrics['accuracy']:.3f} "
        f"({metrics['total']} samples, {metrics['unknown']} unknown)"
    )

    print(f"\n{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}")

    for c in ["Fact", "Opinion"]:
        m = metrics[c]
        print(f"{c:<10}{m['precision']:<12.3f}{m['recall']:<12.3f}{m['f1']:<12.3f}")

    print("\nConfusion matrix:")
    print(f"{'':<10}{'Fact':<10}{'Opinion':<10}{'Unknown':<10}")

    for true_c in ["Fact", "Opinion"]:
        row = matrix[true_c]
        print(
            f"{true_c:<10}"
            f"{row['Fact']:<10}"
            f"{row['Opinion']:<10}"
            f"{row['Unknown']:<10}"
        )

    return metrics

def main():
    zero_shot = load_results("results_zero_shot.csv")
    few_shot = load_results("results_few_shot_cot.csv")

    m1 = print_report("ZERO-SHOT", zero_shot)
    m2 = print_report("FEW-SHOT + COT", few_shot)

    diff = m2["accuracy"] - m1["accuracy"]
    sign = "+" if diff >= 0 else ""

    print(f"Zero-shot accuracy:    {m1['accuracy']:.3f}")
    print(f"Few-shot + COT:       {m2['accuracy']:.3f}")
    print(f"Delta:                {sign}{diff:.3f}")


if __name__ == "__main__":
    main()