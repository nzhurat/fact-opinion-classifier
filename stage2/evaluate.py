import csv
import json
import numpy as np
import matplotlib.pyplot as plt


def load_results(path: str):
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def compute_metrics(rows, true_field="true_label", pred_field="predicted_label"):
    classes = ["Fact", "Opinion"]

    tp = {c: 0 for c in classes}
    fp = {c: 0 for c in classes}
    fn = {c: 0 for c in classes}

    correct = 0
    total = 0
    unknown = 0

    has_labels = true_field in rows[0] if rows else False

    if not has_labels:
        return None

    for r in rows:
        true = r.get(true_field)
        pred = r.get(pred_field)

        if not true:
            continue

        total += 1

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

    has_labels = true_field in rows[0] if rows else False

    if not has_labels:
        return None

    for r in rows:
        true = r.get(true_field)
        pred = r.get(pred_field)

        if not true:
            continue

        if pred not in classes:
            pred = "Unknown"

        matrix[true][pred] += 1

    return matrix


def plot_confusion_matrix(matrix, title):
    labels = ["Fact", "Opinion", "Unknown"]

    data = np.array([
        [
            matrix["Fact"]["Fact"],
            matrix["Fact"]["Opinion"],
            matrix["Fact"]["Unknown"]
        ],
        [
            matrix["Opinion"]["Fact"],
            matrix["Opinion"]["Opinion"],
            matrix["Opinion"]["Unknown"]
        ]
    ])

    fig, ax = plt.subplots(figsize=(6, 4))

    im = ax.imshow(data)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)

    ax.set_yticks(range(2))
    ax.set_yticklabels(["Fact", "Opinion"])

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(
                j,
                i,
                str(data[i, j]),
                ha="center",
                va="center"
            )

    plt.colorbar(im)
    plt.tight_layout()
    plt.show()


def print_report(name: str, rows):
    metrics = compute_metrics(rows)
    matrix = confusion_matrix(rows)

    print(f"\n{name}")

    if metrics is None:
        print("No ground truth labels - metrics not available")
        return None, None

    print(
        f"Accuracy: {metrics['accuracy']:.3f} "
        f"({metrics['total']} samples, {metrics['unknown']} unknown)"
    )

    print(f"\n{'Class':<10}{'Precision':<12}{'Recall':<12}{'F1':<12}")

    for c in ["Fact", "Opinion"]:
        m = metrics[c]
        print(
            f"{c:<10}"
            f"{m['precision']:<12.3f}"
            f"{m['recall']:<12.3f}"
            f"{m['f1']:<12.3f}"
        )

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

    return metrics, matrix


def export_for_charts(mode, m1, mat1, m2, mat2):
    if m1 is None or m2 is None:
        return

    summary = {
        "mode": mode,
        "zero_shot": {
            "accuracy": m1["accuracy"],
            "confusion_matrix": mat1,
        },
        "few_shot_cot": {
            "accuracy": m2["accuracy"],
            "confusion_matrix": mat2,
        },
    }

    out_path = f"chart_data_{mode}.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    zero_shot = load_results("results_zero_shot.csv")
    few_shot = load_results("results_few_shot_cot.csv")

    m1, mat1 = print_report("ZERO-SHOT", zero_shot)
    m2, mat2 = print_report("FEW-SHOT + CoT", few_shot)

    if mat1:
        plot_confusion_matrix(
            mat1,
            "Zero-shot Confusion Matrix"
        )

    if mat2:
        plot_confusion_matrix(
            mat2,
            "Few-shot + CoT Confusion Matrix"
        )

    if m1 and m2:
        diff = m2["accuracy"] - m1["accuracy"]
        sign = "+" if diff >= 0 else ""

        print(f"\nZero-shot accuracy: {m1['accuracy']:.3f}")
        print(f"Few-shot accuracy: {m2['accuracy']:.3f}")
        print(f"Delta: {sign}{diff:.3f}")

        export_for_charts("train", m1, mat1, m2, mat2)


if __name__ == "__main__":
    main()