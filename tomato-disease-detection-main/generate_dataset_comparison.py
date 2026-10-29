from __future__ import annotations

import csv
import html
import os
from pathlib import Path

from tomato_model_utils import summarize_dataset


REPO_ROOT = Path(__file__).resolve().parent
KAGGLE_DATASET = Path(
    os.environ.get("KAGGLE_DATASET", REPO_ROOT.parent / "Tomato_Leaf_Dataset")
)
GITHUB_PREPARED_DATASET = Path(
    os.environ.get("GITHUB_PREPARED_DATASET", REPO_ROOT / "temp_training_dataset")
)
REPORT_DIR = REPO_ROOT / "reports"

CLASS_NAMES = (
    "Early-blight",
    "Healthy",
    "Late-blight",
    "Leaf-mold",
    "Septoria-leaf-spot",
    "Yellow-leaf-curl-virus",
)
KAGGLE_CLASS_NAMES = {
    "Early-blight": "Tomato_Early_blight",
    "Healthy": "Tomato_Healthy",
    "Late-blight": "Tomato_leaf_late_blight",
    "Leaf-mold": "Tomato_mold_leaf",
    "Septoria-leaf-spot": "Tomato_septora_leaf_spot",
    "Yellow-leaf-curl-virus": "Tomato_leaf_yellow_curl_virus",
}


def build_rows() -> list[dict[str, object]]:
    kaggle = summarize_dataset(str(KAGGLE_DATASET))
    github = summarize_dataset(str(GITHUB_PREPARED_DATASET))
    rows = []
    for class_name in CLASS_NAMES:
        kaggle_count = kaggle["classes"].get(KAGGLE_CLASS_NAMES[class_name], 0)
        github_count = github["classes"].get(class_name, 0)
        rows.append(
            {
                "class": class_name,
                "kaggle_images": kaggle_count,
                "github_prepared_images": github_count,
                "difference": github_count - kaggle_count,
            }
        )
    return rows


def write_csv(rows: list[dict[str, object]]) -> Path:
    output = REPORT_DIR / "dataset_comparison.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return output


def write_svg(rows: list[dict[str, object]]) -> Path:
    output = REPORT_DIR / "dataset_comparison.svg"
    width, height = 1000, 620
    chart_left, chart_top, chart_width, chart_height = 260, 100, 650, 400
    maximum = max(
        max(int(row["kaggle_images"]) for row in rows),
        max(int(row["github_prepared_images"]) for row in rows),
    )
    bar_height = 24
    group_height = chart_height / len(rows)
    scale = chart_width / maximum
    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>text{font-family:Arial,sans-serif;fill:#222} "
        ".title{font-size:24px;font-weight:bold}.label{font-size:16px} "
        ".value{font-size:14px}.note{font-size:13px;fill:#555}</style>",
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="40" y="42" class="title">Tomato leaf dataset comparison</text>',
        '<rect x="40" y="62" width="18" height="18" fill="#4285f4"/>',
        '<text x="66" y="77" class="label">Kaggle source</text>',
        '<rect x="210" y="62" width="18" height="18" fill="#f28e2b"/>',
        '<text x="236" y="77" class="label">GitHub prepared copy</text>',
        f'<line x1="{chart_left}" y1="{chart_top}" x2="{chart_left}" '
        f'y2="{chart_top + chart_height}" stroke="#777"/>',
        f'<line x1="{chart_left}" y1="{chart_top + chart_height}" '
        f'x2="{chart_left + chart_width}" y2="{chart_top + chart_height}" stroke="#777"/>',
    ]
    for index, row in enumerate(rows):
        y = chart_top + index * group_height + 12
        kaggle_width = int(row["kaggle_images"]) * scale
        github_width = int(row["github_prepared_images"]) * scale
        label = html.escape(str(row["class"]))
        svg.extend(
            [
                f'<text x="40" y="{y + 22}" class="label">{label}</text>',
                f'<rect x="{chart_left}" y="{y}" width="{kaggle_width}" '
                f'height="{bar_height}" fill="#4285f4"/>',
                f'<rect x="{chart_left}" y="{y + bar_height + 4}" width="{github_width}" '
                f'height="{bar_height}" fill="#f28e2b"/>',
                f'<text x="{chart_left + kaggle_width + 8}" y="{y + 17}" '
                f'class="value">{row["kaggle_images"]:,}</text>',
                f'<text x="{chart_left + github_width + 8}" y="{y + bar_height + 21}" '
                f'class="value">{row["github_prepared_images"]:,}</text>',
            ]
        )
    svg.extend(
        [
            f'<text x="40" y="{height - 44}" class="note">Note: the prepared folder '
            "contains the Kaggle filenames plus duplicated/augmented copies; it is not "
            "an independent source.</text>",
            "</svg>",
        ]
    )
    output.write_text("\n".join(svg), encoding="utf-8")
    return output


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    rows = build_rows()
    csv_path = write_csv(rows)
    svg_path = write_svg(rows)

    print("Class                         Kaggle    GitHub prepared    Difference")
    print("-" * 72)
    for row in rows:
        print(
            f'{row["class"]:<30} {row["kaggle_images"]:>6,}'
            f'{row["github_prepared_images"]:>18,}{row["difference"]:>13,}'
        )
    print("-" * 72)
    print(
        f'{"Total images":<30} '
        f'{sum(int(row["kaggle_images"]) for row in rows):>6,}'
        f'{sum(int(row["github_prepared_images"]) for row in rows):>18,}'
    )
    print(f"\nTable: {csv_path}")
    print(f"Graph: {svg_path}")
    print(
        "\nInterpretation: the GitHub prepared folder is derived from the Kaggle "
        "files in this workspace, so these counts do not measure independent "
        "dataset quality."
    )


if __name__ == "__main__":
    main()
