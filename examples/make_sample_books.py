"""Create two deterministic sample TXT books for a local smoke run.

Usage:
    python examples/make_sample_books.py --output books
"""

from __future__ import annotations

import argparse
from pathlib import Path


def chinese_book() -> str:
    lines = ["\u7b2c\u4e00\u5377 \u6cb3\u6d41", ""]
    for chapter in range(1, 6):
        lines.append(f"\u7b2c{chapter}\u7ae0 \u65c5\u9014{chapter}")
        lines.append("")
        for paragraph in range(8):
            lines.append(
                f"\u4ed6\u5728\u7b2c{chapter}\u7ae0\u7684\u7b2c{paragraph}\u6bb5\u62ac\u5934\u770b\u5411\u8fdc\u65b9\uff0c"
                "\u5c71\u8c37\u91cc\u7684\u98ce\u7a7f\u8fc7\u6811\u6797\uff0c\u6cb3\u6c34\u5728\u77f3\u5934\u4e4b\u95f4\u53d1\u51fa\u6e05\u6670\u7684\u56de\u54cd\u3002"
                "\u4eba\u7269\u6cbf\u7740\u5c0f\u8def\u7ee7\u7eed\u524d\u884c\uff0c\u5fc3\u4e2d\u8bb0\u4e0b\u4e86\u65c1\u8fb9\u7684\u666f\u8c61\u4e0e\u5bf9\u8bdd\u3002\n\n"
            )
    return "".join(lines)


def english_book() -> str:
    lines = ["Volume One: The Coast", ""]
    for chapter in range(1, 5):
        lines.append(f"Chapter {chapter}: A Different Road")
        lines.append("")
        for paragraph in range(8):
            lines.append(
                f"In chapter {chapter}, paragraph {paragraph}, the traveler watched the tide cross the harbor. "
                "The morning light moved over the roofs while a small boat carried supplies toward the pier. "
                "Every observation was recorded clearly so the narrative remained complete.\n\n"
            )
    return "".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="books")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "sample_zh.txt").write_text(chinese_book(), encoding="utf-8")
    (output / "sample_en.txt").write_text(english_book(), encoding="utf-8")
    print(f"wrote {output / 'sample_zh.txt'} and {output / 'sample_en.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())