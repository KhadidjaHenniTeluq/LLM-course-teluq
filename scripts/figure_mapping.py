"""
figure_mapping.py — Figure Mapping Pipeline

Reads a week's raw content file and outputs the chronological
Figure Mapping Table for use during migration.

Usage:
    python scripts/figure_mapping.py <week_number>

Example:
    python scripts/figure_mapping.py 3

Output:
    A table showing each unique figure reference in order of first
    encounter, its course label, and the resolved image filename.
"""

import os
import re
import json
import sys


def load_figure_map(base_dir: str) -> dict[str, str]:
    """Load figure_map.json → { "68.png": "Figure 3-15. Caption..." }"""
    path = os.path.join(base_dir, "source-assets", "figure_map.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_reverse_map(figure_map: dict[str, str]) -> dict[str, str]:
    """Build reverse lookup: "3-15" → "68.png" from figure_map.json.

    Each caption starts with "Figure X-Y." — we extract that label
    and map it back to the filename.
    """
    reverse: dict[str, str] = {}
    for filename, caption in figure_map.items():
        m = re.match(r"Figure\s+(\d+-\d+)\.", caption)
        if m:
            book_ref = m.group(1)  # e.g. "3-15"
            reverse[book_ref] = filename  # e.g. "68.png"
    return reverse


def extract_figure_refs(content: str) -> list[str]:
    """Scan content top-to-bottom and return all figure references
    in order of appearance (including duplicates).

    Handles:
      • Figure 3-15
      • Figures 3-32 et 3-33
      • Figures 1-3 à 1-5
      • Figures 1-1, 1-2 et 1-3
    """
    all_refs: list[str] = []

    # Matches a block of figure references starting with "Figure(s)"
    # Examples:
    # "Figure 1-1"
    # "Figures 1-1 et 1-2"
    # "Figures 1-1, 1-2 et 1-3"
    # "Figures 1-1 à 1-5"
    block_pattern = re.compile(
        r"Figures?\s+(\d+-\d+(?:(?:\s*,\s*|\s+et\s+|\s+[àa]\s+)\d+-\d+)*)",
        re.IGNORECASE,
    )

    for line in content.split("\n"):
        for m in block_pattern.finditer(line):
            block_content = m.group(1)

            # 1. Split by discrete separators (comma, "et")
            discrete_parts = re.split(
                r"\s*,\s*|\s+et\s+", block_content, flags=re.IGNORECASE
            )
            for part in discrete_parts:
                part = part.strip()
                # 2. Check if this part is a range ("à")
                range_parts = re.split(r"\s+[àa]\s+", part, flags=re.IGNORECASE)
                if len(range_parts) == 2:
                    m_start = re.search(r"(\d+)-(\d+)", range_parts[0])
                    m_end = re.search(r"(\d+)-(\d+)", range_parts[1])
                    if m_start and m_end:
                        c1, f1 = int(m_start.group(1)), int(m_start.group(2))
                        c2, f2 = int(m_end.group(1)), int(m_end.group(2))
                        if c1 == c2:
                            for n in range(f1, f2 + 1):
                                all_refs.append(f"{c1}-{n}")
                        else:
                            # Cross-chapter range - just keep ends
                            all_refs.append(f"{c1}-{f1}")
                            all_refs.append(f"{c2}-{f2}")
                else:
                    # 3. Otherwise find any singles in this part
                    for ref_match in re.finditer(r"(\d+)-(\d+)", part):
                        all_refs.append(
                            f"{ref_match.group(1)}-{ref_match.group(2)}"
                        )

    return all_refs


def build_mapping_table(
    week_num: int,
    content: str,
    figure_map: dict[str, str],
) -> list[dict]:
    """Build the chronological figure mapping table.

    Returns a list of dicts:
        {
            "counter": 1,
            "book_ref": "3-15",
            "course_label": "Figure 3-1",
            "image_file": "68.png",
            "caption": "Figure 3-15. A simplified framing...",
        }
    """
    reverse_map = build_reverse_map(figure_map)
    all_refs = extract_figure_refs(content)

    # Assign counters based on order of first encounter
    seen: set[str] = set()
    ordered_unique: list[str] = []

    for ref in all_refs:
        if ref not in seen:
            seen.add(ref)
            ordered_unique.append(ref)

    # Build the table
    table: list[dict] = []
    for i, ref in enumerate(ordered_unique, 1):
        image_file = reverse_map.get(ref, "??? (NOT FOUND)")
        caption = figure_map.get(image_file, "??? (no caption)")
        
        # Truncate caption for table display
        display_caption = caption[:80] + "..." if len(caption) > 80 else caption
        
        table.append(
            {
                "counter": i,
                "book_ref": f"Figure {ref}",
                "course_label": f"Figure {week_num}-{i}",
                "image_file": image_file,
                "caption": display_caption,
            }
        )

    return table


def print_table(table: list[dict], week_num: int) -> None:
    """Pretty-print the mapping table."""
    print(f"\n{'='*90}")
    print(f" FIGURE MAPPING TABLE — WEEK {week_num:02d}")
    print(f"{'='*90}")
    print(
        f" {'#':<4} {'Book Ref':<14} {'Course Label':<14} {'Image':<12} {'Caption (truncated)'}"
    )
    print(f" {'-'*4} {'-'*14} {'-'*14} {'-'*12} {'-'*40}")

    for row in table:
        print(
            f" {row['counter']:<4} {row['book_ref']:<14} {row['course_label']:<14} "
            f"{row['image_file']:<12} {row['caption']}"
        )

    print(f"{'='*90}")
    print(f" Total unique figures: {len(table)}")
    print()


def print_sidecar_fix_script(table: list[dict], week_num: int) -> None:
    """Print a Python dict that can be used to fix sidecar .txt files."""
    print(f"\n--- Sidecar Caption Fix Mapping (for static/images/week{week_num:02d}/) ---")
    print("mapping = {")
    for row in table:
        idx = row["image_file"].replace(".png", "")
        print(f'    "{idx}": "{row["course_label"]}",')
    print("}")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/figure_mapping.py <week_number>")
        print("Example: python scripts/figure_mapping.py 3")
        sys.exit(1)

    week_num = int(sys.argv[1])
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # Find raw content file
    raw_path = os.path.join(
        base_dir, "source-assets", "raw-content", f"week-{week_num}.md"
    )
    if not os.path.exists(raw_path):
        print(f"ERROR: Raw content file not found: {raw_path}")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        content = f.read()

    figure_map = load_figure_map(base_dir)
    table = build_mapping_table(week_num, content, figure_map)

    if not table:
        print(f"No figure references found in week-{week_num}.md")
        sys.exit(0)

    print_table(table, week_num)
    print_sidecar_fix_script(table, week_num)


if __name__ == "__main__":
    main()
