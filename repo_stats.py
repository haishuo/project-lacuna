from pathlib import Path
from collections import defaultdict

ROOT = Path(".")
IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
}

TEXT_EXTENSIONS = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".toml",
    ".json",
}

def is_ignored(path: Path) -> bool:
    return any(part in IGNORE_DIRS for part in path.parts)

def count_lines(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            return sum(1 for _ in f)
    except Exception:
        return 0

def main():
    file_counts = defaultdict(int)
    file_lines = {}
    total_lines = 0

    for path in ROOT.rglob("*"):
        if path.is_file() and not is_ignored(path):
            ext = path.suffix
            file_counts[ext] += 1

            if ext in TEXT_EXTENSIONS:
                lines = count_lines(path)
                file_lines[path] = lines
                total_lines += lines

    print("\n=== Repository Statistics ===")
    print(f"Total files: {sum(file_counts.values())}")
    print(f"Total text LOC: {total_lines}")

    print("\nFiles by extension:")
    for ext, count in sorted(file_counts.items(), key=lambda x: -x[1]):
        print(f"  {ext or '[no ext]'}: {count}")

    print("\nLargest files (top 20):")
    for path, lines in sorted(file_lines.items(), key=lambda x: -x[1])[:20]:
        print(f"  {lines:5d}  {path}")

    print("\nFiles exceeding 500 LOC:")
    offenders = [(p, l) for p, l in file_lines.items() if l > 500]
    if not offenders:
        print("  None 🎉")
    else:
        for path, lines in sorted(offenders, key=lambda x: -x[1]):
            print(f"  {lines:5d}  {path}")

if __name__ == "__main__":
    main()