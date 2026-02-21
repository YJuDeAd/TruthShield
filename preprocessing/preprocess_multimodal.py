import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# ======================
# CONFIG
# ======================

# Supported image formats
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# Label mappings — add synonyms here if new datasets use different strings
FAKE_LABELS = {"fake", "tampered", "manipulated"}
REAL_LABELS = {"real", "non-tampered", "nontampered", "non_tampered"}

TEST_RATIO = 0.2
RANDOM_SEED = 42


# ======================
# 1. Twitter 15/16
# Download: https://github.com/gszswork/Twitter15_16_dataset
# 2. FakeNewsNet
# Download: https://github.com/KaiDMML/FakeNewsNet
# 3. MediaEval
# Download: https://github.com/MKLab-ITI/image-verification-corpus
# ======================


# ======================
# HELPERS
# ======================

@dataclass(frozen=True)
class Sample:
    image_path: str
    text: str
    label: int  # 1 = fake, 0 = real

def normalize_label(raw: str) -> Optional[int]:
    """Convert raw label string to binary int. Returns None if unrecognised."""
    label = raw.strip().lower()
    if label in FAKE_LABELS:
        return 1
    if label in REAL_LABELS:
        return 0
    return None


def parse_image_ids(raw: str) -> List[str]:
    """Split a cell that may contain one or more image IDs."""
    return [item.strip() for item in raw.replace(",", " ").split() if item.strip()]


def read_tsv(path: Path) -> Iterable[List[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        for row in csv.reader(f, delimiter="\t"):
            if row:
                yield row

def find_dataset_files(root: Path) -> Dict[str, List[Path]]:
    """
    Walk the raw data directory and collect known dataset file paths.
    Returns a dict mapping filename/dirname → list of matching paths.
    """
    targets: Dict[str, List[Path]] = {
        "tweets_images.txt": [],
        "tweets_images_update.txt": [],
        "tweets.txt": [],
        "posts.txt": [],
        "posts_groundtruth.txt": [],
        "mediaeval2015": [],
        "mediaeval2016": [],
    }

    for dirpath, dirnames, filenames in os.walk(root):
        lower_dirs = [d.lower() for d in dirnames]

        for key in ("mediaeval2015", "mediaeval2016"):
            if key in lower_dirs:
                idx = lower_dirs.index(key)
                targets[key].append(Path(dirpath) / dirnames[idx])

        for filename in filenames:
            if filename.lower() in targets:
                targets[filename.lower()].append(Path(dirpath) / filename)

    return targets


def build_image_index(mediaeval_dirs: Sequence[Path]) -> Dict[str, str]:
    """
    Index all images found under the given directories by filename stem.
    Used to resolve image IDs from TSV files to absolute paths.
    """
    index: Dict[str, str] = {}
    for base in mediaeval_dirs:
        for dirpath, _, filenames in os.walk(base):
            for filename in filenames:
                if Path(filename).suffix.lower() in IMAGE_EXTENSIONS:
                    full_path = Path(dirpath) / filename
                    index[full_path.stem.lower()] = str(full_path.resolve())
    return index


def select_file(paths: List[Path], required_parts: Sequence[str]) -> Optional[Path]:
    """
    From a list of paths, return the first one whose parts contain all
    required_parts (case-insensitive). Falls back to paths[0] if none match.
    """
    for path in paths:
        lowered = [p.lower() for p in path.parts]
        if all(part.lower() in lowered for part in required_parts):
            return path
    return paths[0] if paths else None

def parse_tweets_images(path: Path) -> Iterable[Tuple[str, List[str], Optional[int]]]:
    """Parser for tweets_images.txt and tweets_images_update.txt."""
    for row in read_tsv(path):
        if row[0].strip().lower() in {"tweet_id", "tweetid"}:
            continue
        if len(row) < 4:
            continue
        image_ids = [img for cell in row[1:-2] for img in parse_image_ids(cell)]
        label = normalize_label(row[-2])
        if label is not None:
            yield "", image_ids, label


def parse_mediaeval2015(path: Path) -> Iterable[Tuple[str, List[str], Optional[int]]]:
    """Parser for MediaEval 2015 tweets.txt (dev and test)."""
    for row in read_tsv(path):
        if row[0].strip().lower() == "tweetid" or len(row) < 7:
            continue
        label = normalize_label(row[6])
        if label is not None:
            yield row[1].strip(), parse_image_ids(row[3]), label


def parse_mediaeval2016_dev(path: Path) -> Iterable[Tuple[str, List[str], Optional[int]]]:
    """Parser for MediaEval 2016 posts.txt (devset)."""
    for row in read_tsv(path):
        if row[0].strip().lower() == "post_id" or len(row) < 7:
            continue
        label = normalize_label(row[6])
        if label is not None:
            yield row[1].strip(), parse_image_ids(row[3]), label


def parse_mediaeval2016_test(path: Path) -> Iterable[Tuple[str, List[str], Optional[int]]]:
    """Parser for MediaEval 2016 posts_groundtruth.txt (testset — image col shifted)."""
    for row in read_tsv(path):
        if row[0].strip().lower() == "post_id" or len(row) < 7:
            continue
        label = normalize_label(row[6])
        if label is not None:
            yield row[1].strip(), parse_image_ids(row[4]), label


def resolve_samples(
    items: Iterable[Tuple[str, List[str], Optional[int]]],
    image_index: Dict[str, str],
) -> Tuple[List[Sample], int]:
    """
    Match image IDs to actual file paths using the image index.
    Returns (matched samples, count of unresolved image IDs).
    """
    samples: List[Sample] = []
    missing = 0

    for text, image_ids, label in items:
        if label is None:
            continue
        for image_id in image_ids:
            key = image_id.strip().lower()
            if not key:
                continue
            path = image_index.get(key)
            if path:
                samples.append(Sample(image_path=path, text=text, label=label))
            else:
                missing += 1

    return samples, missing


def deduplicate(samples: Sequence[Sample]) -> List[Sample]:
    seen = set()
    unique = []
    for s in samples:
        key = (s.image_path, s.text, s.label)
        if key not in seen:
            seen.add(key)
            unique.append(s)
    return unique


def stratified_split(
    samples: Sequence[Sample],
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> Tuple[List[Sample], List[Sample]]:
    """Split samples into train/test while preserving label distribution."""
    rng = random.Random(seed)
    by_label: Dict[int, List[Sample]] = {}

    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)

    train, test = [], []
    for group in by_label.values():
        rng.shuffle(group)
        split = max(1, int(len(group) * (1 - test_ratio)))
        if split >= len(group) and len(group) > 1:
            split = len(group) - 1
        train.extend(group[:split])
        test.extend(group[split:])

    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


def write_csv(path: Path, samples: Sequence[Sample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "text", "label"])
        for s in samples:
            writer.writerow([s.image_path, s.text, s.label])


# ======================
# MAIN
# ======================

def main() -> None:
    root     = Path(__file__).resolve().parents[1]
    raw_root = root / "data" / "raw" / "social_media"

    if not raw_root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_root}")

    print("Discovering dataset files...")
    discovered   = find_dataset_files(raw_root)
    mm_dirs      = discovered["mediaeval2015"] + discovered["mediaeval2016"]
    image_index  = build_image_index(mm_dirs)
    print(f"Indexed {len(image_index)} images")

    # Locate individual TSV files
    files = {
        "tweets_images":        select_file(discovered["tweets_images.txt"],        ["image-verification-corpus-master"]),
        "tweets_images_update": select_file(discovered["tweets_images_update.txt"], ["image-verification-corpus-master"]),
        "mediaeval2015_dev":    select_file(discovered["tweets.txt"],               ["mediaeval2015", "devset"]),
        "mediaeval2015_test":   select_file(discovered["tweets.txt"],               ["mediaeval2015", "testset"]),
        "mediaeval2016_dev":    select_file(discovered["posts.txt"],                ["mediaeval2016", "devset"]),
        "mediaeval2016_test":   select_file(discovered["posts_groundtruth.txt"],    ["mediaeval2016", "testset"]),
    }

    # Parse each source
    parsers = {
        "tweets_images":        (files["tweets_images"],        parse_tweets_images),
        "tweets_images_update": (files["tweets_images_update"], parse_tweets_images),
        "mediaeval2015_dev":    (files["mediaeval2015_dev"],    parse_mediaeval2015),
        "mediaeval2015_test":   (files["mediaeval2015_test"],   parse_mediaeval2015),
        "mediaeval2016_dev":    (files["mediaeval2016_dev"],    parse_mediaeval2016_dev),
        "mediaeval2016_test":   (files["mediaeval2016_test"],   parse_mediaeval2016_test),
    }

    all_samples: List[Sample] = []
    total_missing = 0

    print("\nParsing datasets...")
    for name, (path, parser) in parsers.items():
        if path is None:
            print(f"  {name}: NOT FOUND — skipped")
            continue
        samples, missing = resolve_samples(parser(path), image_index)
        all_samples.extend(samples)
        total_missing += missing
        print(f"  {name}: {len(samples)} samples ({missing} missing images)")

    # Deduplicate and split
    unique  = deduplicate(all_samples)
    train, test = stratified_split(unique)

    # Write outputs
    out_dir = root / "data" / "processed" / "social_media"
    write_csv(out_dir / "train.csv", train)
    write_csv(out_dir / "test.csv",  test)

    # Summary
    print(f"\nPreprocessing complete")
    print(f"  Total images indexed : {len(image_index)}")
    print(f"  Missing image refs   : {total_missing}")
    print(f"  Unique samples       : {len(unique)}")
    print(f"  Train split          : {len(train)}")
    print(f"  Test split           : {len(test)}")

    label_counts = {0: 0, 1: 0}
    for s in unique:
        label_counts[s.label] += 1
    print(f"  Real (0)             : {label_counts[0]}")
    print(f"  Fake (1)             : {label_counts[1]}")


if __name__ == "__main__":
    main()