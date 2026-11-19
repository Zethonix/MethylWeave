"""Read the 13-column MethylWeave feature table, without upstream tools."""
from contextlib import contextmanager
import gzip
import hashlib
import os
from itertools import product
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch.utils.data import Dataset

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def _cache_lock(path):
    """Serialize cache creation across processes on Linux and Windows."""
    with path.open("a+b") as lock:
        if os.name == "nt":
            if lock.seek(0, os.SEEK_END) == 0:
                lock.write(b"\0")
                lock.flush()
            lock.seek(0)
            # LK_LOCK has a bounded retry period; wait for long cache builds too.
            import errno
            import time
            while True:
                try:
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    if exc.errno not in (errno.EACCES, errno.EAGAIN, errno.EDEADLK):
                        raise
                    time.sleep(0.1)
        else:
            fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_UN)


KMER = {"".join(bases): i + 5 for i, bases in enumerate(product("ACGT", repeat=5))}
BASE = {base: i for i, base in enumerate("ACGTN")}
MOTIF = {"CG": 0, "CHG": 1, "CHH": 2, "UNK": 3}
FEATURE_DTYPE = np.dtype([
    ("tokens", "<i2", (13,)), ("stats", "<f4", (13, 3)),
    ("samples", "<f4", (13, 16)), ("base", "u1", (13,)),
    ("motif", "u1"), ("label", "<i8"),
])


def motif_id(seq, context):
    context = context.strip().upper()
    if context in MOTIF:
        return MOTIF[context]
    if seq[6] != "C":
        return MOTIF["UNK"]
    a, b = seq[7:9]
    if a == "G":
        return MOTIF["CG"]
    if a in "ACT" and b == "G":
        return MOTIF["CHG"]
    if a in "ACT" and b in "ACT":
        return MOTIF["CHH"]
    return MOTIF["UNK"]


def parse_features(fields):
    seq = fields[6].upper()
    if len(seq) != 13:
        raise ValueError("the sequence must contain 13 positions")
    padded = "CC" + seq + "CC"
    tokens = np.asarray([KMER.get(padded[i:i+5], 1) for i in range(13)], dtype=np.int16)
    means, stds, dwell = ([float(x) for x in fields[i].split(",")] for i in (7, 8, 9))
    lens = np.log1p(np.array(dwell))
    std = np.std(lens)
    normalized = (lens - np.mean(lens)) / std if std > 1e-8 else np.zeros_like(lens)
    stats = np.asarray(np.stack([means, stds, normalized], axis=1), dtype=np.float32)
    if stats.shape != (13, 3):
        raise ValueError("mean, std and dwell must each contain 13 values")
    raw = np.zeros((13, 16), dtype=np.float32)
    for i, segment in enumerate(fields[10].split(";")[:13]):
        if segment:
            values = np.fromstring(segment, sep=",", dtype=np.float32)[:16]
            raw[i, :len(values)] = values
    if not np.isfinite(stats).all() or not np.isfinite(raw).all():
        raise ValueError("nonfinite input feature")
    base_ids = np.asarray([BASE.get(base, BASE["N"]) for base in seq], dtype=np.uint8)
    return tokens, stats, raw, base_ids, motif_id(seq, fields[12]), int(fields[11])


def iter_features(path, *, labeled=False):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            fields = line.strip().split("\t")
            try:
                if len(fields) != 13:
                    raise ValueError("expected 13 tab-delimited fields without a header")
                item = parse_features(fields)
                if item[-1] not in ((0, 1) if labeled else (-1, 0, 1)):
                    raise ValueError("training and validation labels must be 0 or 1; prediction also accepts -1")
            except (ValueError, IndexError) as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
            yield (line_no, fields[0], fields[1], fields[2], fields[4], fields[12]), item


def _cache_source(path):
    st = path.stat()
    return (str(path), st.st_size, st.st_mtime_ns)


class FeatureDataset(Dataset):
    """Disk-backed fixed-width arrays for random access to large split files."""

    def __init__(self, path, cache_dir):
        path = Path(path).resolve()
        source = _cache_source(path)
        key = hashlib.sha256(repr(("methylweave-final-tsv-v1", source)).encode()).hexdigest()
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache = cache_dir / f"{key}.bin"
        self.array = None
        with _cache_lock(cache_dir / f"{key}.lock"):
            if not self.cache.exists():
                block = []
                with NamedTemporaryFile(dir=cache_dir, delete=False) as temp:
                    temp_path = Path(temp.name)
                    try:
                        for _, item in iter_features(path, labeled=True):
                            block.append(item)
                            if len(block) == 4096:
                                np.asarray(block, dtype=FEATURE_DTYPE).tofile(temp)
                                block.clear()
                        np.asarray(block, dtype=FEATURE_DTYPE).tofile(temp)
                        temp.flush()
                        if _cache_source(path) != source:
                            raise ValueError(f"Input changed during cache build: {path}")
                        if temp_path.stat().st_size == 0:
                            raise ValueError(f"Empty feature table: {path}")
                        temp.close()
                        temp_path.replace(self.cache)
                    finally:
                        temp.close()
                        if temp_path.exists():
                            temp_path.unlink()
            n_bytes = self.cache.stat().st_size
            self.length, remainder = divmod(n_bytes, FEATURE_DTYPE.itemsize)
            if remainder or not self.length:
                raise ValueError(f"Invalid feature cache: {self.cache}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.array is None:
            self.array = np.memmap(self.cache, dtype=FEATURE_DTYPE, mode="r", shape=(self.length,))
        return tuple(self.array[idx])

    def __getstate__(self):
        return dict(self.__dict__, array=None)


def collate(rows):
    tokens, stats, samples, bases, motifs, labels = zip(*rows)
    return (
        torch.from_numpy(np.stack(tokens)).long(),
        torch.from_numpy(np.stack(stats)),
        torch.from_numpy(np.stack(samples)),
        torch.from_numpy(np.stack(bases)).long(),
        torch.as_tensor(motifs, dtype=torch.long),
        torch.as_tensor(labels, dtype=torch.long),
    )
