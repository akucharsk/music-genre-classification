# Copyright (C) 2026 Bartłomiej Kózka
"""Extract MFCC features from the GTZAN dataset into a JSON file."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import librosa

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy as np
    from numpy.typing import NDArray

LOGGER = logging.getLogger(__name__)

DEFAULT_DATASET_DIR = Path("data/genres_original")
DEFAULT_OUTPUT_PATH = Path("data/data.json")
AUDIO_PATTERN = "*.wav"

SAMPLE_RATE = 22050
TRACK_DURATION_SECONDS = 30
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION_SECONDS


class DatasetError(Exception):
    """Raised when the dataset directory is missing or contains no genres."""


class MfccData(TypedDict):
    """Serializable dataset: genre names, one label per segment, one MFCC matrix per segment."""

    genres: list[str]
    labels: list[int]
    mfcc: list[list[list[float]]]


@dataclass(frozen=True, slots=True)
class MfccConfig:
    """Parameters of the MFCC extraction.

    Attributes:
        num_mfcc: Number of MFCC coefficients per frame.
        n_fft: Length of the FFT window.
        hop_length: Number of samples between consecutive frames.
        num_segments: Number of equal segments each track is split into.
    """

    num_mfcc: int = 13
    n_fft: int = 2048
    hop_length: int = 512
    num_segments: int = 10

    def __post_init__(self) -> None:
        """Validate that the configuration is usable.

        Raises:
            ValueError: If any parameter is not positive.
        """
        if min(self.num_mfcc, self.n_fft, self.hop_length, self.num_segments) < 1:
            msg = "All MFCC parameters must be positive."
            raise ValueError(msg)

    @property
    def samples_per_segment(self) -> int:
        """Number of audio samples in one segment."""
        return SAMPLES_PER_TRACK // self.num_segments

    @property
    def frames_per_segment(self) -> int:
        """Number of MFCC frames a full-length segment produces."""
        return math.ceil(self.samples_per_segment / self.hop_length)


def find_genres(dataset_dir: Path) -> list[Path]:
    """List genre directories in alphabetical order.

    Args:
        dataset_dir: Directory containing one sub-directory per genre.

    Returns:
        Sorted genre directories.

    Raises:
        DatasetError: If the directory does not exist or has no genre sub-directories.
    """
    if not dataset_dir.is_dir():
        msg = f"Dataset directory not found: {dataset_dir}"
        raise DatasetError(msg)
    genres = sorted(path for path in dataset_dir.iterdir() if path.is_dir())
    if not genres:
        msg = f"No genre directories in: {dataset_dir}"
        raise DatasetError(msg)
    return genres


def split_segments(
    signal: NDArray[np.floating], config: MfccConfig
) -> Iterator[NDArray[np.floating]]:
    """Cut a signal into equal-length segments, dropping an incomplete tail.

    Args:
        signal: Mono audio samples.
        config: Extraction parameters.

    Yields:
        Consecutive segments of exactly ``config.samples_per_segment`` samples.
    """
    size = config.samples_per_segment
    for index in range(config.num_segments):
        segment = signal[index * size : (index + 1) * size]
        if len(segment) == size:
            yield segment


def segment_mfcc(segment: NDArray[np.floating], config: MfccConfig) -> list[list[float]]:
    """Compute the MFCC matrix of one segment.

    Args:
        segment: Mono audio samples of one segment.
        config: Extraction parameters.

    Returns:
        Matrix of shape (frames, coefficients) as nested lists.
    """
    mfcc = librosa.feature.mfcc(
        y=segment,
        sr=SAMPLE_RATE,
        n_mfcc=config.num_mfcc,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
    )
    return mfcc.T.tolist()  # type: ignore[no-any-return]


def extract_track(path: Path, config: MfccConfig) -> list[list[list[float]]]:
    """Extract MFCC matrices for every full segment of one audio file.

    Args:
        path: Audio file to load.
        config: Extraction parameters.

    Returns:
        One MFCC matrix per segment with the expected number of frames.
    """
    signal, _ = librosa.load(path, sr=SAMPLE_RATE)
    matrices = (segment_mfcc(segment, config) for segment in split_segments(signal, config))
    return [matrix for matrix in matrices if len(matrix) == config.frames_per_segment]


def build_dataset(dataset_dir: Path, config: MfccConfig) -> MfccData:
    """Build the MFCC dataset for all genres found in a directory.

    Tracks that cannot be decoded (GTZAN contains a known corrupt file) are skipped and logged.

    Args:
        dataset_dir: Directory containing one sub-directory of audio files per genre.
        config: Extraction parameters.

    Returns:
        Genre names, per-segment labels and per-segment MFCC matrices.
    """
    data = MfccData(genres=[], labels=[], mfcc=[])
    for label, genre_dir in enumerate(find_genres(dataset_dir)):
        data["genres"].append(genre_dir.name)
        LOGGER.info("Processing genre: %s", genre_dir.name)
        for track in sorted(genre_dir.glob(AUDIO_PATTERN)):
            try:
                matrices = extract_track(track, config)
            except Exception:
                LOGGER.warning("Skipping unreadable track: %s", track, exc_info=True)
                continue
            data["mfcc"].extend(matrices)
            data["labels"].extend([label] * len(matrices))
    return data


def save_dataset(data: MfccData, output_path: Path) -> None:
    """Write the dataset as JSON, creating parent directories when needed.

    Args:
        data: Dataset to serialize.
        output_path: Destination file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file)


def main() -> None:
    """Extract MFCC features from the default dataset directory."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = build_dataset(DEFAULT_DATASET_DIR, MfccConfig())
    save_dataset(data, DEFAULT_OUTPUT_PATH)
    LOGGER.info("Saved %d segments to %s", len(data["labels"]), DEFAULT_OUTPUT_PATH)


if __name__ == "__main__":
    main()
