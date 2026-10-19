# Music Genre Classification

Streamlit app for classifying the genre of an audio clip.

## Dataset

We use [GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)
from Kaggle: 1000 audio tracks, 30 seconds each, evenly split across 10 genres (blues, classical,
country, disco, hiphop, jazz, metal, pop, reggae, rock). The raw data is not stored in the repository
(`data/` and `models/` are git-ignored), download it from Kaggle and place it in `data/`.

## Milestones

- [x] 1. Dataset selection and repository setup (uv, ruff, mypy strict, pytest, pre-commit, CI, Dependabot)
- [ ] 2. Preprocessing
- [ ] 3. Model training
- [ ] 4. App prototype
- [ ] 5. Deployment

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pre-commit install
```

## Run

```sh
uv run music-genre-classification
# or
uv run streamlit run src/music_genre_classification/app.py
```

## Development

```sh
uv run ruff check        # lint (all rules enabled)
uv run ruff format       # format
uv run mypy src tests    # type-check (strict)
uv run pytest            # tests
```

CI runs all of the above on every push to `main` and on pull requests.
