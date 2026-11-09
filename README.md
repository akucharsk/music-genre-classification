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

## Jamendo

The Jamendo tab searches the [Jamendo API](https://developer.jamendo.com/v3.0/tracks).
Create a client id at [devportal.jamendo.com](https://devportal.jamendo.com/) and set
`JAMENDO_CLIENT_ID` in the environment or in `.streamlit/secrets.toml` (git-ignored):

```sh
JAMENDO_CLIENT_ID=your_client_id uv run music-genre-classification
```

`.streamlit/secrets.toml`:

```toml
JAMENDO_CLIENT_ID = "your_client_id"
```

## Preprocessing

Download GTZAN from Kaggle and unpack the `genres_original` folder to `data/genres_original`, then run:

```sh
uv run python -m music_genre_classification.data_preprocessing
```

Each 30 s track is split into 10 segments and converted to 13 MFCC coefficients per frame. The result
is written to `data/data.json` (git-ignored). Unreadable files (GTZAN has one corrupt jazz track) are
skipped with a warning.

## Development

```sh
uv run ruff check        # lint (all rules enabled)
uv run ruff format       # format
uv run mypy src tests    # type-check (strict)
uv run pytest            # tests
```

CI runs all of the above on every push to `main` and on pull requests.
