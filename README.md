# Music Genre Classification

Streamlit app for classifying the genre of an audio clip.

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
