"""Reading the large IMDb files: only feature films, their directors and their top-billed actors are kept."""

import subprocess
from pathlib import Path

import polars as pl

IMDB_NULL = r"\N"


def filtered_tsv(gz_path: Path, awk_filter: str, out_path: Path) -> Path:
    """Decompress `gz_path` through an awk filter into `out_path`, unless that file already exists.

    `awk_filter` is an awk condition on tab-separated fields; the header line is always kept.
    """
    if not out_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        command = f"gzip -dc '{gz_path}' | awk -F'\\t' 'NR == 1 || ({awk_filter})' > '{out_path}'"
        subprocess.run(command, shell=True, check=True)
    return out_path


def read_tsv(path: Path) -> pl.DataFrame:
    """Read an IMDb TSV with every column as text and IMDb's \\N as null; quotes are not special in these files."""
    return pl.read_csv(
        path, separator="\t", quote_char=None, null_values=IMDB_NULL, infer_schema=False
    )


def load_imdb(
    imdb_dir: Path, interim_dir: Path
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Feature films, the people credited on them, and their ratings.

    Returns `movies` (tconst, imdb_title, year, runtime_min, imdb_genres), `credits` (tconst, nconst, role,
    billing, where role is director or actor and billing is 1 to 3 for the top-billed actors, 0 for
    directors) and `ratings` (tconst, imdb_rating, imdb_votes). The three tables are saved as parquet in
    `interim_dir` and read from there when present, since building them takes about two minutes.
    """
    saved = [
        interim_dir / f"imdb_{name}.parquet"
        for name in ("movies", "credits", "ratings")
    ]
    if all(path.exists() for path in saved):
        movies, credits, ratings = (pl.read_parquet(path) for path in saved)
        return movies, credits, ratings
    movies = read_tsv(
        filtered_tsv(
            imdb_dir / "title.basics.tsv.gz",
            '$2 == "movie"',
            interim_dir / "movies.tsv",
        )
    ).select(
        "tconst",
        imdb_title=pl.col("primaryTitle"),
        year=pl.col("startYear").cast(pl.Int32, strict=False),
        runtime_min=pl.col("runtimeMinutes").cast(pl.Int32, strict=False),
        imdb_genres=pl.col("genres"),
    )
    directors = (
        read_tsv(
            filtered_tsv(
                imdb_dir / "title.crew.tsv.gz",
                '$2 != "\\\\N"',
                interim_dir / "crew.tsv",
            )
        )
        .join(movies.select("tconst"), on="tconst", how="semi")
        .select("tconst", nconst=pl.col("directors").str.split(","))
        .explode("nconst")
        .with_columns(role=pl.lit("director"), billing=pl.lit(0, dtype=pl.Int32))
    )
    actors = (
        read_tsv(
            filtered_tsv(
                imdb_dir / "title.principals.tsv.gz",
                '($4 == "actor" || $4 == "actress") && $2 <= 8',
                interim_dir / "actors.tsv",
            )
        )
        .join(movies.select("tconst"), on="tconst", how="semi")
        .with_columns(ordering=pl.col("ordering").cast(pl.Int32))
        .with_columns(
            billing=pl.col("ordering").rank("ordinal").over("tconst").cast(pl.Int32)
        )
        .filter(pl.col("billing") <= 3)
        .select("tconst", "nconst", role=pl.lit("actor"), billing="billing")
    )
    ratings = read_tsv(imdb_dir / "title.ratings.tsv.gz").select(
        "tconst",
        imdb_rating=pl.col("averageRating").cast(pl.Float64),
        imdb_votes=pl.col("numVotes").cast(pl.Int64),
    )
    credits = pl.concat([directors, actors])
    for frame, path in zip((movies, credits, ratings), saved, strict=True):
        frame.write_parquet(path)
    for tsv in ("movies.tsv", "crew.tsv", "actors.tsv"):
        (interim_dir / tsv).unlink()
    return movies, credits, ratings


def load_tmdb(tmdb_csv: Path, interim_dir: Path) -> pl.DataFrame:
    """TMDB titles that are released, carry an IMDb id and have a budget and a revenue above 0.

    Keeps id-level columns only (imdb_id, title, release_date, budget, revenue, original_language,
    production_countries, production_companies, keywords, vote_count) and saves them to
    `interim_dir/tmdb_titles.parquet`, read from there when present, since the full dump is 500 MB.
    """
    saved = interim_dir / "tmdb_titles.parquet"
    if saved.exists():
        return pl.read_parquet(saved)
    titles = (
        pl.read_csv(tmdb_csv, infer_schema_length=None)
        .filter(
            pl.col("status") == "Released",
            pl.col("imdb_id").str.starts_with("tt"),
            pl.col("budget") > 0,
            pl.col("revenue") > 0,
        )
        .select(
            "imdb_id",
            "title",
            "release_date",
            "budget",
            "revenue",
            "original_language",
            "production_countries",
            "production_companies",
            "keywords",
            "vote_count",
        )
    )
    interim_dir.mkdir(parents=True, exist_ok=True)
    titles.write_parquet(saved)
    return titles


def load_franchise_lists(movies_metadata_csv: Path, interim_dir: Path) -> pl.DataFrame:
    """The Movies Dataset's franchise lists: imdb_id, collection name and release date, one row per film.

    Saved to `interim_dir/franchise_lists.parquet` and read from there when present.
    """
    saved = interim_dir / "franchise_lists.parquet"
    if saved.exists():
        return pl.read_parquet(saved)
    franchises = (
        pl.read_csv(movies_metadata_csv, infer_schema_length=0)
        .filter(pl.col("imdb_id").str.starts_with("tt"))
        .select(
            "imdb_id",
            collection=pl.col("belongs_to_collection").str.extract(
                r"'name': ['\"](.+?)['\"], '"
            ),
            md_release_date=pl.col("release_date").str.to_date(strict=False),
        )
        .unique("imdb_id")
    )
    interim_dir.mkdir(parents=True, exist_ok=True)
    franchises.write_parquet(saved)
    return franchises
