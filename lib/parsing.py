"""Text fixes needed to read the four movie sources and match films across them."""

import re
import unicodedata

import polars as pl

# parent studio -> name fragments that identify it in TMDB's production-company lists
MAJOR_STUDIOS: dict[str, tuple[str, ...]] = {
    "Disney": (
        "walt disney",
        "pixar",
        "marvel studios",
        "lucasfilm",
        "touchstone",
        "buena vista",
    ),
    "Warner": ("warner bros", "new line", "castle rock"),
    "Universal": (
        "universal pictures",
        "focus features",
        "illumination",
        "working title",
    ),
    "Paramount": ("paramount", "nickelodeon movies"),
    "Sony": ("columbia pictures", "sony pictures", "screen gems", "tristar"),
    "Fox": (
        "twentieth century fox",
        "20th century fox",
        "fox searchlight",
        "fox 2000",
        "regency",
    ),
    "Lionsgate": ("lionsgate", "lions gate", "summit entertainment"),
    "DreamWorks": ("dreamworks",),
    "MGM": ("metro-goldwyn-mayer", "united artists"),
}


def fix_mojibake(text: str) -> str:
    """Undo UTF-8 text that was decoded as Latin-1 ("PenÃ©lope" -> "Penélope"); other text is returned as is."""
    try:
        return text.encode("latin-1").decode("utf-8")
    except UnicodeError:
        return text


def title_key(title: str) -> str:
    """Lowercase ASCII letters and digits only, with a leading or trailing "the" dropped, so spellings of one title match."""
    ascii_title = (
        unicodedata.normalize("NFKD", fix_mojibake(title))
        .encode("ascii", "ignore")
        .decode()
    )
    without_the = re.sub(r"^the |, the$", "", ascii_title.strip().lower())
    return re.sub(r"[^a-z0-9]", "", without_the)


def major_studio(production_companies: list[str]) -> str:
    """The first major parent studio among a film's production companies, or "Independent" when none is a major."""
    for company in production_companies:
        for studio, fragments in MAJOR_STUDIOS.items():
            if any(fragment in company.lower() for fragment in fragments):
                return studio
    return "Independent"


def attach_closest_year(
    films: pl.DataFrame, source: pl.DataFrame, max_year_gap: int
) -> pl.DataFrame:
    """Add `source`'s columns to `films`, matching on title_key and taking the source row whose year is closest.

    Both frames need title_key and year; `films` also needs film_id. A source row more than `max_year_gap`
    years away does not count, and a film without a match keeps nulls in the added columns.
    """
    closest = (
        films.select("film_id", "title_key", "year")
        .join(source, on="title_key", how="inner", suffix="_source")
        .with_columns(year_gap=(pl.col("year") - pl.col("year_source")).abs())
        .filter(pl.col("year_gap") <= max_year_gap)
        .sort("film_id", "year_gap")
        .unique("film_id", keep="first", maintain_order=True)
        .drop("title_key", "year", "year_source", "year_gap")
    )
    return films.join(closest, on="film_id", how="left")
