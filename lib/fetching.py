"""Pulling pre-release facts about films from Wikidata's public SPARQL endpoint."""

import time
from pathlib import Path
from typing import cast

import polars as pl
import requests

WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
USER_AGENT = "movie-studio-course-project/0.1 (NUS DAO2702X student project)"

# for each IMDb id: the film series it belongs to and its number in it, whether it is based on another work,
# its distributors and its MPAA rating; one row per combination, so a film can appear more than once
QUERY = """
SELECT ?imdb ?seriesLabel ?ordinal ?basedOn ?distributorLabel ?mpaaLabel WHERE {
  VALUES ?imdb { %s }
  ?film wdt:P345 ?imdb .
  OPTIONAL { ?film p:P179 ?seriesStatement . ?seriesStatement ps:P179 ?series .
             OPTIONAL { ?seriesStatement pq:P1545 ?ordinal . } }
  OPTIONAL { ?film wdt:P144 ?basedOn . }
  OPTIONAL { ?film wdt:P750 ?distributor . }
  OPTIONAL { ?film wdt:P1657 ?mpaa . }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
}
"""


def query_wikidata(
    session: requests.Session, imdb_ids: list[str], attempts: int
) -> str | None:
    """The CSV answer for one batch of ids, or None when every attempt failed or timed out."""
    values = " ".join(f'"{imdb_id}"' for imdb_id in imdb_ids)
    for attempt in range(attempts):
        try:
            response = session.get(
                WIKIDATA_SPARQL, params={"query": QUERY % values}, timeout=90
            )
            if response.status_code == 200:
                return response.text
        except requests.RequestException:
            pass
        time.sleep(5 * (attempt + 1))
    return None


def fetch_wikidata_facts(
    imdb_ids: list[str], cache_path: Path, batch_size: int = 200
) -> pl.DataFrame:
    """Wikidata facts for every IMDb id, read from `cache_path` when it exists, otherwise pulled and saved there.

    Columns: imdb, seriesLabel, ordinal, basedOn, distributorLabel, mpaaLabel, all text. Ids are sent in
    batches of `batch_size` over one connection with a pause between requests, as the endpoint asks of
    automated clients; a batch that keeps failing is retried as four smaller ones, since heavy batches can
    time out. Each batch is saved as it arrives, so an interrupted pull resumes where it stopped.
    """
    if cache_path.exists():
        return pl.read_csv(cache_path, infer_schema=False)
    batch_dir = cache_path.parent / f"{cache_path.stem}_batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"Accept": "text/csv", "User-Agent": USER_AGENT})
    for start in range(0, len(imdb_ids), batch_size):
        batch_path = batch_dir / f"{start:06d}.csv"
        if batch_path.exists():
            continue
        batch = imdb_ids[start : start + batch_size]
        answer = query_wikidata(session, batch, attempts=3)
        if answer is None:
            quarter = -(-len(batch) // 4)
            pieces = [
                query_wikidata(session, batch[i : i + quarter], attempts=4)
                for i in range(0, len(batch), quarter)
            ]
            if any(piece is None for piece in pieces):
                raise RuntimeError(
                    f"Wikidata did not answer the batch starting at id {start}"
                )
            # every piece answered, checked just above
            answered = [cast(str, piece) for piece in pieces]
            header = answered[0].splitlines()[0]
            rows = [line for piece in answered for line in piece.splitlines()[1:]]
            answer = "\n".join([header, *rows]) + "\n"
        batch_path.write_text(answer)
        time.sleep(1)
    facts = pl.concat(
        [
            pl.read_csv(path, infer_schema=False)
            for path in sorted(batch_dir.glob("*.csv"))
        ],
        how="diagonal",
    )
    facts.write_csv(cache_path)
    return facts
