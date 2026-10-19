# Movie studio data

Every file in `raw/` is exactly as downloaded. All cleaning and joining happens in `movie_studio.py` (section 1.2, "Parse and join") and the `lib/` modules it calls. Three sources are too large for the repository, so `interim/` holds the extracts the notebook actually reads, built from those sources by `lib/loading.py`.

## Files the notebook uses

Ranked by how much the analysis leans on each one.

| File | Rows | Used for | Source |
|---|---|---|---|
| `interim/tmdb_titles.parquet` (from `raw/tmdb_full/`, not in the repo) | 9,544 titles with budget and revenue | The base list: budget, revenue, release date, producers, original language, production countries, keywords, IMDb id | TMDB full dump, December 2024 |
| `raw/the_numbers/movie-budgets.csv` | 5,460 films | US release date, budget, domestic and worldwide gross, story source, MPAA rating; first choice for every money figure | The Numbers, via the delafields scrape |
| `interim/imdb_movies.parquet`, `imdb_credits.parquet`, `imdb_ratings.parquet` (from `raw/imdb/`, not in the repo) | 760,000 feature films, 2.3M credits | Genres and runtime for every film; each director's and top-three actor's full filmography, and the ratings of their earlier films | IMDb official datasets |
| `raw/wikidata/film_facts.csv` | one or more rows per film | Film series and number in it, whether the film adapts another work, distributors, MPAA rating | Wikidata, pulled for these films on 19 Oct 2026 |
| `raw/fred/CPIAUCSL.csv` | monthly, Jan 1947 to Sep 2026 | Puts every dollar at 2024 prices | FRED |
| `raw/fred/UMCSENT.csv` | monthly, Nov 1952 to Aug 2026 | Consumer sentiment the month before release | FRED |
| `interim/franchise_lists.parquet` (from `raw/movies_dataset/`, not in the repo) | 45,000 films | TMDB's franchise lists up to 2017, to check the sequel flag | The Movies Dataset |
| `raw/movie_data.csv` | 3,647 films | MPAA rating where the other sources have none | DAO2702X course file |
| `raw/lock5/` | 2,265 films, 2007-2018 | Lead distributor where Wikidata has none | Lock5Data |

## Files not used

| File | What it holds | Why not |
|---|---|---|
| `raw/tmdb/tmdb_5000_movies.csv`, `tmdb_5000_credits.csv` | TMDB's 2017 export of about 5,000 films with cast and crew | The full 2024 dump and IMDb cover the same and more |
| `raw/the_numbers_json/` | The scrape before it was merged into `movie-budgets.csv` | Same data as the CSV |

## Getting the large sources

Only needed to rebuild `interim/`. Put each in the folder named and delete the matching extract.

1. **IMDb** non-commercial datasets, <https://datasets.imdbws.com/>: `title.basics.tsv.gz`, `title.crew.tsv.gz`, `title.principals.tsv.gz` and `title.ratings.tsv.gz` into `raw/imdb/` (about 1.1 GB). Licensed for personal and non-commercial use: <https://developer.imdb.com/non-commercial-datasets/>.
2. **TMDB full dump**, `TMDB_movie_dataset_v11.csv` (500 MB), from <https://huggingface.co/datasets/ada-datadruids/full_tmdb_movies_dataset>, a mirror of the Kaggle "Full TMDB Movies Dataset 2024" by asaniczka, into `raw/tmdb_full/`.
3. **The Movies Dataset**, `movies_metadata.csv`, from <https://huggingface.co/datasets/gaygaaa/TheMoviesDataset>, a mirror of Rounak Banik's Kaggle set, into `raw/movies_dataset/`.

Wikidata needs no download: delete `raw/wikidata/film_facts.csv` and the notebook pulls it again from <https://query.wikidata.org/> in about ten minutes.

## Sources

1. **TMDB**, The Movie Database, <https://www.themoviedb.org/>. Budgets and revenues are entered by users.
2. **The Numbers** budget table, <https://www.the-numbers.com/movie/budgets/all>, scraped by GitHub user delafields on 5 Dec 2019: <https://github.com/delafields/movie-budgets>. Its budgets and grosses come from studios and trade reporting.
3. **IMDb** datasets, refreshed daily by IMDb; ours were downloaded on 19 Oct 2026.
4. **Wikidata** properties used: P345 (IMDb id), P179 (part of the series) with P1545 (series ordinal), P144 (based on), P750 (distributor), P1657 (MPAA film rating).
5. **The Movies Dataset**, TMDB and MovieLens metadata for 45,000 films released up to July 2017.
6. **`movie_data.csv`** came with the DAO2702X course (NUS). Its columns match the Kaggle IMDB 5000 Movie Dataset, scraped from IMDb in 2016: <https://www.kaggle.com/datasets/carolzhangdc/imdb-5000-movie-dataset>.
7. **Lock5** datasets from Lock, Lock, Lock Morgan, Lock and Lock, *Statistics: Unlocking the Power of Data*, as CSVs on the book's site, for example <https://www.lock5stat.com/datasets3e/HollywoodMovies.csv>.
8. **FRED**, the St. Louis Fed: CPIAUCSL, the Consumer Price Index for All Urban Consumers (1982-84 = 100), <https://fred.stlouisfed.org/series/CPIAUCSL>, and UMCSENT, the University of Michigan consumer sentiment index, <https://fred.stlouisfed.org/series/UMCSENT>.

## Units and traps, file by file

### TMDB

- `budget` and `revenue` are in plain dollars, and 0 means unknown. Users sometimes type them in millions (a budget of 50 for a $50M film), which is why the notebook drops budgets under $1M and grosses under $10,000.
- `release_date` is often the first release anywhere, such as a festival, not the US release.
- `genres`, `production_companies` and `keywords` are comma-separated text.

### The Numbers

- Money and dates are text: `"$400,000,000"`, `"Apr 23, 2019"`.
- In about 300 rows the scrape shifted columns, so `story_source`, `production_method` or `MPAA_Rating` holds text like "All Time Domestic Box Office for...".
- It covers only films released up to December 2019.

### IMDb

- Missing values are written `\N`, and fields hold no quotes, so read with `quote_char=None`.
- `title.principals` lists cast and crew in billing order; the notebook keeps the first three actors per film.
- Ratings are a snapshot from the download date, not from the time each film came out.

### Wikidata

- A film appears once per combination of series, distributor and so on, so the notebook groups by film first.
- Distributors include later streaming and home-video distributors, not only the cinema release.

### `movie_data.csv`

- Money is in millions of dollars, `gross` is the US box office only, and budgets of non-US films are in their own currency (*The Host* shows 12,215 million won). The notebook takes only the MPAA rating from it.
- Text saved in the wrong encoding (`PenÃ©lope`) and a byte-order mark: read with `encoding="utf8-lossy"`.

### All sources

Coverage leans toward films people remember. Every source thins out before 1980, and small films that flopped are mostly missing, so the low-budget tier looks better than it is.
