# What should a film studio greenlight?

DAO2702X (Programming for Business Analytics, NUS) group project, team Jupyter.

A studio decides before filming how much to spend, on what kind of film, with which director and lead, and when to release it. We use about 3,500 films to show which of those choices raise the return per dollar, which raise or lower the chance of a flop, and what slate of films a studio should build for the risk it can carry.

## The brief

From Appendix C of the DAO2702 course overview: one practical business problem on real data.

| Part | What | Weight | When |
|---|---|---|---|
| Report | 8-page PDF with no code, plus the notebook as HTML, zipped as `Project_Jupyter` | 30% | Fri 13 Nov 23:59 |
| Presentation | 15 minutes for the whole team, everyone speaks, no scripts, graded individually | 30% | Weeks 11-12 (26 Oct to 6 Nov), slot not booked yet |

The report is graded on data visualization 50%, problem statement 20% (novelty counts), analysis 25% and writing 5%. The presentation comes before the report, so the charts and findings have to exist first.

## The question and the targets

Which choices a studio makes before filming change what a film earns per dollar, and how often it loses money?

| Target | Definition | Meaning |
|---|---|---|
| `gross_multiple` | worldwide gross / production budget | 3.0 means the film took in 3 dollars at the box office for every dollar it cost to make |
| `flop` | `gross_multiple` < 2.5 | the film did not earn back its costs |

The 2.5 line is an assumption. Cinemas keep about half the box office, and marketing costs roughly another half of the production budget, so a film that costs 100 to make costs about 150 in total and needs about 300 at the box office to cover it. Trade rules of thumb run from 2 to 3 times the budget. The main results get checked at 2 and 3 as well.

## Decided

- Only what is known before release goes into the model (19 Oct). IMDb ratings, Facebook likes, opening weekend and theatre counts all come after release. The likes were counted around 2016, years after most films came out, so they partly measure the film's own success. These can still appear in descriptive charts, labelled as post-release.
- Two targets, the average return and the flop risk (19 Oct), because a studio cares about both and a big budget can change one without the other.
- Competition is the number of films released within 30 days (19 Oct), measured against the year, since the sources cover recent years far more completely.
- The split follows time (19 Oct): `dev` is films up to 2012, `test` is 2013-2015, scored once, and `reserve` is 2016, scored once at the end. The studio always judges a new film from the ones before it. Three test years because one year holds only about 150 films.
- One model, proposed and waiting for a go (19 Oct): a linear regression on the log of the multiple, with an error spread that depends on the budget. Its coefficients say how much each choice moves the return, and its error spread gives the chance of a flop. The predictions feed a Monte Carlo simulation of film slates.

## Where it stands

Data, split and exploration are done in `movie_studio.ipynb`. 3,464 of the 3,627 unique films in the course dataset keep a release date, a dollar budget and a worldwide gross. What the exploration found on `dev`, ranked by how much it should change the model:

1. Budget sets the risk more than the return. From the cheapest budget tier to the most expensive, the 10th-90th percentile range of multiples narrows from 0.09-13.6 to 0.95-5.5, while the median only rises from about 1.9 to 2.7. So the model needs an error spread that depends on the budget.
2. Films before 1980 and budgets under $1M are not like the rest. Pre-1980 films flopped 16% of the time against 60% after, because the sources kept only the remembered ones, and 55 budgets under $1M look like errors.
3. Release month is the largest choice after budget. Summer and holiday releases flop 48-53% of the time, spring and autumn releases 65-69%.
4. Genre moves return and risk together, and depends on the budget. Animation, family and horror do best, war, sport and history worst, romance is average. Horror's edge holds only for cheap films.
5. A proven director or lead lowers the flop rate by about 11 points, by shifting the average rather than narrowing the range.
6. Crowding matters little, and only for mid-budget films, once it is measured against the year.
7. Independents flop more (69%) and spread wider than any major, partly because their budgets are small.

### Next

1. Martin's go on the cleaning (drop films before 1980 and budgets under $1M) and on the one model.
2. Fit the model, check it on `test`, then the slate simulation.
3. Book the presentation slot.
4. Write the report and export the notebook as HTML.

## Hypotheses

From the outline, with what the exploration says so far:

1. Bigger budgets lower the risk, not just raise the revenue. Holds (finding 1).
2. Romance is cheap, easy money. Does not hold: romance is average, and horror beats it when the budget is small.
3. Releasing into a crowded month cuts the multiple. Weak once measured against the year (finding 6).
4. A director or lead with a strong track record lowers the flop chance more than it raises the upside. Holds (finding 5).
5. Big and small studios should pick different slates, because they can carry different amounts of risk. Comes from the slate simulation.

## Charts, ranked

Visuals are half the report grade. One accent colour for the group being discussed, grey for the rest, red only for the flop line. Every chart is a different type and answers one question.

| Rank | Chart | Type | Question |
|---|---|---|---|
| 1 | Budget vs worldwide gross, log axes, with the 1x and 2.5x lines | Scatter | Does spending more pay? Opens the report |
| 2 | Gross multiple by budget tier | Violin | Does a bigger budget narrow the spread? |
| 3 | Median multiple vs flop rate by genre, bubble size = number of films | Bubble scatter | Where do romance and horror sit? |
| 4 | Flop rate by genre and budget tier | Heatmap | Which combinations are safest? |
| 5 | Median multiple by number of competitors within 30 days | Line with band | Does a crowded window hurt? |
| 6 | Domestic vs international share by genre | Stacked bar | Which genres depend on foreign markets? |
| 7 | Outcome by director track record | Box plots | Is proven talent safer? |
| 8 | Regression coefficients with 95% intervals | Dot-and-whisker | Which choices matter most with the others held fixed? |
| 9 | Predicted range for three example pitches | Interval plot | What should a studio expect from pitch A, B and C? |
| 10 | Slate risk-return frontier | Scatter with frontier line | Which slate fits a big studio, and which a small one? |

Charts 1, 2, 3, 8 and 10 are the core. If pages run out, 4 and 6 go first.

## The slate simulation

A studio has, say, $300M to spend in a year. It can fund one $150M film plus three $50M films, or ten $30M horror and comedy films, and so on. For each slate, draw 10,000 outcomes from the model's predicted ranges, turn gross into studio profit with the 2.5 rule, and record the expected profit and the chance the whole slate loses money. Plotted together, the slates form a risk-return frontier like a stock portfolio's, and a big studio that can absorb a bad year picks a different point on it than a small one.

## Limitations to state in the report

1. Box office is not profit. Streaming, TV and home video revenue are missing, and so are real marketing budgets. The 2.5 rule stands in for both.
2. Actor and director pay is not public, so track record stands in for star power.
3. Genre is a rough stand-in for effects spending.
4. The sources favour films big enough to be tracked, so small films that vanished are missing and the low-budget tier looks better than it is.
5. These are associations, not causes. A big budget may go with lower risk because studios only give big budgets to projects they already trust.

## Files

| Path | What |
|---|---|
| `movie_studio.ipynb`, `movie_studio.py` | The analysis, the same notebook in two formats, paired with jupytext |
| `lib/` | Parsing, feature and schema code the notebook imports |
| `data/README.md` | Every data file, its source link, units and traps |
| `data/raw/` | The downloaded data, not in git |
| `docs/outline-2026-10-19.pdf` | The first outline, as sent to the group |
