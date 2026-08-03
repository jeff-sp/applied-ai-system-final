# 🎵 HarmonyRanker — Music Recommender Simulation

**Original project:** *Music Recommender Simulation*
The original goal was to represent songs and a listener's "taste profile" as structured data, design a scoring rule that turns that data into ranked recommendations, and evaluate where the rule succeeds and fails. Its capabilities were: load a small song catalog from CSV, score each song against a user profile using genre, mood, and energy, and return the top-k matches. This final version completes that implementation, adds human-readable explanations for every recommendation, adds a multi-profile demo harness, and adds a test suite.

---

## Summary

HarmonyRanker is a **content-based** music recommender. It does not listen to audio and it does not use collaborative filtering — it compares each song's labeled attributes (genre, mood, energy) against a user's stated preferences, produces a numeric score, and ranks the catalog.

Why it matters: it is a working, fully legible model of how commercial recommenders actually behave. Every number in the ranking is traceable to a specific weight in a specific line of code, which makes it a good vehicle for seeing how a single weighting decision can quietly dominate an entire system's output — the exact failure mode documented in [Testing Summary](#testing-summary) and [model_card.md](model_card.md).

---

## Architecture Overview

The system diagram lives at [diagrams/architecture.mmd](diagrams/architecture.mmd) (Mermaid). It maps the recommender onto a retrieval-augmented pattern:

- **Input** — a user taste profile (`favorite_genre`, `favorite_mood`, `target_energy`) enters through the CLI runner, [src/main.py](src/main.py).
- **Retriever** ([src/retriever.py](src/retriever.py)) — owns the knowledge base. `load_songs()` parses, type-casts and validates `data/songs.csv`; `evaluate_song()` scores each song against the profile *and records the evidence for that score*; `rank_songs()` sorts descending and cuts to top-k. Scoring (a per-song number) stays separate from ranking (the sort-and-cut policy) so either can change without touching the other.
- **Retrieved records** — the retriever does not hand back bare songs. Each result is a `RetrievedSong` carrying the song, its score, a `SignalMatch` per signal (hit *or* miss), a confidence value, and a confidence band.
- **Explanation Agent** ([src/explainer.py](src/explainer.py)) — the generation half. It never touches the catalog; it sees only the retrieved record and builds its sentences from that evidence. Retrieval doesn't decorate the answer, it *is* the answer: the wording, the hedging, and the caveats are all derived from which signals matched. Generation is template-based and deterministic — no model call, no network, no API key — which is what makes the output testable.
- **Guardrails** ([src/guardrails.py](src/guardrails.py)) — three layers: malformed catalog rows are skipped and logged instead of crashing the run or poisoning the ranking; invalid profiles are rejected before scoring (an out-of-range `target_energy` is clamped and logged); and every claim the agent wants to make is checked against the retrieved record before it can be printed. An unsupported claim is dropped; if no claim survives, the explanation is withheld rather than invented.
- **Logging** ([src/logging_setup.py](src/logging_setup.py)) — every stage logs to `logs/run.log` (DEBUG) with warnings surfaced on the console. `--log-level INFO` shows the pipeline narrating itself.
- **Evaluation** — 63 automated tests across five files, plus a per-run reliability report (average confidence, low-confidence share, grounding rate) and human review of seven contrasting profiles. See [Testing Summary](#testing-summary), and [Reproducible Execution Evidence](#reproducible-execution-evidence) for the captured runs behind every number in it.

The class structure is in [diagrams/uml.mmd](diagrams/uml.mmd): `Song`, `UserProfile`, and `Recommender`.

---

## Setup Instructions

**Requirements:** Python 3.8+

1. Clone and enter the repo:

```bash
git clone https://github.com/jeff-sp/applied-ai-system-final.git && cd applied-ai-system-final
```

2. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv && source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

> The recommender itself uses only the Python standard library — `pip install` is needed for the test runner, not for the app.

4. Run the demo from the repo root:

```bash
python3 -m src.main
```

> Run it from the repo root — the catalog is resolved at the relative path `data/songs.csv`. (`python3 src/main.py` also works.)

5. Run the test suite:

```bash
python3 -m pytest -q
```

Expected: `63 passed`.

### CLI options

```bash
python3 -m src.main --profile lofi -k 3 --log-level INFO
```

| Flag | Default | What it does |
|---|---|---|
| `--profile` | `all` | One of `pop`, `lofi`, `rock`, `metal`, `city pop`, `jazz`, `funk`, or `all`. Quote the ones with a space: `--profile "city pop"` |
| `-k` | `5` | How many songs to recommend |
| `--catalog` | `data/songs.csv` | Point at a different catalog CSV |
| `--log-level` | `WARNING` | Console verbosity; `logs/run.log` always gets the full DEBUG trail |

### Trying your own profile

Add an entry to `PROFILES` in [src/main.py](src/main.py). A profile needs exactly three keys:

```python
my_profile = {
    "favorite_genre": "jazz",     # must match a genre string in data/songs.csv
    "favorite_mood": "relaxed",
    "target_energy": 0.35,        # 0.0 (calm) to 1.0 (intense)
}
```

Anything else is rejected by the profile guardrail with a message naming the offending field, rather than silently producing confident nonsense.

---

## Sample Interactions

Three of the seven demo profiles are shown below — real output against the 203-song catalog. Every sentence after the score line is generated by the Explanation Agent from the retrieved evidence — nothing is hard-coded per song. Each block is verbatim except for the leading `Loaded 203 songs from data/songs.csv` line and the trailing run report, which are trimmed here and shown in full under [Reproducible Execution Evidence](#reproducible-execution-evidence).

### Example 1 — High-energy pop

**Input:** `{"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}`

```
$ python3 -m src.main --profile pop -k 3

====================================================================
  HIGH-ENERGY POP
  profile: genre=pop · mood=happy · energy=0.8
  average confidence: 0.92
====================================================================

  1. Can't Stop the Feeling!  —  Justin Timberlake
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.

  2. Levitating  —  Dua Lipa
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.78) sits close to your target of 0.80.

  3. Run the World (Girls)  —  Beyoncé
     score 3.40 · confidence 0.78 (high)
     Strong match (confidence 0.78): recommended because it's pop, your favorite genre and its energy (0.85) sits close to your target of 0.80. Caveat: the mood is confident, not happy.
```

Rank 3 is where the genre-over-mood weighting shows itself: a `pop`/`confident` track (1.5 + 0 + 1.90 = 3.40) outranks every `indie pop`/`happy` candidate, which can only reach 0.5 + energy. Under the earlier flat 1.0/1.0 weighting those two were interchangeable. The agent still names the mood miss as a caveat rather than letting it read as a full match.

### Example 2 — Chill lofi

**Input:** `{"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.3}`

```
$ python3 -m src.main --profile lofi -k 3

====================================================================
  CHILL LOFI
  profile: genre=lofi · mood=chill · energy=0.3
  average confidence: 0.97
====================================================================

  1. Staying There  —  L'Indécis
     score 3.90 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.35) sits close to your target of 0.30.

  2. Soulful  —  L'Indécis
     score 3.84 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.38) sits close to your target of 0.30.

  3. Aruarian Dance  —  Nujabes
     score 3.76 · confidence 0.96 (high)
     Strong match (confidence 0.96): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.42) sits close to your target of 0.30.
```

All three are full three-signal matches, and the top two are the same artist which is the no-diversity-control limitation listed below showing up in practice.

### Example 3 — Intense rock

**Input:** `{"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.7}`

```
$ python3 -m src.main --profile rock -k 3

====================================================================
  INTENSE ROCK
  profile: genre=rock · mood=intense · energy=0.7
  average confidence: 0.84
====================================================================

  1. Search and Destroy  —  The Stooges
     score 3.62 · confidence 0.94 (high)
     Strong match (confidence 0.94): recommended because it's rock, your favorite genre, the mood is intense, exactly what you wanted and its energy (0.89) sits close to your target of 0.70.

  2. Mr. Brightside  —  The Killers
     score 3.58 · confidence 0.80 (high)
     Strong match (confidence 0.80): recommended because it's rock, your favorite genre and the mood is intense, exactly what you wanted. Caveat: its energy (0.91) is off your target of 0.70.

  3. Sweet Child O' Mine  —  Guns N' Roses
     score 3.48 · confidence 0.79 (high)
     Strong match (confidence 0.79): recommended because it's rock, your favorite genre and its energy (0.71) sits close to your target of 0.70. Caveat: the mood is happy, not intense.
```

Rank 2 is the interesting one: it matches genre *and* mood but misses energy by 0.21 — just past the 0.20 threshold that counts energy as "close" — so the agent flags energy as the caveat while still calling it a strong match. Compare rank 3, which trades the mood match for a near-perfect energy fit and lands lower. This profile is the hardest of the seven for the catalog and still averages 0.84; at 53 songs it averaged 0.79, and at 18 songs its rank 3 was a **low-confidence 0.43** stretch pick. That is not straightforwardly an improvement — see [Testing Summary](#testing-summary).

The low band never fires for any of the seven demo profiles — not for `metal`/`angry` (0.90 average, 5 of 5 high) and not for the thinnest ones. It takes a *double* miss to reach it: `polka` — a genre with **zero** songs in the catalog — still returns 5 of 5 *medium* at 0.63 when the requested mood exists, and only drops to 5 of 5 *low* at 0.43 when the mood is absent too. Both cases are captured under [Uncovered genre](#e6--uncovered-genre-polka).

---

## Reproducible Execution Evidence

Every block in this section is verbatim captured output from this repo — nothing is retyped or tidied. The system is deterministic (pure standard library, no model call, no network, no randomness), so re-running any command reproduces its block exactly, apart from the timestamps in `logs/run.log` and pytest's own timing line.

Reading the blocks:

- A line starting with `$` is the command; everything under it is that command's output.
- `[exit N]` is `echo $?` immediately after the command. `0` = ran, `1` = a fatal condition handled cleanly.
- Log output goes to **stderr**, program output to **stdout**. Blocks were captured with `2>&1`, so log lines may appear grouped ahead of the stdout they relate to; in a live terminal they interleave.
- Captured on Python 3.8.17 (`python3 --version`), macOS, from the repo root.

### E1 — Test suite

```bash
python3 -m pytest -q
```

```
...............................................................          [100%]
63 passed in 0.36s
[exit 0]
```

63 tests across 5 files. The timing figure is the only part of this block that varies between runs.

### E2 — Full demo run (all seven profiles)

```bash
python3 -m src.main -k 5
```

This is the run behind every headline number in the [Testing Summary](#testing-summary). Full transcript is 213 lines; the first profile, one middle profile and the run report are shown, with the cuts marked.

```
$ python3 -m src.main -k 5

Loaded 203 songs from data/songs.csv

====================================================================
  HIGH-ENERGY POP
  profile: genre=pop · mood=happy · energy=0.8
  average confidence: 0.86
====================================================================

  1. Can't Stop the Feeling!  —  Justin Timberlake
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.

  2. Levitating  —  Dua Lipa
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.78) sits close to your target of 0.80.

  3. Run the World (Girls)  —  Beyoncé
     score 3.40 · confidence 0.78 (high)
     Strong match (confidence 0.78): recommended because it's pop, your favorite genre and its energy (0.85) sits close to your target of 0.80. Caveat: the mood is confident, not happy.

  4. Higher Love  —  Kygo, Whitney Houston
     score 3.34 · confidence 0.77 (high)
     Strong match (confidence 0.77): recommended because it's pop, your favorite genre and its energy (0.72) sits close to your target of 0.80. Caveat: the mood is hopeful, not happy.

  5. Don't Stop the Music  —  Rihanna
     score 3.30 · confidence 0.76 (high)
     Strong match (confidence 0.76): recommended because it's pop, your favorite genre and its energy (0.90) sits close to your target of 0.80. Caveat: the mood is energetic, not happy.

====================================================================

… (cut: CHILL LOFI, INTENSE ROCK — 56 lines)

====================================================================
  ANGRY METAL
  profile: genre=metal · mood=angry · energy=0.9
  average confidence: 0.90
====================================================================

  1. Angel of Death  —  Slayer
     score 3.88 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's metal, your favorite genre, the mood is angry, exactly what you wanted and its energy (0.96) sits close to your target of 0.90.

  2. Master of Puppets  —  Metallica
     score 3.86 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's metal, your favorite genre, the mood is angry, exactly what you wanted and its energy (0.97) sits close to your target of 0.90.

  3. Raining Blood  —  Slayer
     score 3.84 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's metal, your favorite genre, the mood is angry, exactly what you wanted and its energy (0.98) sits close to your target of 0.90.

  4. Rise Today  —  Alter Bridge
     score 3.50 · confidence 0.79 (high)
     Strong match (confidence 0.79): recommended because it's metal, your favorite genre and its energy (0.90) sits close to your target of 0.90. Caveat: the mood is hopeful, not angry.

  5. Hallowed Be Thy Name  —  Iron Maiden
     score 3.46 · confidence 0.79 (high)
     Strong match (confidence 0.79): recommended because it's metal, your favorite genre and its energy (0.88) sits close to your target of 0.90. Caveat: the mood is moody, not angry.

====================================================================

… (cut: DRIVEABLE CITY POP, HIGH-ENERGY JAZZ, CONFIDENT FUNK — 84 lines)

--------------------------------------------------------------------
  RUN REPORT
--------------------------------------------------------------------
  queries served        : 7
  recommendations made  : 35
  average confidence    : 0.88
  low-confidence picks  : 0/35
  claims made / dropped : 105 / 0
  grounding rate        : 1.00
  explanations withheld : 0
  full log              : logs/run.log
--------------------------------------------------------------------

[exit 0]
```

Note ranks 3–5 of the pop block: three genre-only picks whose mood misses, every one of them still labelled "Strong match". That is the confidence-threshold failure described in [What didn't work](#what-didnt-work), visible in the output rather than asserted.

### E3 — Pipeline narrating itself

```bash
python3 -m src.main --profile lofi -k 3 --log-level INFO
```

```
$ python3 -m src.main --profile lofi -k 3 --log-level INFO
[INFO] music_rec.recommender: Loading catalog from data/songs.csv
[INFO] music_rec.recommender: Loaded 203 song(s), skipped 0
[INFO] music_rec.retriever: Retriever ready: 203 songs from data/songs.csv across 19 genre(s)
[INFO] music_rec.retriever: Query: genre=lofi mood=chill target_energy=0.30 k=3
[INFO] music_rec.main: Run complete: 3 recommendation(s), avg confidence 0.975, grounding rate 1.000

Loaded 203 songs from data/songs.csv

====================================================================
  CHILL LOFI
  profile: genre=lofi · mood=chill · energy=0.3
  average confidence: 0.97
====================================================================

  1. Staying There  —  L'Indécis
     score 3.90 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.35) sits close to your target of 0.30.

  2. Soulful  —  L'Indécis
     score 3.84 · confidence 0.98 (high)
     Strong match (confidence 0.98): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.38) sits close to your target of 0.30.

  3. Aruarian Dance  —  Nujabes
     score 3.76 · confidence 0.96 (high)
     Strong match (confidence 0.96): recommended because it's lofi, your favorite genre, the mood is chill, exactly what you wanted and its energy (0.42) sits close to your target of 0.30.

====================================================================

--------------------------------------------------------------------
  RUN REPORT
--------------------------------------------------------------------
  queries served        : 1
  recommendations made  : 3
  average confidence    : 0.97
  low-confidence picks  : 0/3
  claims made / dropped : 9 / 0
  grounding rate        : 1.00
  explanations withheld : 0
  full log              : logs/run.log
--------------------------------------------------------------------

[exit 0]
```

### E4 — The DEBUG trail in `logs/run.log`

The console shows warnings only by default; the file always gets everything, including the per-rank scoring trace.

```bash
rm -f logs/run.log && python3 -m src.main --profile metal -k 2 > /dev/null && cat logs/run.log
```

```
2026-08-02 23:37:58,140 INFO     music_rec.recommender | Loading catalog from data/songs.csv
2026-08-02 23:37:58,143 INFO     music_rec.recommender | Loaded 203 song(s), skipped 0
2026-08-02 23:37:58,143 INFO     music_rec.retriever | Retriever ready: 203 songs from data/songs.csv across 19 genre(s)
2026-08-02 23:37:58,144 INFO     music_rec.retriever | Query: genre=metal mood=angry target_energy=0.90 k=2
2026-08-02 23:37:58,147 DEBUG    music_rec.retriever |   #1 Angel of Death (metal) score=3.88 confidence=0.98 band=high
2026-08-02 23:37:58,147 DEBUG    music_rec.retriever |   #2 Master of Puppets (metal) score=3.86 confidence=0.98 band=high
2026-08-02 23:37:58,147 INFO     music_rec.main | Run complete: 2 recommendation(s), avg confidence 0.980, grounding rate 1.000
```

Timestamps are the only non-reproducible values anywhere in this section.

### E5 — Reliability sweep across all seven profiles

Where the "15 of 35 genre-only, 0 flagged" figure comes from. The sweep runs the same `Retriever` and `ExplanationAgent` the CLI uses, and counts bands and signal hits per profile.

```bash
python3 - <<'PY'
from collections import Counter
from src.explainer import ExplanationAgent
from src.logging_setup import configure_logging
from src.main import PROFILES
from src.retriever import Retriever
configure_logging(console_level="ERROR")

retriever, agent = Retriever.from_csv("data/songs.csv"), ExplanationAgent()
print(f"{'profile':<10} {'avg':>5} {'high':>5} {'med':>4} {'low':>4} {'genre-only':>11}")
totals, genre_only_total = [], 0
for name, profile in PROFILES.items():
    results = retriever.retrieve(profile["prefs"], k=5)
    for r in results:
        agent.explain(profile["prefs"], r)
    bands = Counter(r.band for r in results)
    matched = [{s.name for s in r.breakdown.signals if s.matched} for r in results]
    genre_only = sum(1 for m in matched if m in ({"genre", "energy"}, {"genre"}))
    genre_only_total += genre_only
    totals.extend(r.confidence for r in results)
    print(f"{name:<10} {retriever.average_confidence(results):>5.2f} {bands['high']:>5} "
          f"{bands['medium']:>4} {bands['low']:>4} {genre_only:>11}")
print(f"{'ALL':<10} {sum(totals)/len(totals):>5.2f} {'':>5} {'':>4} "
      f"{sum(1 for c in totals if c < 0.45):>4} {genre_only_total:>11}")
print()
print("recommendations:", len(totals), "| claims:", agent.stats["claims_made"],
      "| dropped:", agent.stats["claims_dropped"], "| grounding:", agent.grounding_rate(),
      "| withheld:", agent.stats["withheld"])
PY
```

```
profile      avg  high  med  low  genre-only
pop         0.86     5    0    0           3
lofi        0.90     5    0    0           2
rock        0.82     5    0    0           3
metal       0.90     5    0    0           2
city pop    0.90     5    0    0           2
jazz        0.95     5    0    0           1
funk        0.85     5    0    0           2
ALL         0.88               0          15

recommendations: 35 | claims: 105 | dropped: 0 | grounding: 1.0 | withheld: 0
[exit 0]
```

35 of 35 land in the `high` band. 15 of those 35 matched genre and energy but **not** the requested mood, and none was flagged — the detector's recall on its own failure mode is zero. `rock` (0.82) is the weakest profile, `jazz` (0.95) the strongest.

### E6 — Uncovered genre (`polka`)

`polka` has zero songs in the catalog. The retriever says so; the confidence layer's reaction depends on whether the *mood* is covered too.

```bash
python3 - <<'PY'
from src.logging_setup import configure_logging
from src.retriever import Retriever
configure_logging(console_level="WARNING")

prefs = {"favorite_genre": "polka", "favorite_mood": "cheerful", "target_energy": 0.6}
retriever = Retriever.from_csv("data/songs.csv")
print("input:", prefs)
for rank, record in enumerate(retriever.retrieve(prefs, k=5), 1):
    print(f"  {rank}. {record.song['title']} ({record.song['genre']}) "
          f"score {record.score:.2f} confidence {record.confidence:.2f} ({record.band})")
PY
```

```
[WARNING] music_rec.retriever: No song in data/songs.csv has genre 'polka' - results will lean on energy alone
[WARNING] music_rec.retriever: Every result for genre=polka mood=cheerful is low confidence - the catalog does not really serve this profile
input: {'favorite_genre': 'polka', 'favorite_mood': 'cheerful', 'target_energy': 0.6}
  1. Four (jazz) score 2.00 confidence 0.43 (low)
  2. Yes Sir, That's My Baby (jazz) score 2.00 confidence 0.43 (low)
  3. Midnight Pretenders (city pop) score 2.00 confidence 0.43 (low)
  4. Wait for the Moment (funk) score 2.00 confidence 0.43 (low)
  5. Cigarette Daydreams (indie rock) score 2.00 confidence 0.43 (low)
[exit 0]
```

Both guardrail warnings fire and all five picks are correctly banded `low`. But swap the mood for one the catalog *does* contain, or just move the requested energy, and the bands move with it — same catalog, same weights, same missing genre:

```bash
python3 - <<'PY'
from collections import Counter
from src.logging_setup import configure_logging
from src.retriever import Retriever
configure_logging(console_level="CRITICAL")

retriever = Retriever.from_csv("data/songs.csv")
probes = [
    ("polka", "cheerful", 0.6), ("polka", "happy", 0.6), ("polka", "energetic", 0.7),
    ("ska", "energetic", 0.3), ("ska", "energetic", 0.5), ("ska", "energetic", 0.7),
    ("classical", "relaxed", 0.3), ("classical", "relaxed", 0.8),
]
for genre, mood, energy in probes:
    results = retriever.retrieve(
        {"favorite_genre": genre, "favorite_mood": mood, "target_energy": energy}, k=5)
    bands = Counter(r.band for r in results)
    print(f"{genre:<9} {mood:<9} energy={energy}  avg={retriever.average_confidence(results):.2f}  "
          f"{dict(bands)}")
PY
```

```
polka     cheerful  energy=0.6  avg=0.43  {'low': 5}
polka     happy     energy=0.6  avg=0.63  {'medium': 5}
polka     energetic energy=0.7  avg=0.64  {'medium': 5}
ska       energetic energy=0.3  avg=0.60  {'medium': 5}
ska       energetic energy=0.5  avg=0.69  {'high': 2, 'medium': 3}
ska       energetic energy=0.7  avg=0.85  {'high': 5}
classical relaxed   energy=0.3  avg=0.83  {'high': 5}
classical relaxed   energy=0.8  avg=0.60  {'medium': 5}
[exit 0]
```

Two readings of the same run. First, a single mood match plus a decent energy fit is enough to lift a *completely off-genre* result out of the `low` band: `polka` is unavailable at any price, and the system says `medium`. The retriever knows the genre is missing and logs it; the confidence score never sees that fact. Second, `ska` and `classical` move across two bands on `target_energy` alone — the songs, the weights and the genre coverage are identical in every row. That is the "confidence is relative to the catalog, not to the user" limitation, measured.

### E7 — Guardrail results

| # | Guardrail | Trigger | Observed | Exit |
|---|---|---|---|---|
| G1 | Malformed catalog rows | `--catalog data/broken.csv` | 3 rows skipped with line + reason, run continues on the 3 valid ones | 0 |
| G2 | Missing catalog | `--catalog data/nope.csv` | one-line error, no traceback | 1 |
| G3 | No valid rows | catalog where every row is corrupt | rows skipped, then a fatal `contained no valid songs` | 1 |
| G4 | Invalid `k` | `-k 0` | rejected before any work is done | 1 |
| G5 | Invalid profile | missing key / non-numeric energy / empty genre | `ProfileError` naming the offending field | n/a |
| G6 | Out-of-range energy | `target_energy=5.0` | clamped to 1.0, warning logged, run continues | n/a |
| G7 | Ungrounded claim | evidence desynchronised from the song | claim dropped; all claims corrupt ⇒ explanation withheld | n/a |

**G1 — malformed rows are skipped, not fatal.** `data/broken.csv` holds 6 rows: an out-of-range energy, a row missing artist and mood, an unparseable energy value, and 3 good rows.

```bash
python3 -m src.main --catalog data/broken.csv --profile pop -k 3
```

```
$ python3 -m src.main --catalog data/broken.csv --profile pop -k 3
[WARNING] music_rec.guardrails: data/broken.csv: skipped 3 malformed row(s); first: line 5: energy=9.9 is outside the 0.0-1.0 range

Loaded 3 songs from data/broken.csv

====================================================================
  HIGH-ENERGY POP
  profile: genre=pop · mood=happy · energy=0.8
  average confidence: 0.53
====================================================================

  1. Can't Stop the Feeling!  —  Justin Timberlake
     score 3.96 · confidence 0.99 (high)
     Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.

  2. Mr. Brightside  —  The Killers
     score 1.78 · confidence 0.40 (low)
     Weak match (confidence 0.40): recommended because its energy (0.91) sits close to your target of 0.80. Caveat: it's rock, not the pop you asked for; the mood is intense, not happy. Treat this as a stretch pick rather than a real match.

  3. Aruarian Dance  —  Nujabes
     score 1.24 · confidence 0.19 (low)
     Weak match (confidence 0.19): nothing in your profile lines up with this song. Caveat: it's lofi, not the pop you asked for; the mood is chill, not happy; its energy (0.42) is off your target of 0.80. Treat this as a stretch pick rather than a real match.

====================================================================

--------------------------------------------------------------------
  RUN REPORT
--------------------------------------------------------------------
  queries served        : 1
  recommendations made  : 3
  average confidence    : 0.53
  low-confidence picks  : 2/3
  claims made / dropped : 9 / 0
  grounding rate        : 1.00
  explanations withheld : 0
  full log              : logs/run.log
--------------------------------------------------------------------

[exit 0]
```

Two things at once: the catalog guardrail degrades instead of collapsing, **and** the confidence layer works exactly as designed when the catalog is genuinely thin — 2 of 3 picks flagged `low`, both worded as stretch picks with the mismatch spelled out. The band is not broken; the 203-song catalog simply never puts it under strain.

**G2 — missing catalog.**

```bash
python3 -m src.main --catalog data/nope.csv
```

```
$ python3 -m src.main --catalog data/nope.csv
[ERROR] music_rec.main: Catalog load failed: Catalog not found at data/nope.csv - run from the repo root
Error: Catalog not found at data/nope.csv - run from the repo root
[exit 1]
```

**G3 — a catalog where nothing survives validation.** Build a two-row catalog, both rows broken (row 1 has no artist, row 2 has `energy=7.7`):

```bash
printf 'id,title,artist,genre,mood,energy,tempo_bpm,valence,danceability,acousticness\n1,Ghost Track,,pop,happy,0.5,120,0.5,0.5,0.5\n2,Bad Energy,Someone,pop,happy,7.7,120,0.5,0.5,0.5\n' > data/all_invalid.csv && python3 -m src.main --catalog data/all_invalid.csv; echo "[exit $?]"; rm data/all_invalid.csv
```

```
[WARNING] music_rec.guardrails: data/all_invalid.csv: skipped 2 malformed row(s); first: line 2: missing value(s) for artist
[ERROR] music_rec.main: Catalog load failed: data/all_invalid.csv contained no valid songs (2 row(s) rejected)
Error: data/all_invalid.csv contained no valid songs (2 row(s) rejected)
[exit 1]
```

**G4 — invalid `k`.**

```bash
python3 -m src.main -k 0
```

```
$ python3 -m src.main -k 0
Error: -k must be a positive integer.
[exit 1]
```

**G5 / G6 — profile validation and energy clamping.** Four inputs through `validate_user_prefs()`: a missing key, a non-numeric energy, an empty genre, and a valid-but-out-of-range profile.

```bash
python3 - <<'PY'
from src.guardrails import validate_user_prefs, ProfileError
from src.logging_setup import configure_logging
configure_logging(console_level="WARNING")

cases = [
    {"favorite_mood": "happy", "target_energy": 0.8},
    {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": "loud"},
    {"favorite_genre": "", "favorite_mood": "happy", "target_energy": 0.8},
    {"favorite_genre": "POP ", "favorite_mood": "Happy", "target_energy": 5.0},
]
for prefs in cases:
    print("input :", prefs)
    try:
        print("output:", validate_user_prefs(prefs))
    except ProfileError as exc:
        print("output: ProfileError:", exc)
    print()
PY
```

```
[WARNING] music_rec.guardrails: target_energy=5.0 is outside 0.0-1.0; clamping to 1.0
input : {'favorite_mood': 'happy', 'target_energy': 0.8}
output: ProfileError: profile is missing required key(s): favorite_genre

input : {'favorite_genre': 'pop', 'favorite_mood': 'happy', 'target_energy': 'loud'}
output: ProfileError: profile.target_energy must be numeric, got 'loud'

input : {'favorite_genre': '', 'favorite_mood': 'happy', 'target_energy': 0.8}
output: ProfileError: profile.favorite_genre must be a non-empty string, got ''

input : {'favorite_genre': 'POP ', 'favorite_mood': 'Happy', 'target_energy': 5.0}
output: {'favorite_genre': 'pop', 'favorite_mood': 'happy', 'target_energy': 1.0}

[exit 0]
```

Three fatal cases name the exact offending field; the fourth is recoverable, so `target_energy` is clamped to 1.0 and the case of the strings normalised — with the clamp logged rather than applied silently.

**G7 — fault injection: ungrounded claims.** This is the grounding guardrail under deliberate attack. A real retrieved record is fetched, then its evidence is desynchronised from the song it describes — first one signal, then all of them.

```bash
python3 - <<'PY'
from src.explainer import ExplanationAgent
from src.logging_setup import configure_logging
from src.retriever import Retriever
configure_logging(console_level="ERROR")

prefs = {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}
record = Retriever.from_csv("data/songs.csv").retrieve(prefs, k=1)[0]
agent = ExplanationAgent()

print("song   :", record.song["title"], "| genre:", record.song["genre"], "| mood:", record.song["mood"])
print("normal :", agent.explain(prefs, record))
print("stats  :", agent.stats)
print()

# Fault injection 1: desynchronise one signal from the song it describes.
record.breakdown.signals[0].song_value = "polka"
print("after mutating the genre signal to 'polka' (song is still pop):")
print("output :", agent.explain(prefs, record))
print("stats  :", agent.stats)
print()

# Fault injection 2: desynchronise every signal.
for signal in record.breakdown.signals:
    signal.song_value = "nonsense" if isinstance(signal.song_value, str) else -1.0
print("after corrupting every signal:")
print("output :", agent.explain(prefs, record))
print("stats  :", agent.stats)
print("grounding rate:", agent.grounding_rate())
PY
```

```
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': genre='polka' not supported by the record
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': genre='nonsense' not supported by the record
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': mood='nonsense' not supported by the record
[ERROR] music_rec.explainer: Dropped ungrounded claim on 'Can't Stop the Feeling!': energy=-1.0 not supported by the record
[ERROR] music_rec.explainer: Withholding explanation for 'Can't Stop the Feeling!': no claim survived grounding
song   : Can't Stop the Feeling! | genre: pop | mood: happy
normal : Strong match (confidence 0.99): recommended because it's pop, your favorite genre, the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.
stats  : {'explanations': 1, 'claims_made': 3, 'claims_dropped': 0, 'withheld': 0}

after mutating the genre signal to 'polka' (song is still pop):
output : Strong match (confidence 0.99): recommended because the mood is happy, exactly what you wanted and its energy (0.82) sits close to your target of 0.80.
stats  : {'explanations': 2, 'claims_made': 6, 'claims_dropped': 1, 'withheld': 0}

after corrupting every signal:
output : No verifiable evidence for this pick, so no explanation is offered.
stats  : {'explanations': 3, 'claims_made': 9, 'claims_dropped': 4, 'withheld': 1}
grounding rate: 0.556
[exit 0]
```

The system never says "polka". The unsupported claim is dropped and the sentence is rebuilt from what remains; when nothing survives, the explanation is withheld rather than invented, and the run report's grounding rate falls from 1.00 to 0.556 so the damage is visible in the metrics.

### E8 — Exit codes

| Command | Exit |
|---|---|
| `python3 -m src.main -k 5` | 0 |
| `python3 -m src.main --catalog data/broken.csv --profile pop -k 3` | 0 |
| `python3 -m src.main --catalog data/nope.csv` | 1 |
| `python3 -m src.main --catalog data/all_invalid.csv` | 1 |
| `python3 -m src.main -k 0` | 1 |
| `python3 -m pytest -q` | 0 |

No failure path produces a traceback; every one prints a single line naming what went wrong.

---

## Design Decisions

**Content-based, not collaborative.** There is one user and no interaction history, so there is nothing to collaboratively filter. Content-based scoring is also fully inspectable which is a requirement for the explanation feature.

**Categorical vs. continuous fields are scored differently.** `genre` and `mood` can only match or miss, so they earn a flat bonus. `energy` lives on a 0–1 scale where "close" is meaningfully different from "far," so it earns partial credit via linear closeness: `1 - abs(song.energy - target_energy)`. Because energy is already normalized to 0–1, this needs no scaling and is guaranteed bounded, so it composes cleanly with the flat bonuses.

**The weights, and the ones I changed.** The Phase 2 design (see [docs/algorithm_recipe.md](docs/algorithm_recipe.md)) was *genre-first*: genre +2.0, mood +1.0, energy ×1.0. I first shipped an *energy-first* variant — genre +1.0, mood +1.0, energy ×2.0 — deliberately, as an experiment, and kept it because what it revealed about the system was more interesting than the safer configuration. That experiment flattened genre and mood into a single interchangeable tie-breaker, which is wrong: a listener who asks for rock would rather hear a *confident* rock track than an *intense* funk one. The current weights restore the ranking between the two categoricals while keeping energy's granularity: **genre +1.5, mood +0.5, energy ×2.0**. A perfect song caps at 4.0 under all three configurations, so the pinned scoring tests still assert exact numbers.

**Trade-offs I accepted:**

| Decision | Gained | Gave up |
|---|---|---|
| Energy weighted ×2.0 | Smooth, granular ranking; no large blocks of tied scores | Genre/mood demoted to tie-breakers; off-genre songs surface as soon as a genre has fewer songs than `k` |
| Genre +1.5 vs. mood +0.5 | Right-genre/wrong-mood (3.5) always beats wrong-genre/right-mood (2.5) | A mood match alone is now nearly invisible next to energy; mood barely moves the order |
| Score only 3 of 10 fields in v1 | Readable math, explanations a non-programmer can follow | `tempo_bpm`, `valence`, `danceability`, `acousticness` are parsed but unused |
| Exact string match on genre | Trivially simple and predictable | `indie pop` ≠ `pop`; no notion of related genres |
| Reasons built during scoring, not after | Explanations can't drift from the score that produced them | Reason wording is coupled to the scoring function |
| Dict-based functional pipeline for the CLI | Direct CSV→dict flow, easy to test | Duplicates the `Song`/`UserProfile`/`Recommender` dataclass API, which remains as scaffolding |

**Why `tempo_bpm` is excluded.** It ranges 60–168, not 0–1. Dropping it into the same closeness formula without min-max scaling would let it swamp every other signal. Excluding it was cheaper than scaling it correctly for v1.

---

## Testing Summary

**The short version:** 63 of 63 automated tests pass. Across the seven demo profiles the system made 35 recommendations with an average confidence of **0.88**, flagging **0 of 35** as low-confidence. All **105 generated claims were grounded** in retrieved data (grounding rate 1.00, 0 explanations withheld). Fault injection confirmed the guardrails: a catalog with 3 corrupt rows loaded the 3 good ones and skipped the rest, and an artificially desynchronised record had its unsupported claim dropped rather than printed. The biggest problem is still the confidence layer: **15 of those 35 picks were genre-only matches that missed the requested mood, and not one was flagged** — and it takes a genre *and* a mood the catalog does not contain before the low band fires at all.

Every figure in this section is reproduced by a command in [Reproducible Execution Evidence](#reproducible-execution-evidence): the test count in [E1](#e1--test-suite), the 35/0.88/105/1.00 run figures in [E2](#e2--full-demo-run-all-seven-profiles) and [E5](#e5--reliability-sweep-across-all-seven-profiles), the band behaviour in [E6](#e6--uncovered-genre-polka), and the fault injections in [E7](#e7--guardrail-results).

### How it is tested

| Layer | File | Tests | What it pins down |
|---|---|---|---|
| Scoring & ranking | [tests/test_recommender.py](tests/test_recommender.py) | 10 | The weight invariants (genre > mood; weights sum to `MAX_SCORE`), exact score values, descending order, `k`, the empty-reasons fallback, CSV typing |
| Guardrails | [tests/test_guardrails.py](tests/test_guardrails.py) | 14 | Bad rows rejected, bad profiles rejected, energy clamping, grounding check |
| Retrieval & confidence | [tests/test_retriever.py](tests/test_retriever.py) | 16 | Catalog failures, ranking, confidence bounds/ordering/bands |
| Explanation grounding | [tests/test_explainer.py](tests/test_explainer.py) | 9 | Wording tracks evidence, ungrounded claims dropped, explanations withheld, no genre leaks across every demo profile |
| End-to-end CLI | [tests/test_main.py](tests/test_main.py) | 14 | Full pipeline runs, every demo profile runs alone and is covered by the catalog, reliability report prints, clean failure + exit code 1 |

Three independent reliability mechanisms, not one:

1. **Automated tests** — including two *fault-injection* tests that deliberately break the system to prove the guardrail fires: a catalog whose rows are corrupt, and a retrieved record whose evidence has been desynchronised from the song it describes.
2. **Confidence scoring** — every recommendation is rated 0–1 from two ingredients (share of points earned, and how many distinct signals matched), then banded high/medium/low. The band controls the *wording*: only a high-confidence pick is allowed to say "Strong match", and a low-confidence pick must carry a "stretch pick" warning.
3. **Logging and error handling** — every stage writes to `logs/run.log`; malformed rows are skipped with the line number and reason; unrecoverable problems (missing catalog, no valid rows, invalid profile) exit with a one-line message and code 1, never a traceback.
4. **Human evaluation** — reading the output of contrasting runs and judging it by eye, which is the only mechanism that catches "plausible but wrong."

```json
{
  "human_evaluation": [
    {
      "test_input": "-k 3",
      "evaluation_criteria": "Songs match by genre then mood",
      "result": "Pass"
    },
    {
      "test_input": "Empty input",
      "evaluation_criteria": "Handles gracefully",
      "result": "Pass"
    }
  ]
}
```

| Test input | Criteria | Result | Evidence |
|---|---|---|---|
| `-k 3` | Songs match by genre first, then mood | Pass | [Sample Interactions](#sample-interactions) — every rank matches the requested genre; mood is the signal that slips |
| Empty / unusable input | Handles gracefully, no traceback | Pass | [E7 G2–G5](#e7--guardrail-results) — missing catalog, all-invalid catalog, `-k 0`, malformed profiles |

### What worked

- **Scoring math is exact and pinned.** A perfect match returns precisely 4.0; a genre/mood miss at energy distance 0.4 returns precisely 1.2. Any weight change breaks these loudly, which is the point.
- **The explanation cannot outrun the evidence.** Claims are built from the same `SignalMatch` objects that produced the score and checked against the retrieved record before printing. Injecting a mismatch drops the claim; corrupting every signal makes the agent withhold the explanation instead of inventing one.
- **The caveats are always right, even when the confidence number isn't.** Whatever the band says, the agent names every mismatch it found: `city pop`, `indie pop`, `funk` and `jazz` results all state what they actually are instead of passing as the requested genre. Swept across all seven profiles and the full 203-song catalog — 1,421 explanations — not one names a genre the song does not have, and no claim was dropped. The grounded-claim mechanism held up under both catalog changes; the threshold policy did not.
- **Genre now reliably outranks mood in the output, not just in the constants.** The pop profile's rank 3 is a `pop`/`confident` track at 3.40, ahead of every `indie pop`/`happy` candidate — the ordering the 1.5/0.5 split was meant to produce. Two tests pin the rule itself (`W_GENRE > W_MOOD`, and the weights summing to `MAX_SCORE`) so a future retune fails as a violated rule rather than as an unexplained arithmetic mismatch.
- **Guardrails degrade instead of collapsing.** 3 corrupt rows out of 6 → the run continued on the 3 valid songs and logged the reason for each skip.
- **Two problems from the last phase are fixed.** The `Song`/`UserProfile`/`Recommender` dataclasses no longer duplicate the pipeline — they delegate to it. And `python3 -m src.main` now works (both invocations do).

### What didn't work

- **Growing the catalog broke the confidence signal, and the second expansion finished the job.** This is still the biggest finding. At 18 songs the demo profiles produced 6 low-confidence picks and averaged 0.66; at 53 songs, **zero** and 0.76; at 203 songs, zero and **0.88**. Nothing improved — a deeper catalog just guarantees that *some* song has the right genre and close energy, which now clears the `high` threshold outright. The scoreboard across 35 picks: **20 full genre+mood matches, 15 genre-only picks, 0 flagged.** A detector that flags nothing has perfect recall and no value. The thresholds (0.75 / 0.45) were hand-picked against an 18-song catalog and have not survived either expansion.
- **The low band now takes a double miss to reach.** Worse than the last phase, where it still fired on thin genres. With ten songs in every genre but jazz, `metal`/`angry` averages 0.90 with 5 of 5 high, and `classical`, `country`, `ska` and `r&b` land 5 of 5 high whenever the requested energy sits inside the range that genre actually spans. The decisive case: `polka` — **zero** songs in the catalog — returns 5 of 5 *medium* at 0.63 when the requested mood exists. The retriever logs "no song has genre 'polka'" and the confidence layer still declines to call any of it a weak match; coverage is measured by the retriever and ignored by the score. Only when the *mood* misses as well does the band drop to 5 of 5 low at 0.43 ([E6](#e6--uncovered-genre-polka)). So the mechanism is not dead — it is just calibrated so that one matching categorical, or a lucky energy target, is enough to suppress it. The same run also shows the band working properly on a genuinely thin catalog: against `data/broken.csv` it flags 2 of 3 picks low and words them as stretch picks ([E7 G1](#e7--guardrail-results)).
- **Confidence is relative to the catalog, not to the user.** Because both ingredients are catalog-relative, the same song can score 0.43 in a small catalog and 0.70 in a large one without changing at all. Any threshold expressed as an absolute number inherits that instability. A rank-relative measure (how much better is this than the median candidate?) would not.
- **Confidence cannot audit the policy it comes from.** It is computed from the same weights it reports on, so it measures *agreement with the scoring rule*, not correctness. If the energy-first weighting is the wrong policy — and the previous phase argued it is — a confidently wrong recommendation still scores high. Self-reported confidence is a consistency check, not a truth check.
- **Energy still dominates the ranking.** With `W_ENERGY = 2.0` energy contributes to every song and is always positive, so genre and mood act as tie-breakers on an energy-driven order. Splitting the categorical weight 1.5/0.5 in favor of genre made the tie-break break the *right way* — an off-genre song can no longer draw level with an on-genre one on a mood match — but energy's 2.0 share is untouched, so the underlying dominance remains. The confidence layer makes it visible rather than fixing it: rock is still the weakest profile (0.82 against a 0.88 mean), which is the one the catalog serves worst. The cost lands on mood, not genre: at 0.5 a mood match moves a score less than an energy difference of 0.25 does, which is why 15 of 35 picks miss the requested mood.
- **Exact string matching is unchanged by the new data.** `city pop` and `indie pop` both score zero for a `pop` listener, and `indie rock` scores zero for `rock` — three near-misses the catalog now contains and the matcher still cannot see.

**What I learned:** the useful question is not "did the tests pass" but "what would the system say when it shouldn't be sure?" All 7 tests passed in the previous phase while the system confidently recommended pop to a rock listener. What changed that wasn't more tests of the same kind — it was making the system *rate its own evidence* and forcing the wording to obey the rating, then testing that mechanism by breaking it on purpose.

Then adding songs silently disabled it — twice. Every test still passed, the average confidence went *up* both times (0.66 → 0.76 → 0.88), and the run report looked healthier at every step, while the system's ability to tell a real match from a coincidence went to zero. That is the same lesson as the energy-weight finding, one level up: the metric moved in the flattering direction for a reason that had nothing to do with quality. A number that only ever gets better is not measuring anything.

The profile expansion taught a narrower version of the same thing. Four new profiles went in and the whole suite stayed green — because the end-to-end test asserted on three profile names it already knew about. A test that names its expectations instead of deriving them will keep passing over exactly the code you just added, so the CLI tests now build their assertions from `PROFILES` itself.

---

## Limitations and Risks

- **Genre coverage is no longer the binding constraint (203 songs)** — this limitation is largely resolved. The catalog now spans 19 genres with 10 songs each, plus 23 jazz; there are no singleton genres left, and every demo profile has both its genre and its mood represented. What it exposed is that coverage was never the real problem: with the thin-catalog excuse removed, the confidence layer stopped flagging anything across the seven demo profiles, and it will not flag a genre the catalog lacks entirely unless the requested mood is missing too. Mood coverage is thinner than genre coverage — 13 moods across 203 songs — so a mood miss remains the common failure.
- **No audio understanding** — it compares human-assigned labels, never the music itself. Every bias in the labeling is inherited wholesale.
- **Energy over-weighting** — the dominant signal by construction; documented above and in the model card.
- **No related-genre knowledge** — exact string equality only.
- **No diversity control** — nothing prevents the top-5 from being the same artist repeatedly.
- **Single-preference profiles** — a listener with genuinely mixed taste cannot be expressed.
- **Self-reported confidence** — the confidence score is derived from the same weights it rates, so it measures internal consistency, not correctness. It cannot flag a recommendation that the scoring policy itself gets wrong.
- **Template-based explanations** — deterministic and fully auditable, but the phrasing is fixed. It explains *why the score came out that way*, which is not the same as musical insight.

Full analysis, including responsible-AI reflection and where bias enters: [model_card.md](model_card.md).

---

## Project Structure

```
├── src/
│   ├── main.py            # CLI runner: argparse, demo profiles, run report
│   ├── retriever.py       # Retriever: knowledge base + top-k retrieval with evidence
│   ├── explainer.py       # ExplanationAgent: grounded generation from retrieved records
│   ├── recommender.py     # scoring/ranking core + load_songs + OOP facade
│   ├── guardrails.py      # row/profile validation, grounding checks, error types
│   ├── models.py          # Song, UserProfile, SignalMatch, RetrievedSong, confidence
│   └── logging_setup.py   # console + logs/run.log configuration
├── data/songs.csv         # 203-song catalog, 19 genres, 10 fields per song
├── tests/
│   ├── test_recommender.py    # scoring + ranking math
│   ├── test_guardrails.py     # validation and grounding
│   ├── test_retriever.py      # catalog failures, ranking, confidence
│   ├── test_explainer.py      # grounded generation, dropped claims
│   └── test_main.py           # end-to-end CLI
├── docs/algorithm_recipe.md   # Phase 2 scoring design (code-free, superseded)
├── diagrams/
│   ├── uml.mmd            # class diagram
│   └── architecture.mmd   # system flow diagram
├── conftest.py            # puts the repo root on sys.path for pytest
└── model_card.md          # intended use, data, limitations, reflection
```

Data flows one way: `main -> retriever -> (recommender core) -> explainer -> output`, with `guardrails` and `logging_setup` used at every stage. `models.py` imports no project code, so the dependency graph stays acyclic.

---

## Reflection

Building this taught me that a recommender doesn't understand anything. It compares labels and numbers and adds them up, and everything a user experiences as "taste" or "insight" is really a set of weights somebody picked. The system's opinion about music is entirely an artifact of the three constants at the top of `score_song()`.

The sharpest lesson was how invisible that is from the inside. I raised the energy weight and the system kept producing recommendations that read as perfectly reasonable — plausible songs, confident explanations, no errors, all tests green. Only when I laid several different user profiles next to each other did the pattern show: the same reason text on almost every result, genre demoted to a tiebreaker. Nothing *failed*; the system just quietly became a different product than the one I designed. Splitting the categorical weight 1.5/0.5 fixed the ordering between genre and mood, but it took writing the rule down as a test — `W_GENRE > W_MOOD` — before the diagrams stopped advertising three different weightings the code had never used. That reframed problem-solving for me — evaluating an AI system isn't checking whether the output looks right, it's checking whether the output is right *for reasons you can name*. Now when a streaming app recommends something, my first thought is about whoever decided which factors count and how much.

The graded responsible-AI reflection — how I collaborated with AI tools, one helpful and one flawed AI suggestion, and the system's limitations — is in [model_card.md](model_card.md).
