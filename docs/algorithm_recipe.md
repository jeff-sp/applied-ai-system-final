> **Superseded — kept as the Phase 2 design record.** The weights below (genre
> +2.0, mood +1.0, energy ×1.0) were the original genre-first design and were
> never shipped. The implementation uses **genre +1.5, mood +0.5, energy ×2.0**
> (`src/recommender.py:44-46`), so the worked examples at the end of this
> document do not reproduce under the current code. See
> [README.md](../README.md) § *The weights, and the ones I changed* for how and
> why they moved.
>
> The **genre-first stance itself still holds**: a genre match is worth more
> than a mood match — 3× more, in fact, up from 2× here. What changed is that
> energy was promoted from a fine-tuner (×1.0) to the largest single share
> (×2.0), so genre and mood now break ties on an energy-driven ranking rather
> than deciding the order outright.
>
> **Superseded a second time, in scope rather than in values.** This document
> describes scoring as the *retrieval* mechanism — the thing that decides which
> songs come back. It is now the **reranking** mechanism. Embeddings retrieve a
> 20-song shortlist from a vector index, and the weights below reorder that
> shortlist; they no longer see the whole catalog. The arithmetic is unchanged
> and `src/recommender.py` was not modified, but "score every song, sort, cut to
> top-k" now reads "score every *shortlisted* song, sort, cut to top-k".
>
> Why the split: an embedding understands *"something mellow for studying"* but
> cannot honour a numeric `target_energy`, and a weighted scorer can do the
> reverse. The retrieval layer is documented in
> [retrieval_recipe.md](retrieval_recipe.md).

# Phase 2 — Algorithm Recipe: Scoring Logic

This document is the **design** for how `score_song()` ranks songs against a
user's preferences. It is intentionally code-free — it is the recipe the
implementation should follow.

## Design stance: genre-first

This recommender is designed as a **genre-first** tool. Genre is the coarsest,
most identity-defining filter of taste: a listener who asks for `rock` and is
handed `classical` experiences a hard miss, no matter how well the mood lines
up. The primary job is to get the *category* right; mood is a secondary
refinement that fine-tunes within the desired genre.

Consequence: **genre is weighted above mood.**

## The two kinds of fields

| Field type   | Fields                                              | Nature                        | Scoring approach            |
|--------------|-----------------------------------------------------|-------------------------------|-----------------------------|
| Categorical  | `genre`, `mood`                                     | Exact match or no match       | Flat bonus (all-or-nothing) |
| Continuous   | `energy`, `valence`, `danceability`, `acousticness` | 0.0–1.0 scale, closeness matters | Distance-based (partial credit) |

Categorical fields can only match or miss, so they earn a fixed bonus.
Continuous fields live on a 0–1 scale, so "close" should earn partial credit.

## Point weighting

| Signal            | Weight | Max contribution | Rationale                                              |
|-------------------|--------|------------------|--------------------------------------------------------|
| **Genre match**   | +2.0   | 2.0              | Primary signal — the category that defines taste       |
| **Mood match**    | +1.0   | 1.0              | Secondary refinement within the desired genre          |
| **Energy closeness** | ×1.0 | 1.0             | Meaningful, but should not outvote the genre match     |

A "perfect" song therefore caps at **4.0 points**, with genre as the dominant lever.

### Why these numbers

- Genre at 2× mood encodes the genre-first stance directly in the math: getting
  the category right is worth twice getting the mood right.
- Energy is capped at 1.0 so that even a flawless energy match cannot, on its
  own, outrank a genre match. Energy fine-tunes; it does not decide.

## Similarity math for continuous fields

Because energy is already on a 0–1 scale, use **linear closeness** rather than
raw distance:

```
energy_points = W_energy * (1 - abs(song.energy - user.target_energy))
```

- Identical energy → `1 - 0 = 1.0` → full points
- Opposite extremes (0.0 vs 1.0) → `1 - 1 = 0.0` → nothing
- Always bounded to [0, 1], so it composes cleanly with the flat bonuses.

## Scope decisions for v1

1. **Energy only, for now.** `UserProfile` also carries targets for `valence`,
   `danceability`, and `acousticness`. v1 scores **energy only** to keep the
   logic simple and the explanations readable. Once scoring + explanations work,
   the same closeness formula can be extended to the other three continuous
   fields (suggested weight ×0.5 each) for a richer signal.

2. **Collect reasons while scoring.** `score_song()` returns `(score, reasons)`.
   Build the `reasons` list *in the same pass* that computes the score, so
   `explain_recommendation()` can reuse it — e.g. "Matched your favorite genre
   (lofi); mood also matched (chill); energy was a close match." Do not score and
   explain in two separate passes.

## Worked example

User wants: genre `lofi`, mood `chill`, target energy `0.40`.

Song #2 "Midnight Coding" (lofi, chill, energy 0.42):

| Signal  | Match?              | Points                        |
|---------|---------------------|-------------------------------|
| Genre   | lofi == lofi ✓      | +2.0                          |
| Mood    | chill == chill ✓    | +1.0                          |
| Energy  | 1 - |0.42 - 0.40|   | +0.98                         |
| **Total** |                   | **3.98** (near-perfect)       |

Song #6 "Spacewalk Thoughts" (ambient, chill, energy 0.28):

| Signal  | Match?              | Points                        |
|---------|---------------------|-------------------------------|
| Genre   | ambient != lofi ✗   | +0.0                          |
| Mood    | chill == chill ✓    | +1.0                          |
| Energy  | 1 - |0.28 - 0.40|   | +0.88                         |
| **Total** |                   | **1.88**                      |

The genre-first stance is visible here: the ambient track scores much lower
because it misses the genre, even though it nails the mood and energy.
