# Retrieval Recipe: Chunking, Embedding, and Reranking

The companion to [algorithm_recipe.md](algorithm_recipe.md). That document
describes how a song is **scored**; this one describes how it is **found** in the
first place, and why the two are separate stages.

Intentionally code-free. It is the reasoning the implementation follows, not a
walkthrough of it.

---

## Why two stages

An embedding and a weighted scorer fail in opposite directions.

An embedding understands *"something mellow for a rainy Sunday"*. It has no idea
that `target_energy` is 0.3, and it cannot be told — a request for energy below
0.4 is not a semantic concept, it is arithmetic. Ask an embedding to honour it and
you get songs that talk about being calm rather than songs that are calm.

The weighted scorer has the opposite problem. It handles `energy: 0.3` exactly and
cannot parse "rainy Sunday" at all. Given only structured preferences it is
precise; given a sentence it is useless.

So: **embeddings retrieve, the scorer reranks.** Retrieval reduces 203 songs to a
shortlist of 20 using the parts of the request that are semantic. Reranking orders
those 20 using the parts that are arithmetic. Each stage does what it is good at.

The measurement that justifies this: recall@20 for the requested genre is **19/19
across all catalog genres** and **7/7 on the demo queries**, even with the crude
offline embedder. The shortlist reliably contains what the reranker needs, which
is the only property the design depends on. Top-1 retrieval accuracy is much
weaker — and irrelevant, because the reranker fixes the ordering.

---

## What gets embedded

Two kinds of chunk share one index.

**Song cards.** Each catalog row is rendered as two sentences of natural language.
This is not decoration: an embedding matches *"high energy"* far more reliably than
it matches `0.82`. But the raw number stays in the text alongside the word, because
the grounding check verifies generated claims against those exact digits. Drop the
number and a truthful statement about energy becomes uncitable.

Cards are never split. One card is roughly 50 tokens.

**Prose.** Around 7,200 words describing all 19 genres, all 13 moods, seven eras,
and ten listening contexts. This is what bridges everyday phrasing to the catalog's
categorical labels — the path from *"something for the gym"* to `energetic` to
`edm / metal / funk` runs through this text, not through the song cards.

The listening-contexts file earns its place more than any other. Most real requests
name a *situation*, not a genre.

---

## Chunking policy

**Target 900 characters, 150 of overlap.**

900 characters is about 225 tokens — far below the 8,192-token per-item input
limit, and small enough that injecting three passages into a prompt costs roughly
700 tokens. Larger chunks retrieve more coarsely: a 2,000-character chunk covering
three genres matches every query about any of them and answers none precisely.

150 characters of overlap is about one sentence. The purpose is that a thought
spanning a chunk boundary is retrievable from either side. Overlap is snapped
forward to a word boundary, because an overlap starting mid-word hands the
embedder a token that means nothing.

**Boundaries follow structure before size.** Splitting on `##` headings first means
each chunk is about one subject, and the heading becomes the chunk's title — which
then feeds the document prefix, so the embedder is told what the passage is about.
Only within a section does size take over: whole paragraphs are packed greedily,
an oversized paragraph falls back to sentence splitting, and an oversized sentence
is hard-split on whitespace. That last rule never fires on hand-written prose; it
exists so the function cannot loop or emit something enormous.

**Overlap never crosses a heading.** Carrying the tail of the metal section into
the first folk chunk would make the folk chunk match metal queries. The two
subjects would contaminate each other's retrieval in both directions.

**A short final chunk is merged backwards.** A 40-character orphan scores
deceptively high against short queries, because cosine similarity normalises away
length. Merging it into its predecessor costs nothing and removes a whole class of
spurious matches.

---

## The prefix scheme

`gemini-embedding-2` removed the `task_type` parameter that `gemini-embedding-001`
had. Asymmetric retrieval — embedding a question differently from a passage — is
now expressed as instruction text prepended to the input:

- documents: `title: {title} | text: {content}`
- queries: `task: search result | query: {text}`

If retrieval quality ever disappoints, the documented fallback is
`gemini-embedding-001` with explicit `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` task
types. That path is not built, only noted.

**The offline embedder deliberately skips these prefixes**, and the reason is worth
recording because it was found by measurement rather than reasoning. The prefixes
are instruction text that a semantic model interprets. A bag-of-words model does
not interpret anything — it sees the literal tokens `task`, `search`, `result`,
`query` in every single query. Weighted by inverse document frequency, those became
the three highest-scoring terms in every request, above the actual subject. The
query *"jazz music"* retrieved the rock song **"Search and Destroy"**, on the word
*search*. Retrieval accuracy went from 5/32 to 21/32 the moment the prefix was
removed.

The general lesson: a technique that helps one class of model can actively harm
another, and "apply the same preprocessing everywhere" is not automatically right.

**The prefix scheme is versioned into the corpus fingerprint.** Changing either
template alters what gets embedded without altering any chunk's text, so without
that version number a stale index would look perfectly fresh.

---

## Stratified retrieval

Song cards and prose are retrieved to separate depths rather than in one combined
top-k.

Without this, one kind crowds out the other, and which one wins is arbitrary. With
203 cards against 74 prose chunks the cards usually dominate on numbers alone — but
conversational queries actually match prose *more* strongly, since prose shares
vocabulary with how people talk and a templated card does not. In one measured run
the top prose similarities (0.129, 0.124) exceeded every song-card similarity
(0.117). A single top-k would have returned prose only, leaving nothing to
recommend.

Retrieving 20 cards and 3 passages guarantees the prompt gets both the candidates
and the context that explains them.

---

## Blending

The sort key is `0.6 × similarity + 0.4 × (score / 4.0)`.

Weighted toward similarity because retrieval is what understood the request; the
scorer is refining an ordering, not overturning it. But 0.4 is enough that a song
matching genre, mood, and energy exactly will outrank one that merely sounds
related.

**The blend is never written back onto the record.** `RetrievedSong.score` means
what it has always meant — the weighted score out of 4.0 — and every test written
against that still holds. The blend exists only as an ordering decision inside the
retriever.

---

## Confidence

Retrieval similarity is folded into confidence at 30%, leaving the existing
score-and-coverage blend at 70%. Confidence stays mostly a statement about the
scorer, because that is the part whose behaviour is explainable.

Raw cosine cannot be used directly. Similarities do not span [0, 1] in practice —
real Gemini scores cluster in roughly 0.25–0.85 — so fed in raw the term would sit
near 0.6 for everything and carry no information. Each backend declares the band it
actually produces, and scores are rescaled against it. **These bands are tuned
policy, not math**, exactly as the confidence thresholds are.

**The offline embedder declines to contribute at all.** Its similarities were
measured at min −0.01, median 0.06, max 0.16, with variance across queries wide
enough that no single band fits. Lexical overlap is a good enough signal to build a
shortlist from and too noisy to tell a user how sure the system is. So it orders
results and leaves confidence alone — which has the useful side effect that offline
confidence numbers match classic mode exactly, instead of being depressed by an
artefact of the fallback.

---

## Cache invalidation

The index header carries a fingerprint: SHA-256 over every chunk id and chunk text,
sorted, plus the prefix-scheme version.

On startup the fingerprint is recomputed from the live corpus and compared. A
mismatch produces a loud warning naming the drift and a fall back to the offline
embedder. `python3 -m src.ingest --check` exits 1 on the same condition, which is
what a pre-commit hook or CI job runs.

**It never rebuilds automatically.** An automatic rebuild would mean that editing a
CSV silently triggers 277 paid API calls. A warning that a human must act on is the
correct failure mode; a surprise invoice is not.
