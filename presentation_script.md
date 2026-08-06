# HarmonyRanker 1.0 — 5-minute class presentation

> Working notes for the live talk. Not part of the submission — delete or gitignore after class.
> Every command and every output line below was captured from a real run on 2026-08-06.

---

## Command cheat-sheet

```bash
source .venv/bin/activate
```
```bash
python3 -m src.main --query "mellow lofi beats to study to, nothing too energetic" -k 3 --embedder offline --generator template --show-chunks
```
```bash
python3 -m pytest -q
```
```bash
grep "Withholding" logs/run.log
```
```bash
python3 -m src.main --query "mellow lofi beats to study to, nothing too energetic" -k 3 --show-chunks
```

Total runtime: ~1s + ~14s + instant + ~4s.

---

## Pre-flight (do this before you walk up)

- [ ] `cd ~/codepath/applied-ai-system-final && source .venv/bin/activate`
      **This is not optional.** `google-genai` is only installed in `.venv`. Without it the live
      demo silently falls back to the offline embedder and prints a scary yellow index warning.
      Verify with: `python3 -c "import google.genai; print('ok')"`
- [ ] Terminal font to ~18–20pt, window ≥ 80 cols, `clear` the scrollback
- [ ] **Never put `.env` on the projector.** No `cat .env`, no `ls -la` in the repo root, no editor
      open on the root directory. Your live API key is in there.
- [ ] Run all four commands once to warm the imports
- [ ] Slide page open in a second browser tab, scrolled to the top
- [ ] `grep -c Withholding logs/run.log` returns 4 — if it returns 0, the log rotated and demo 3 dies
- [ ] Decide *now* whether wifi is good enough for demo 4. If unsure, skip it and close on the KB
      limitation instead.

---

# THE SCRIPT

Stage directions in `[brackets]`. Speak the rest.

---

## [0:00 – 0:40] THE PROBLEM

*[Slide page up, title section]*

> This is HarmonyRanker — a music recommender you can argue with.
>
> The problem I set out to solve wasn't "recommend good songs." Plenty of things do that. The
> problem was: when a recommender hands you a song, can it tell you *why* — and can you check that
> the reason is true?
>
> So I built under one constraint, and it's the whole thesis: **the model never picks the songs.**
> A language model writes the explanation, but by the time it writes a single word, the list is
> already decided and frozen. If the model tries to change that list, the system throws its answer
> away.
>
> The catalog is 203 songs across 19 genres, plus about 7,200 words of prose I wrote about genres,
> moods, eras, and listening situations. And you type a sentence — not a form.

---

## [0:40 – 1:50] THE LOGIC

*[Scroll to the simplified architecture diagram]*

> It's RAG — retrieval-augmented generation — in three stages.
>
> **Retrieve.** Your sentence becomes a 768-dimension embedding, and I cosine-search 277 vectors:
> one per song, plus the prose chunks. The retrieval is *stratified* — top 20 songs and top 3 prose
> passages, taken separately. That's not a stylistic choice: when I measured it, the prose passages
> out-scored every single song card. A plain top-k would have returned three essays and zero music.
>
> **Rerank.** That shortlist goes through a deterministic weighted scorer — genre plus 1.5, mood
> plus 0.5, energy times 2.0, capped at 4. No model anywhere near it. This is the part that
> actually picks.
>
> **Generate.** The model gets the passages, the query, and the fixed song list, with an instruction
> not to add, drop, or substitute anything.
>
> Let me show you the reading step, because that's the part people don't expect.

*[Switch to terminal. TYPE — this is pinned offline, so it's identical every time and needs no wifi]*

```bash
python3 -m src.main --query "mellow lofi beats to study to, nothing too energetic" -k 3 --embedder offline --generator template --show-chunks
```

*Expected output — the two lines to point at are marked:*

```
  YOUR QUERY
  query: "mellow lofi beats to study to, nothing too energetic"
  understood as: genre=lofi · mood=chill · energy=0.25          <-- POINT HERE
  average confidence: 0.84
====================================================================

  retrieved context:                                             <-- THEN HERE
    +0.143  [listening_contexts:intro:0] listening_contexts
    +0.095  [genres:lofi:1] Lofi
    +0.092  [eras:the-internet-lofi-era:0] The internet lofi era

  1. Staying There  —  L'Indécis
     score 3.80 · confidence 0.97 (high) · similarity 0.015
```

> Look at this line — `understood as: genre=lofi, mood=chill, energy=0.25`.
>
> I said "nothing too **energetic**," and it did *not* read that as energetic. There's negation
> handling in there, and there's no model call in this step at all — it's longest-match over the
> genre and mood vocabularies, with synonyms and negation cues.
>
> And up here, "retrieved context" — those are prose passages, not song rows. That's the knowledge
> base bridging what you *said* to what you *want*. Every recommendation also carries a confidence
> and a band, and where it's a stretch, the explanation says so out loud in a caveat.

---

## [1:50 – 3:20] THE RELIABILITY

> So how do I know any of this works?
>
> 232 tests. But the number isn't the interesting part — this is.

*[Optional: open `conftest.py` for 5 seconds, or just say it]*

> Every single test runs with sockets monkeypatched to raise, and with the API key deleted from the
> environment. So a green suite doesn't mean "it passed on my laptop with my key set." It's proof
> that nothing was sent anywhere and nothing was billed.

*[TYPE]*

```bash
python3 -m pytest -q
```

```
........................................................................ [ 31%]
........................................................................ [ 62%]
........................................................................ [ 93%]
................                                                         [100%]
232 passed in 14.10s
```

> Three guardrails, at three different severities:
>
> — a malformed row in the CSV gets skipped and logged, and the run continues;
> — a fabricated citation drops that one sentence;
> — and if the model names a song the ranker didn't pick, the **entire answer** is discarded and we
> fall back to the deterministic template.
>
> That last one is not hypothetical.

*[TYPE — this is the strongest 20 seconds of the talk. Slow down.]*

```bash
grep "Withholding" logs/run.log
```

```
2026-08-03 ERROR music_rec.explainer  | Withholding explanation for 'Can't Stop the Feeling!': no claim survived grounding
2026-08-03 ERROR music_rec.guardrails | Withholding generated answer: only 0/2 sentences were cited (minimum ratio 0.6)
2026-08-03 ERROR music_rec.guardrails | Withholding generated answer: answer named song(s) the ranker did not select: Master of Puppets
2026-08-06 ERROR music_rec.guardrails | Withholding generated answer: answer named song(s) the ranker did not select: Physical
```

> That bottom line is from **earlier today**. I asked for confident funk. The ranker picked five
> funk songs. Gemini's answer named "Physical" by Dua Lipa.
>
> Here's the part I want you to notice: "Physical" **is** in my catalog. So a naive "is this a real
> song?" check would have waved it straight through. The guardrail doesn't check the catalog — it
> checks *the list the ranker actually selected*. And it threw the whole answer away.
>
> That run's grounding rate dropped to 0.556, and it's sitting in my log on purpose. A system that
> hides its failures is worse than one that has them.

---

## [3:20 – 4:30] THE REFLECTION — what surprised me

*[Scroll the slide page to "The bug". Terminal can stay up.]*

> Early on, the offline retriever — the fallback that runs with no API key — looked completely
> fine. Sensible songs. Confident explanations. Every test green.
>
> Then I actually measured it, and it was scoring 5 out of 32. Near random.
>
> I had two confident theories. Hash collisions — I fixed those, and it got *worse*. Missing IDF
> weighting — I added it, and it barely moved.
>
> The real cause: Gemini's embedding API wants an instruction prefix, `task: search result | query:`.
> I was passing that prefix to my offline bag-of-words fallback too. So the three heaviest terms in
> every query I ever made were "task," "search," and "result."
>
> The query **"jazz music"** was retrieving a rock song called **"Search and Destroy."**
>
> Deleting one prefix took it from 5 out of 32 to 21 out of 32.
>
> But the bug isn't what surprised me. What surprised me is that **the tests were green the whole
> time — and they were right to be.** Every function did exactly what I'd told it to do. Tests prove
> the code does what you said. They don't prove you said the right thing. Nothing caught this except
> measuring output quality against expectations, and I didn't have that harness until I built it.

---

## [4:30 – 5:00] CLOSE

> One honest limitation before I stop.
>
> Those 7,200 words of prose steering every retrieval? I wrote all of them. My opinions about what
> city pop is *for* are now infrastructure for everyone who uses this. I have more to say about city
> pop than about ska, and that asymmetry is measurable in the results. That's what I'd fix next.

### Optional — the live call (only if wifi is good; ~4 seconds)

*[TYPE — same question as before, but with the real backends]*

```bash
python3 -m src.main --query "mellow lofi beats to study to, nothing too energetic" -k 3 --show-chunks
```

> Same question — now with the API key on. Watch the similarity numbers.

```
  understood as: genre=lofi · mood=chill · energy=0.25
  average confidence: 0.91

  retrieved context:
    +0.780  [genres:lofi:0] Lofi
    +0.780  [moods:focused:0] Focused          <-- POINT HERE
    +0.777  [moods:chill:0] Chill

  ANSWER (gemini-flash-lite-latest):
    1. "Staying There" — L'Indécis features muted drums, warm and
    slightly detuned keys, vinyl crackle, tape hiss ... [genres:lofi:0]
```

> Offline, average similarity was 0.09. With Gemini, 0.71.
>
> And look at the second passage it pulled: **"Focused."** I never used that word. It connected
> "study to" to focus through the prose, not through a keyword.
>
> Now it's writing real sentences instead of a template — and every claim carries the passage it
> came from, in brackets. Those brackets are what the guardrail checks.
>
> That's HarmonyRanker. Happy to take questions.

**If it hangs more than ~8 seconds:** Ctrl-C, and say —
> "That's the network, and honestly it makes my point — this is why the offline path is a
> first-class citizen and not a degraded mode."
Then re-run with `--embedder offline --generator template`.

---

# Q&A PREP

**Why not just let the LLM pick the songs?**
Accountability. A bad pick from my scorer is a weight I can point at and change. A bad pick from a
model is a shrug. Also, the model can't see `energy=0.3` — it understands "for a late night drive."
The scorer is the exact opposite. Each stage does the thing the other one is bad at.

**How do you know retrieval is actually good?**
Recall@20 for the requested genre: 19 out of 19 genres, 7 out of 7 demo queries. Top-1 accuracy is
14/19 offline versus 19/19 with Gemini — I deliberately didn't tune the offline path up to match,
because it's a fallback, and pretending it's as good would be the dishonest move.

**What's the biggest weakness?**
Adding RAG made it more capable and slightly *less* accountable in the same change. The reranker is
fully inspectable. The retrieval shortlist is not — I can't tell you why one embedding beat another.
And explanations are no longer reproducible run to run, which an earlier version of my README
incorrectly claimed they were.

**What does it cost to run?**
One embedding call and one generation call per query. The index build is one-time — 277 vectors,
committed to the repo. And it never rebuilds automatically, even when the CSV changes: a surprise
invoice is not an acceptable failure mode. It warns and tells you to run `python3 -m src.ingest`.

**Why is energy weighted 2.0 — isn't that too high?**
It is, and it's a known flaw I documented rather than quietly fixed. The formula is
`1 - abs(difference)` on a 0-to-1 scale, which means *every* song earns some energy points. That
quietly demoted genre and mood to tie-breakers. Interestingly, an AI assistant spotted the math bug
but proposed a weight rebalance that didn't actually fix the underlying issue.

**What happens with no API key at all?**
The whole pipeline still runs — deterministic hashing embedder, template explainer, zero network.
That's what I showed you first. It's also what the entire test suite runs against.

**What's a "chunk id"?**
The citation handle — `song:47`, or `genres:city-pop:0`. When the model writes a claim it has to
tag it with one, and the guardrail verifies that tag against what was actually retrieved. Untagged
sentences count against a density floor of 0.6.

**Did you use AI to build this?**
Yes, and I logged where it helped and where it didn't. It caught the energy-baseline math bug. It
also proposed a fix for the retrieval problem that was confidently wrong — I only found the real
cause by measuring. And it warned me about a Gemini batch-embedding footgun where passing a list
returns one averaged vector, silently; that's guarded in three places now.

---

# DO NOT PUT ON SCREEN

- `.env` — live API key
- `ai_interactions.md` — still blank scaffold (the content lives in `model_card.md` §12)
- `model_card.md` — still has the assignment's boilerplate interleaved ("Example: **VibeFinder 1.0**")
- `docs/algorithm_recipe.md` — deliberately superseded; its weights (2.0/1.0/1.0) contradict the
  shipped ones (1.5/0.5/2.0). If someone asks, lead with "that's a design record, not current."
