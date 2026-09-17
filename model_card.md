# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

Give your model a short, descriptive name.  
Example: **VibeFinder 1.0**  

HarmonyRanker 1.0
---

## 2. Intended Use  

Describe what your recommender is designed to do and who it is for. 

Prompts:  

- What kind of recommendations does it generate  
- What assumptions does it make about the user  
- Is this for real users or classroom exploration  

A content-based music recommendation model using weighted preference scoring.

---

## 3. How the Model Works  

Explain your scoring approach in simple language.  

Prompts:  

- What features of each song are used (genre, energy, mood, etc.)  
- What user preferences are considered  
- How does the model turn those into a score  
- What changes did you make from the starter logic  

Avoid code here. Pretend you are explaining the idea to a friend who does not program.

Every song gets a score based on how well it matches your preferences for genre, mood, and energy. The better the match, the higher the score, and the highest-scoring songs are recommended first. The three parts are not worth the same: getting the genre right earns 1.5 points, getting the mood right earns 0.5, and energy is worth up to 2.0 depending on how close it lands. Genre counts for three times as much as mood on purpose — if you ask for rock, a confident rock song should beat an intense funk one.

**What changed when the system learned to read plain English.** You can now type "something nostalgic for a late night drive" instead of filling in three fields. Behind that, the system does something in two steps rather than one.

First it *finds* candidates. Every song has been turned into a short written description ("a city pop song with a nostalgic mood, energy is high…"), and those descriptions — along with about 7,200 words I wrote about genres, moods, eras, and listening situations — have been converted into lists of numbers that capture roughly what a piece of text is *about*. Your question gets converted the same way, and the system pulls the twenty songs whose descriptions are closest to it, plus a few background passages.

Then it *ranks* them, using exactly the same scoring rule as before. This split exists because the two halves are good at opposite things. The text-matching half understands "for a late night drive" but has no idea what an energy of 0.3 means. The scoring rule knows precisely what 0.3 means and cannot read a sentence. Neither one alone does the job.

---

## 3a. Models used

Two Google Gemini models, one API key. Both are optional: with no key the system runs end to end on a local text-matching method and its original template explanations.

| Role | Model | What it does |
|---|---|---|
| Retrieval | `gemini-embedding-2` (768 dimensions) | Converts song descriptions, background prose, and your question into number lists so they can be compared |
| Generation | `gemini-flash-lite-latest` | Writes the sentence or two explaining each recommendation |

**The most important thing on this page: the model does not choose the songs.** Retrieval narrows the field and the scoring rule picks the winners. The language model is handed a finished list and asked only to describe it, in a prompt that says explicitly *do not add, drop, or substitute*. So the answer to "why was I recommended this?" is still a number produced by a weight I chose, not a decision made inside a model I cannot inspect.

**What it is allowed to do:** write at most two sentences per song, using only facts contained in the passages it was given, ending each factual sentence with a marker identifying which passage the fact came from.

**What it is not allowed to do, and what stops it:** every sentence citing a passage that was not actually retrieved is deleted before you see it. Any answer mentioning a song the ranker did not select is thrown away *entirely* and replaced with the original template explanation — not partially repaired, because a model quietly swapping in a different song is precisely the failure this whole design exists to prevent. An answer where too few sentences are supported is discarded the same way. All three checks are in `src/guardrails.py`; the prompt itself is in `src/answerer.py`.

An honest limit: these checks verify *provenance*, not *truth*. A sentence that cites a real passage and then misdescribes it would get through. Capping the model at two sentences per song limits the room for that, without eliminating it.

**What leaves the machine:** your question, the retrieved song descriptions, and the retrieved background prose. All of it comes from `data/songs.csv` and `docs/kb/`, which are in this repository. No personal data, no listening history, no identifiers. There is nothing to leak because the system stores nothing about you between runs.

**Version drift:** `gemini-flash-lite-latest` is an alias that follows Google's latest release. Any output recorded in this repository is dated, and a run six months from now may legitimately differ for reasons that have nothing to do with this code.

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

The recommender uses a song library of 203 songs across 19 genres, each with labels for title, artist, genre, mood, and sound features like energy, tempo, positivity, danceability, and acousticness. It does not listen to the music; it compares these labels and numbers to your preferences. The library started at 18 songs, where most genres had only one entry and variety was poor; every genre now has at least 10 songs (jazz has 23). Moods are still thinner — 13 of them across the whole library — so it is much easier to find a song in the genre you asked for than one in the mood you asked for.

**A second data source, and a new place bias enters.** There is now roughly 7,200 words of prose in `docs/kb/` describing what each genre sounds like, what each mood means musically, seven musical eras, and ten listening situations. It is searched alongside the songs, and it is how a request like "something for the gym" reaches the label `energetic`.

**I wrote all of it, and that is a problem worth stating plainly.** These are one person's characterisations. When I write that reggae suits "sunshine, cooking, and social gatherings that should stay relaxed", that is an opinion shaped by how I encountered the genre — and it is now infrastructure, steering what gets retrieved for every user. A reggae listener who associates the genre primarily with protest music is served worse by my description, and nothing in the system surfaces that. The song data at least came from an external source with its own documented labels; this prose has no such check on it.

The scale of it makes this sharper, not softer: 3,500 words on genres means the genres I find easiest to describe are the ones most likely to be retrieved well. I have more to say about city pop than about ska, and that asymmetry is now measurable in the system's behaviour.

There is a second, quieter bias in the retrieval model itself. `gemini-embedding-2` was trained on internet text, which is English-dominant and popularity-weighted. Genres with a large online footprint are represented more richly in its internal space than genres without one. A catalog spanning regional traditions with thin web presence would retrieve those traditions worse, and the system would give no indication that it was doing so.

---

## 5. Strengths  

Where does your system seem to work well  

Prompts:  

- User types for which it gives reasonable results  
- Any patterns you think your scoring captures correctly  
- Cases where the recommendations matched your intuition  

The system seems to work well when someone has a clear, simple taste like "high-energy pop" or "calm, low-energy lofi". 

---

## 6. Limitations and Bias 

Where the system struggles or behaves unfairly. 

Prompts:  

- Features it does not consider  
- Genres or moods that are underrepresented  
- Cases where the system overfits to one preference  
- Ways the scoring might unintentionally favor some users  

Energy is added to every song, even bad matches. Since energy is on a 0-1 scale, abs(diff) never seeds 1.0, so energy_points is essentially always positive (0 to 2.0). Combined with weight (W_ENERGY = 2.0 vs. 1.5 for genre and 0.5 for mood), energy is both the loudest signal and a near-constant baseline. "Energy was a close match" appears in almost every single recommendation. Genre and mood become tie-breakers on top of an energy-driven ranking, not the other way around. Within that tie-break, genre is weighted 3x mood, so a wrong-genre song can no longer match an on-genre one by hitting the mood — but the flip side is that mood, at 0.5, now barely moves the ranking at all.

**New limitations introduced by adding retrieval and generation.**

*It can now make things up, in a narrow band.* Before, the explanations were assembled from templates and could not say anything the data did not support. A language model can. The three guardrails in §3a catch fabricated sources and swapped songs, but they check where a claim came from, not whether it is accurate — a sentence citing a real passage and describing it wrongly would pass.

*Two songs can now be recommended for a reason nobody can inspect.* The scoring rule is legible: 1.5 for genre, 0.5 for mood, energy times 2.0. The retrieval step that produces the shortlist is not. If a genuinely good match never enters the top twenty, the scoring rule never sees it, and there is no way to explain that omission in terms a user could act on. The system became more capable and slightly less accountable in the same change.

*The prose knowledge base is an opinion.* Covered in §4, but it belongs here too: my characterisations of 19 genres now shape what every user is shown.

*Explanations are no longer reproducible.* Run the same query twice with Gemini enabled and the wording may differ. The *songs* will not — selection is still fully deterministic — but a claim that the whole system is reproducible would now be false, and earlier versions of the README made exactly that claim.

*It depends on a company.* The system needs an API key, an internet connection, and Google continuing to serve these models. The offline fallback exists so that none of those is fatal, but the offline path is genuinely worse at retrieval, and a user who never sets a key gets a quietly inferior product.

*Understanding a request is heuristic.* Turning "nothing too energetic" into structured preferences uses word matching, a synonym list, and a negation rule — not real language understanding. It handles the phrasings I tested. It will mishandle sarcasm, comparisons ("less intense than metal"), and anything conditional. The damage is bounded: it only affects ranking within an already-relevant shortlist, never what gets retrieved.

---

## 7. Evaluation  

How you checked whether the recommender behaved as expected. 

Prompts:  

- Which user profiles you tested  
- What you looked for in the recommendations  
- What surprised you  
- Any simple tests or comparisons you ran  

No need for numeric metrics unless you created some.

Pop profile prefers higher energy songs. Lofi profile prefers lower energy songs.

**Evaluation after adding retrieval and generation.** 252 automated tests across 12 files. The suite blocks network access and deletes the API key before every test, so a passing run is evidence that nothing was sent to Google and nothing was billed — and that a machine with a key produces the same results as one without.

The measures that mattered most:

| Measure | Result | Why it is the right measure |
|---|---|---|
| Recall@20 for the requested genre (offline) | 19/19 genres, 7/7 demo queries | The only property the two-stage design depends on. If the shortlist contains the right genre, the scoring rule can surface it |
| Recall@20 for the requested genre (Gemini) | 19/19 genres | Same measure on the better backend |
| Top-1 retrieval accuracy | 14/19 offline, 19/19 Gemini | Deliberately *not* optimised. The reranker fixes ordering, so this number can be mediocre without hurting output — and the gap between the two backends is what a key buys |
| Fabricated citations | 0 across all demo runs | Sentences citing a source that was not retrieved |
| Substituted songs | 0 across all demo runs | Answers naming a song the ranker did not select |
| Grounding rate | 1.00 | Share of generated sentences surviving the checks |

Every row is reproduced by a command in the README: the retrieval rows by [E12](README.md#e12--retrieval-quality), the grounding rows by [E2](README.md#e2--full-demo-run-all-seven-profiles) and [E5](README.md#e5--reliability-sweep-across-all-seven-profiles). An earlier version of this table reported top-1 as `21/32`, measured with an ad-hoc 32-query probe set that was never committed; the figures above replace it with a measurement anyone can re-run.

The most useful test is the one that deliberately misbehaves. `BrokenGenerator` produces a fabricated citation and a swapped-in song on purpose, and the suite asserts that the first loses one sentence while the second is discarded entirely. Testing that the guardrail *fires* is worth more than testing that it stays quiet when nothing is wrong.

**What surprised me during evaluation** is recorded in §11 — the offline retrieval failure was found by measuring, after two wrong guesses about its cause.

---

## 8. Future Work  

Ideas for how you would improve the model next.  

Prompts:  

- Additional features or preferences  
- Better ways to explain recommendations  
- Improving diversity among the top results  
- Handling more complex user tastes  

I would improve the model with a bigger library of songs and genres so it makes better picks. I'd let mood, danceability, and tempo count more instead of energy doing most of the deciding. More variety in the top picks so the listener actually discovers something. Handle complicated taste to uggle more than one mood.

---

## 9. Personal Reflection  

A few sentences about your experience.  

Prompts:  

- What you learned about recommender systems  
- Something unexpected or interesting you discovered  
- How this changed the way you think about music recommendation apps  

I learned these systems don't actually understand music. They just compare labels, numbers, and add up a score using simple math. AI tools helped me to see what biases the recommender may run into. I needed to double-check them after implementing each function. What surprised me was how much one setting can take over. I gave energy extra weight and suddenly it was steering almost every recommendation so genre and mood barely mattered. Now when Spotify suggests a song, I think about whoever decided which factors count and how much. 

## 10. Misuse

Could your AI be misused, and how would you prevent that?

Misuse risks: Someone could game the scoring (artists mislabeling songs' genre/energy to rank higher), the system could be used to push certain artists unfairly if weights were tuned with a commercial bias, or user preference data could be collected and used for profiling beyond recommendations.

Prevention: Validate song metadata from a trusted source rather than self-reported labels, keep the scoring logic transparent (the model card itself helps here), don't store user preferences longer than needed, and audit recommendations periodically to check no artist or genre is being systematically favored beyond what the weights intend.

## 11. Testing AI's reliability

What surprised me was that reliability problems don't always look like errors. The system never crashed and its output always seemed reasonable. Energy quietly dominated every ranking because its scoring always adds points, even for bad matches. The tell was suspicious consistency: "energy was a close match" in nearly every explanation. I learned to check why something scored high, not just whether the output looked fine.

**The same lesson, harder, when I added retrieval.** The offline retrieval path returned results that looked fine and were nearly random — a request for mellow lofi returned metal. Nothing raised, nothing logged a warning, and the ranked list looked exactly like a working one.

I guessed at the cause twice and was wrong both times. First I assumed hash collisions and raised the vector size sixteenfold; accuracy got *worse*. Then I assumed the shared template wording in the song descriptions was drowning out the genre words, added inverse-document-frequency weighting, and it barely moved.

Only when I stopped guessing and printed the actual weight of every term in a query did the cause appear. The instruction prefix that Gemini's embedding model expects — `task: search result | query:` — was being fed to the local text-matching fallback too, which cannot interpret instructions and simply saw the words. `search`, `result`, and `task` became the three heaviest terms in every single query, outweighing the subject. The query "jazz music" was retrieving the rock song **"Search and Destroy"**, matching on the word *search*. Removing the prefix from that one code path took accuracy from 5/32 to 21/32.

Three things I took from it. Reliability failures in a retrieval system look like plausible results, not exceptions — there is no equivalent of a stack trace. Two confident hypotheses in a row can both be wrong, and the cost of guessing is much higher than the cost of measuring. And a technique that is correct for one model can be actively harmful to another; "apply the same preprocessing everywhere" is an assumption, not a principle.

## 12. Collaboration with AI

I used AI throughout the project to write scoring functions, expand the song library, and stress-test the recommender's logic. The most helpful moment was when the AI spotted the energy bias: it pointed out that because energy difference is on a 0–1 scale, the energy term always adds points even for terrible matches, which explained why "energy was a close match" appeared in nearly every recommendation. I hadn't noticed that pattern was a math problem, not a coincidence. A flawed suggestion came when I asked it to rebalance the weights. AI initially suggested changes that looked reasonable but didn't fix the real issue, since energy still contributed a positive baseline to every song no matter its weight. I had to test each function myself to catch that, which reinforced that AI suggestions need verification rather than blind trust.

**Adding retrieval produced a sharper example of the same thing.** The Gemini embedding API has a trap: if you hand it a list of texts, it returns *one* averaged vector for the whole list instead of one per text. There is no error. Every song would have received an identical vector and retrieval would have degraded to noise — the exact failure mode that looks like a working system. Each item has to be wrapped in its own `Content` object.

I would not have found that by reading my own code, because the code looks right either way. I found it because the documentation named it, and I then guarded it in three places: an assertion inside the embedding call that fires in production, a test asserting we send the correct object type, and a test asserting a wrongly-aggregating response raises rather than writing a corrupt index.

The broader pattern across both halves of this project: the dangerous failures were the silent ones. Energy dominating the ranking, batch embeddings collapsing into one vector, an instruction prefix poisoning a text-matching fallback — none raised an exception, and all three produced output that looked entirely reasonable. What caught them was deciding in advance what the numbers *should* look like and then checking, rather than reading output and judging whether it seemed fine.
