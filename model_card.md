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

---

## 4. Data  

Describe the dataset the model uses.  

Prompts:  

- How many songs are in the catalog  
- What genres or moods are represented  
- Did you add or remove data  
- Are there parts of musical taste missing in the dataset  

The recommender uses a song library of 203 songs across 19 genres, each with labels for title, artist, genre, mood, and sound features like energy, tempo, positivity, danceability, and acousticness. It does not listen to the music; it compares these labels and numbers to your preferences. The library started at 18 songs, where most genres had only one entry and variety was poor; every genre now has at least 10 songs (jazz has 23). Moods are still thinner — 13 of them across the whole library — so it is much easier to find a song in the genre you asked for than one in the mood you asked for.

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

## 12. Collaboration with AI

I used AI throughout the project to write scoring functions, expand the song library, and stress-test the recommender's logic. The most helpful moment was when the AI spotted the energy bias: it pointed out that because energy difference is on a 0–1 scale, the energy term always adds points even for terrible matches, which explained why "energy was a close match" appeared in nearly every recommendation. I hadn't noticed that pattern was a math problem, not a coincidence. A flawed suggestion came when I asked it to rebalance the weights. AI initially suggested changes that looked reasonable but didn't fix the real issue, since energy still contributed a positive baseline to every song no matter its weight. I had to test each function myself to catch that, which reinforced that AI suggestions need verification rather than blind trust.
