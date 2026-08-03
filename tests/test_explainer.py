"""
Tests for the Explanation Agent.

The central property under test is grounding: the agent may only say things
the retrieved record supports, and when it can't, it must say nothing rather
than something plausible.
"""

from src.explainer import WITHHELD_EXPLANATION, ExplanationAgent
from src.main import PROFILES
from src.models import BAND_HIGH, BAND_LOW
from src.recommender import Recommender, build_record
from src.models import Song, UserProfile
from src.retriever import Retriever

POP_SONG = {
    "id": 1, "title": "Test Pop Track", "artist": "Test Artist",
    "genre": "pop", "mood": "happy", "energy": 0.8, "tempo_bpm": 120,
    "valence": 0.9, "danceability": 0.8, "acousticness": 0.2,
}


def pop_prefs():
    return {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}


def rock_prefs():
    return {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.8}


# --- generation is driven by the retrieved evidence -------------------------

def test_perfect_match_explanation_cites_all_three_signals():
    record = build_record(pop_prefs(), POP_SONG)
    text = ExplanationAgent().explain(pop_prefs(), record)

    assert text.startswith("Strong match")
    assert "pop" in text and "happy" in text and "0.80" in text
    assert record.band == BAND_HIGH


def test_weak_match_hedges_and_states_the_mismatch():
    # A rock listener handed a pop song: the agent must not sound confident,
    # and must name what actually differs.
    record = build_record(rock_prefs(), POP_SONG)
    text = ExplanationAgent().explain(rock_prefs(), record)

    assert record.band == BAND_LOW
    assert text.startswith("Weak match")
    assert "not the rock you asked for" in text
    assert "stretch pick" in text


def test_explanation_reports_the_confidence_it_was_given():
    record = build_record(pop_prefs(), POP_SONG)
    text = ExplanationAgent().explain(pop_prefs(), record)

    assert f"confidence {record.confidence:.2f}" in text


def test_explanation_is_deterministic():
    record = build_record(pop_prefs(), POP_SONG)
    agent = ExplanationAgent()

    assert agent.explain(pop_prefs(), record) == agent.explain(pop_prefs(), record)


# --- grounding guardrail ----------------------------------------------------

def test_ungrounded_claim_is_dropped_not_printed():
    record = build_record(pop_prefs(), dict(POP_SONG))
    # Simulate the record and its evidence drifting apart: the breakdown still
    # says "pop" while the record now says "polka".
    record.song["genre"] = "polka"

    agent = ExplanationAgent()
    text = agent.explain(pop_prefs(), record)

    assert "pop" not in text            # the unsupported claim never reaches the user
    assert agent.stats["claims_dropped"] == 1
    assert agent.grounding_rate() < 1.0


def test_explanation_is_withheld_when_no_claim_survives():
    record = build_record(pop_prefs(), dict(POP_SONG))
    record.song.update({"genre": "polka", "mood": "gloomy", "energy": 0.01})

    agent = ExplanationAgent()
    text = agent.explain(pop_prefs(), record)

    assert text == WITHHELD_EXPLANATION
    assert agent.stats["withheld"] == 1


def test_grounding_rate_is_one_for_a_clean_run():
    agent = ExplanationAgent()
    for record in Retriever.from_csv("data/songs.csv").retrieve(pop_prefs(), k=5):
        agent.explain(pop_prefs(), record)

    assert agent.grounding_rate() == 1.0
    assert agent.stats["withheld"] == 0


def test_no_explanation_ever_names_a_genre_the_song_does_not_have():
    # Sweep the whole catalog across every demo profile and assert the agent
    # never attributes another song's genre to the one it is explaining.
    # Driven off PROFILES so a newly added profile is swept too.
    retriever = Retriever.from_csv("data/songs.csv")
    all_genres = set(retriever.genre_index)
    agent = ExplanationAgent()

    for profile in PROFILES.values():
        prefs = profile["prefs"]
        for record in retriever.retrieve(prefs, k=len(retriever.songs)):
            text = agent.explain(prefs, record)
            song_genre = record.song["genre"]
            for genre in all_genres:
                # The substring check is deliberately strict, so genres that
                # nest inside a legitimately named one have to be excused:
                # "pop" is inside both "indie pop" (the song's genre) and
                # "the city pop you asked for" (the requested genre).
                if genre in song_genre or genre in prefs["favorite_genre"]:
                    continue
                assert genre not in text, f"{genre!r} leaked into: {text}"

    assert agent.stats["claims_dropped"] == 0


# --- the OOP facade uses the same agent -------------------------------------

def test_recommender_facade_explains_through_the_agent():
    song = Song(**POP_SONG)
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy", target_energy=0.8,
        target_valence=0.9, target_danceability=0.8, target_acousticness=0.2,
    )

    recommender = Recommender([song])

    assert recommender.recommend(user, k=1) == [song]
    explanation = recommender.explain_recommendation(user, song)
    assert explanation.startswith("Strong match")
    assert "pop" in explanation
