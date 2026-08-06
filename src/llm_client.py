"""
Generation backends — the "generate" stage of the RAG pipeline.

`GeminiGenerator` mirrors the shape used in the DocuBot activity
(`ai110-module4tinker-docubot-starter/llm_client.py`): one module-level model
constant, `generate_content`, `(response.text or "").strip()`, and a try/except
around the call.

One deliberate difference from that starter: DocuBot *returns* the error string
as the answer, so a failed call prints "API error - ..." where an answer should
be. Here the generator **raises** instead, which lets AnswerAgent fall back to
the deterministic template explainer. A recommender that silently degrades to a
correct-but-plainer explanation is better than one that prints a stack trace
fragment to the user.

`TemplateGenerator` is the offline backend. The two test doubles at the bottom
live here rather than in tests/ so they can be imported without making tests a
package.
"""

import abc
from typing import List, Optional

from src.config import GEMINI_GENERATION_MODEL, api_key
from src.logging_setup import get_logger

logger = get_logger("llm_client")

# Single constant so the model is swappable in one edit. `-latest` tracks
# upstream automatically; see model_card.md on why that means recorded outputs
# are dated.
GEMINI_MODEL_NAME = GEMINI_GENERATION_MODEL


class GenerationError(Exception):
    """A generation call failed."""


class GenerationUnavailable(Exception):
    """The generation backend cannot be constructed (usually a missing key)."""


class Generator(abc.ABC):
    """Interface every generation backend implements."""

    name: str = "generator"

    @abc.abstractmethod
    def generate(self, prompt: str) -> str:
        """Returns generated text, or raises GenerationError."""


class GeminiGenerator(Generator):
    """Text generation via the google-genai SDK."""

    def __init__(self, key: Optional[str] = None, model: str = GEMINI_MODEL_NAME):
        key = (key or api_key()).strip()
        if not key:
            raise GenerationUnavailable(
                "Missing GEMINI_API_KEY environment variable. "
                "Set it in your shell or .env file to enable LLM features."
            )

        # Lazy import: the offline path must not require google-genai.
        try:
            from google import genai
        except ImportError as exc:
            raise GenerationUnavailable(
                "google-genai is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.client = genai.Client(api_key=key)
        self.model = model
        self.name = model

    def generate(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
            )
        except Exception as exc:  # the SDK surfaces many transport error types
            raise GenerationError(f"{type(exc).__name__}: {exc}") from exc

        return (response.text or "").strip()


class TemplateGenerator(Generator):
    """
    The deterministic offline backend.

    Delegates to the existing ExplanationAgent and appends the citation marker
    the grounding check expects. That matters: it means the offline path runs
    through `check_generated_answer` for real rather than bypassing it, so the
    guardrail is exercised by every keyless run and every test.
    """

    name = "template"

    def __init__(self) -> None:
        # Imported here to avoid a module-level cycle: explainer imports
        # guardrails, guardrails imports config, and answerer imports both.
        from src.explainer import ExplanationAgent
        self.agent = ExplanationAgent()

    def generate(self, prompt: str) -> str:  # pragma: no cover - not prompt-driven
        raise GenerationError(
            "TemplateGenerator does not consume prompts; AnswerAgent calls it directly"
        )


class FakeGenerator(Generator):
    """Test double that replays a fixed list of responses in order."""

    name = "fake"

    def __init__(self, responses: List[str]):
        self.responses = list(responses)
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.responses:
            raise GenerationError("FakeGenerator ran out of scripted responses")
        return self.responses.pop(0)


class BrokenGenerator(Generator):
    """
    Test double that misbehaves the way a real model can.

    Emits a fabricated citation and, optionally, the title of a song the ranker
    never selected. Exists so the grounding guardrail is proven to fire against
    output no honest backend would produce.
    """

    name = "broken"

    def __init__(self, substitute_title: Optional[str] = None):
        self.substitute_title = substitute_title

    def generate(self, prompt: str) -> str:
        if self.substitute_title:
            return (
                f'Strong match: you should listen to "{self.substitute_title}", '
                f"which is a perfect fit [song:9999]."
            )
        return "This song is a great pick for you [song:9999]. It has excellent vibes."


class RaisingGenerator(Generator):
    """Test double that always fails, to exercise the fallback path."""

    name = "raising"

    def generate(self, prompt: str) -> str:
        raise GenerationError("simulated API failure")


def make_generator(preference: str = "auto") -> Optional[Generator]:
    """
    Resolves a generator from a CLI preference.

    Returns None for 'off', meaning retrieval runs but nothing is generated.
    'auto' uses Gemini when a key is available and falls back to the template
    backend otherwise, logging the reason so a silent downgrade is never
    mistaken for a working Gemini run.
    """
    if preference == "off":
        return None

    if preference == "template":
        return TemplateGenerator()

    if preference == "gemini":
        return GeminiGenerator()

    try:
        return GeminiGenerator()
    except GenerationUnavailable as exc:
        logger.info("Falling back to the template explainer: %s", exc)
        return TemplateGenerator()
