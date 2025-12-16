# p_book_gen/custom_agents.py
"""
Custom agent factory for the parallel book generator 

Current behaviour:

- Expect the user to send ONE JSON object as their first message, with fields like:
  {
    "book_topic": "...",
    "author_name": "...",
    "author_bio": "...",
    "author_voice_style": "...",
    "target_audience": "...",
    "book_purpose": "...",
    "min_chapters": 20
  }

Workflow:

1) Parallel chapter generation
   - Three LlmAgents (chapters 1..N) run in parallel.
   - Each writes a complete chapter and stores it in state under:
       chapter_1_text, chapter_2_text, ..., chapter_N_text

2) Merge agent
   - Single LlmAgent reads the original JSON (from conversation) and
     the chapter texts from state.
   - Produces a single Markdown manuscript with:
       - Title page
       - Short introduction
       - Chapters 1..N in order.

No tools or GCS saves yet - this step is to stabilise the fan-out / fan-in pattern.
"""

#from typing import List

#from google.adk.agents import LlmAgent, SequentialAgent
#from google.adk.agents.parallel_agent import ParallelAgent
import os
import uuid
from datetime import datetime
from typing import List, AsyncGenerator

from google.cloud import storage
from google.genai import types as genai_types

from google.adk.agents import BaseAgent, LlmAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.adk.agents.parallel_agent import ParallelAgent


GEMINI_MODEL = "gemini-2.0-flash"  # or any other model you prefer

# ---------------------------------------------------------------------
# Outline / planning agent
# ---------------------------------------------------------------------


def _outline_instruction(num_chapters: int) -> str:
    """
    Instruction for the outline planning agent.

    It reads the JSON spec from the user message and produces a chapter-by-chapter
    outline, including a UNIQUE famous quote for each chapter.
    """
    return f"""
You are a planning agent for a non-fiction Kindle book.

The user will provide ONE JSON object as their first message. It will include fields like:
- book_topic
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

Your task:

1. Read and understand the JSON object.
2. Create a chapter-by-chapter outline for exactly {num_chapters} chapters.
3. For each chapter, define:
   - Chapter number
   - Chapter title
   - A short one-line description of the chapter's focus
   - One specific, well-known, real quote from a respected person
     that matches the chapter theme.
4. Each quote must be:
   - Real and widely attributed to an actual person
   - Assigned to exactly ONE chapter in this outline
   - NOT reused across chapters (no duplicate quotes, no duplicate quote texts)
5. Try to vary the voices across chapters (e.g. mix philosophers, leaders, writers,
   technologists, etc.), but always stay relevant to the book topic.

Output format:

- Use plain text or Markdown.
- Use a numbered list, one entry per chapter, in this pattern:

  Chapter 1: <Chapter title>
  Description: <one line>
  Quote: "Exact quote text."
  Author: <Name of person>

  Chapter 2: ...
  ...

- Do NOT include any JSON or code in your output.
- Do NOT mention tools or models.
- Use UK English spelling wherever applicable.
"""


def build_outline_agent(num_chapters: int) -> LlmAgent:
    """
    Build the planning agent that creates the outline and assigns
    a unique quote to each chapter.

    The outline text is stored in state under 'chapter_outline'.
    """
    return LlmAgent(
        name="outline_agent",
        model=GEMINI_MODEL,
        description="Creates the book outline and assigns a unique quote to each chapter.",
        instruction=_outline_instruction(num_chapters),
        output_key="chapter_outline",
    )

# ---------------------------------------------------------------------
# Front matter agent (Dedication, Foreword, Intro, About the Author)
# ---------------------------------------------------------------------


def _front_matter_instruction() -> str:
    """
    Instruction for the front-matter agent.

    It creates dedication, foreword, introduction, and about-the-author
    pages in UK English, based on the JSON spec.
    """
    return """
You are a specialist non-fiction book writer responsible for the front matter
of a Kindle-ready book.

The user will provide ONE JSON object as their first message. It will include fields like:
- book_topic
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

Your task:

1. Read and understand the JSON object.
2. Draft the following front-matter sections, in order:

   a) Dedication
      - 1–3 short lines, personal and sincere.
      - Do not invent specific names unless the JSON explicitly provides them.
      - If you cannot infer specific people, keep the dedication general
        (for example “To everyone who chooses to lead when it would be easier to stand still.”).

   b) Foreword
      - 500–800 words.
      - Written in a third-person voice, as if by a thoughtful peer who
        understands why this book matters for the target audience.
      - Explain the problem space, why the author is credible, and what
        the reader will gain.

   c) Introduction
      - 800–1,200 words.
      - Written in the author’s own voice (first person or close third,
        as feels natural), aligned with author_voice_style.
      - Set out the central promise of the book, how it is structured,
        and how the reader should engage with it.

   d) About the Author
      - 250–400 words.
      - Based on author_name and author_bio in the JSON.
      - You may lightly elaborate on the bio for narrative flow, but do
        not invent specific employers or roles that clearly contradict
        the given bio.

3. Use UK English spelling and a natural human voice throughout.
4. Do not mention models, AI, or tools.
5. Do not include any JSON or code in your output.

Output format:

- Use Markdown with clear headings, for example:

  # Dedication
  ...

  # Foreword
  ...

  # Introduction
  ...

  # About the Author
  ...

- Make the sections self-contained so they can be dropped directly into the book.
"""


def build_front_matter_agent() -> LlmAgent:
    """
    Build the agent that generates front matter.

    The result is stored in state under 'front_matter'.
    """
    return LlmAgent(
        name="front_matter_agent",
        model=GEMINI_MODEL,
        description="Creates Dedication, Foreword, Introduction, and About the Author.",
        instruction=_front_matter_instruction(),
        output_key="front_matter",
    )


def _chapter_instruction(chapter_number: int) -> str:
    """
    Create the system instruction for a given chapter agent.

    The agent will:
    - Read the JSON spec from the user's first message.
    - Write one complete chapter with heading, subheading, and body text.
    - Use UK English and a natural human voice.
    """
    return f"""
You are a specialist non-fiction book writer for chapter {chapter_number} of a Kindle book.

The user will provide ONE JSON object as their first message. It will include fields like:
- book_topic
- author_name
- author_bio
- author_voice_style
- target_audience
- book_purpose
- min_chapters

Before your turn, an outline planning agent has already created a full outline
for all chapters, including a unique famous quote for each chapter. That outline
appears earlier in the conversation and is also stored in shared state.

Your task for chapter {chapter_number}:

1. Carefully read and parse the JSON object from the user message.
2. Read the outline that was generated earlier and identify the entry for chapter {chapter_number}.
   - Use the chapter title and short description from that outline as the basis for this chapter.
   - Use the EXACT quote and author assigned to chapter {chapter_number} in the outline.
3. Use UK English and a natural, human narrative voice. Avoid any meta-commentary about being an AI.
4. Draft a chapter of about 1000–2000 words with:
   - A chapter heading line, e.g. "Chapter {chapter_number}: <Title from outline>"
   - Directly below the heading, include the quote from the outline in this exact format:

       “Exact quote text.”
       — Name of Person

   - A short subheading line, e.g. "How leaders respond when the ground shifts"
   - Body text that aligns with the book topic, purpose, target audience, and the outline description.
5. The quote you use MUST:
   - Be exactly the one that the outline assigned to chapter {chapter_number}.
   - Not be reused from any other chapter in the outline.
   - Not be replaced by a different quote unless absolutely necessary.
6. The body should be written as conventional non-fiction prose, with paragraphs and no bullet points.
7. Do NOT include any tool calls, code, or JSON in your response.
8. Do NOT mention models, tools, Gemini, Vertex AI, or artificial intelligence.
9. Do NOT apologise or describe the writing process. Just write the chapter.

Style and tone:
- UK English spelling.
- Clear, confident, practical, and aimed at the specified target audience.
- Ensure the chapter clearly connects back to the quote and shows why it matters for the reader.

"""


def build_chapter_agent(chapter_number: int) -> LlmAgent:
    """
    Build one chapter-writing LlmAgent, for the given chapter number.

    The agent's output is stored in the shared state at:
    state[f"chapter_{chapter_number}_text"]
    via the output_key parameter.
    """
    return LlmAgent(
        name=f"chapter_{chapter_number}_agent",
        model=GEMINI_MODEL,
        description=f"Writes chapter {chapter_number} content.",
        instruction=_chapter_instruction(chapter_number),
        output_key=f"chapter_{chapter_number}_text",

        # No tools in this first test version.
        # We let the agent reply with the full chapter directly.
    )


def build_parallel_book_agent(max_chapters: int) -> ParallelAgent:

    """
    Build the root ParallelAgent that runs up to `max_chapters` chapter agents
    in parallel.
    """
    num_chapters = max(1, max_chapters)

    chapter_agents: List[LlmAgent] = [
        build_chapter_agent(i) for i in range(1, num_chapters + 1)
    ]

    parallel_agent = ParallelAgent(
        name="p_book_gen_parallel",
        description="Runs chapter-writing agents in parallel.",
        sub_agents=chapter_agents,
    )
    return parallel_agent


# ---------------------------------------------------------------------
# Merge agent (runs after parallel step)
# ---------------------------------------------------------------------


def _merge_instruction(num_chapters: int) -> str:
    """
    Instruction for the merge agent.

    It reads chapter texts from state keys:
      chapter_1_text, chapter_2_text, ..., chapter_N_text
    and assembles a single Kindle-style manuscript in Markdown.

    The original JSON spec is available in the conversation history,
    so it can respect book_topic, author_name, author_voice_style,
    target_audience, and book_purpose.
    """
    # Build a small description of available keys, for clarity in the prompt.
    chapter_keys_desc = ", ".join(
        f"chapter_{i}_text" for i in range(1, num_chapters + 1)
    )

    return f"""
You are a book assembler agent.

Context:
- The user provided ONE JSON object as their first message with fields such as:
  book_topic, author_name, author_bio, author_voice_style, target_audience, book_purpose, min_chapters.
- An outline planning agent has already created an outline for all chapters.
- A front-matter agent has already created Dedication, Foreword, Introduction, and About the Author
  and stored that text in shared state under 'front_matter'.
- Multiple chapter-writing agents have already run and stored their outputs in shared state.

You can access the chapter texts via the following state keys:
- {chapter_keys_desc}

You can also access the front matter via:
- front_matter

Use templated variables to access chapter content, e.g.:
- {{chapter_1_text}}
- {{chapter_2_text}}
- {{chapter_3_text}}  (and so on, up to chapter_{num_chapters}_text)

Your task:

1. Read the original JSON spec from the conversation history.
2. Read the front matter from state['front_matter'] if it exists.
3. Read all available chapter texts from the state keys above.
3. Assemble a single, coherent manuscript in Markdown that includes:
   - A title page:
     - Main title based on the book_topic.
     - Subtitle based on the book_purpose / target_audience.
     - Author name from the JSON.
- The front matter, largely as written:
     - Dedication
     - Foreword
     - Introduction
     - About the Author
     If front_matter is present, preserve its headings and structure. You may make very light edits
     for continuity, but do not substantially rewrite or reorder its content.
   - All chapters in order, using the existing chapter texts as the base.
     Keep the chapter headings and subheadings, and lightly adjust transitions if needed.
4. Do not radically rewrite chapters. Improve flow and transitions if necessary,
   but preserve the core content and intent.

Output format rules:

- Output MUST be a single Markdown document, ready for Kindle ingestion.
- Do NOT include any JSON, code, or commentary.
- Do NOT mention tools, models, Gemini, Vertex AI, or artificial intelligence.
- Use UK English spelling consistently.
- Avoid anything that suggests the manuscript was generated by a machine.
"""


def build_merge_agent(num_chapters: int) -> LlmAgent:
    """
    Build the LlmAgent that merges the chapters into a single manuscript.

    The final manuscript is stored in state under 'book_manuscript'.
    """
    return LlmAgent(
        name="merge_book_agent",
        model=GEMINI_MODEL,
        description="Merges chapter texts into a single Kindle-style manuscript.",
        instruction=_merge_instruction(num_chapters),
        output_key="book_manuscript",
    )
# ---------------------------------------------------------------------
# Save chapters agent (runs after parallel chapters)
# ---------------------------------------------------------------------


class SaveChaptersAgent(BaseAgent):
    """
    Deterministic agent that saves each chapter in state['chapter_X_text']
    to Google Cloud Storage using the BOOK_GEN_BUCKET environment variable.

    It sets state['chapter_gs_uris'] to a dict mapping chapter numbers to GCS URIs,
    and returns a brief summary message listing the URIs.
    """

    num_chapters: int   

    def __init__(self, num_chapters: int, *, name: str = "save_chapters_agent") -> None:
        super().__init__(
            name=name,
            description="Saves each chapter to GCS as a separate Markdown file.",
            num_chapters = num_chapters
        )
        self._storage_client = None  # lazy init

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text=(
                            "BOOK_GEN_BUCKET environment variable is not set. "
                            "Cannot save chapter files to Google Cloud Storage."
                        )
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        client = self._get_client()
        bucket = client.bucket(bucket_name)

        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")

        chapter_uris: dict[int, str] = {}
        lines: list[str] = []

        for i in range(1, self.num_chapters + 1):
            key = f"chapter_{i}_text"
            chapter_text = state.get(key)
            if not chapter_text:
                continue

            object_name = (
                f"books/run-{session_id}/chapters/"
                f"{timestamp}-chapter-{i:02d}.md"
            )

            blob = bucket.blob(object_name)
            blob.upload_from_string(
                chapter_text,
                content_type="text/markdown; charset=utf-8",
            )

            gs_uri = f"gs://{bucket_name}/{object_name}"
            chapter_uris[i] = gs_uri
            lines.append(f"Chapter {i}: {gs_uri}")

        # Store mapping in state for downstream agents or external tooling.
        state["chapter_gs_uris"] = chapter_uris

        if not lines:
            msg = "No chapter texts were found in state; no chapter files were saved."
        else:
            msg = "Saved chapter files to GCS:\n" + "\n".join(lines)

        content = genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=msg)],
        )
        yield Event(author=self.name, content=content)


# ---------------------------------------------------------------------
# Save agent (runs after merge, writes manuscript to GCS)
# ---------------------------------------------------------------------


class SaveManuscriptAgent(BaseAgent):
    """
    Deterministic agent that saves the final manuscript in state['book_manuscript']
    to Google Cloud Storage using the BOOK_GEN_BUCKET environment variable.

    It also sets state['book_gs_uri'] and returns a message with the GCS URI
    plus the full manuscript, so you still see the content in ADK Web.
    """

    def __init__(self, *, name: str = "save_manuscript_agent") -> None:
        super().__init__(
            name=name,
            description="Saves the final manuscript to GCS.",
        )
        self._storage_client = None  # lazy init

    def _get_client(self) -> storage.Client:
        if self._storage_client is None:
            self._storage_client = storage.Client()
        return self._storage_client

    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        state = ctx.session.state

        manuscript = state.get("book_manuscript")
        if not manuscript:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text="No manuscript found in state['book_manuscript']. Nothing to save."
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        bucket_name = os.environ.get("BOOK_GEN_BUCKET")
        if not bucket_name:
            content = genai_types.Content(
                role="system",
                parts=[
                    genai_types.Part(
                        text=("BOOK_GEN_BUCKET environment variable is not set. "
                        "Cannot save manuscript to Google Cloud Storage."
                    )
                    )
                ],
            )
            yield Event(author=self.name, content=content)
            return

        client = self._get_client()
        bucket = client.bucket(bucket_name)

        session_id = ctx.session.id
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        random_id = uuid.uuid4().hex[:8]

        object_name = (
            f"books/run-{session_id}/manuscripts/"
            f"{timestamp}-{random_id}.md"
        )

        blob = bucket.blob(object_name)
        blob.upload_from_string(
            manuscript,
            content_type="text/markdown; charset=utf-8",
        )

        gs_uri = f"gs://{bucket_name}/{object_name}"
        state["book_gs_uri"] = gs_uri

        # Return a message that includes the URI and also the full manuscript,
        # so you can still read it directly in ADK Web.
        text = (
            f"Saved manuscript to {gs_uri}\n\n"
            "Below is the manuscript that was saved:\n\n"
            f"{manuscript}"
        )

        content = genai_types.Content(
            role="system",
            parts=[genai_types.Part(text=text)],
        )
        yield Event(author=self.name, content=content)


# ---------------------------------------------------------------------
# Full workflow: SequentialAgent(root) = Parallel chapters → Merge
# ---------------------------------------------------------------------


def build_full_workflow_agent(max_chapters: int = 20) -> SequentialAgent:

    """
    Build the root SequentialAgent for this step.

    - Uses max_chapters as N, the maximum number of chapter agents.
    - Sub-agent 1: Outline agent for N chapters.
    - Sub-agent 2: Front-matter agent.
    - Sub-agent 3: ParallelAgent with N chapter writers.
    - Sub-agent 4: SaveChaptersAgent saving up to N chapters to GCS.
    - Sub-agent 5: Merge agent that merges those N chapter texts into one manuscript.
    - Sub-agent 6: SaveManuscriptAgent saving the full book to GCS.
    """

    num_chapters = max(1, max_chapters)
    outline_agent = build_outline_agent(num_chapters)
    front_matter_agent = build_front_matter_agent()
    parallel_agent = build_parallel_book_agent(num_chapters)
    save_chapters_agent = SaveChaptersAgent(num_chapters=num_chapters)

    merge_agent = build_merge_agent(num_chapters)

    workflow = SequentialAgent(
        name="p_book_gen_workflow",
        description=(
            "Deterministic workflow: outline planning, front-matter generation, "
            "parallel chapter generation, save chapters to GCS, merge into a single "
            "manuscript, then save the full book to GCS."

        ),
        sub_agents=[outline_agent, 
                    front_matter_agent, 
                    parallel_agent, 
                    save_chapters_agent,
                    merge_agent, 
                    SaveManuscriptAgent()],
    )
    return workflow