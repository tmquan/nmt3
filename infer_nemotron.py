#!/usr/bin/env python3
"""
infer_nemotron.py — Pedagogical Nemotron inference on the VMLU dataset.

A clean, configurable script for running any Nemotron model against the
Vietnamese Multitask Language Understanding (VMLU) benchmark via the
NVIDIA NIM API.

Everything is driven by a single ``InferenceConfig`` object that you edit
at the bottom of the file (or override from the CLI).  Results are captured
as Pydantic models and saved to both CSV and JSON.

Quick start
-----------
    export NVIDIA_API_KEY="nvapi-..."
    python infer_nemotron.py                        # defaults
    python infer_nemotron.py --reasoning             # chain-of-thought ON
    python infer_nemotron.py --workers 32 --limit 50 # fast test run
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════════════════
# 1.  PYDANTIC MODELS — typed configuration and results
# ═══════════════════════════════════════════════════════════════════════════


class ReasoningMode(str, Enum):
    """Whether chain-of-thought reasoning is enabled."""
    OFF = "off"
    ON = "on"


class InferenceConfig(BaseModel):
    """
    Everything you need to configure a run, in one place.

    Edit the defaults here or override via CLI flags.
    """

    # ── Model & endpoint ──────────────────────────────────────────────────
    model: str = Field(
        default="nvidia/nemotron-3-nano-30b-a3b",
        description="HuggingFace-style model ID served by the NIM endpoint.",
    )
    base_url: str = Field(
        default="https://integrate.api.nvidia.com/v1",
        description="OpenAI-compatible API base URL.",
    )
    api_key: str = Field(
        default="",
        description="NVIDIA API key.  Falls back to $NVIDIA_API_KEY env var.",
    )

    # ── Prompt engineering ────────────────────────────────────────────────
    system_prompt: str = Field(
        default=(
            # "You are a Vietnamese educational assessment expert. "
            # "You answer multiple-choice questions accurately and concisely."
            """
- Situation
The assistant is operating in a multilingual educational assessment environment where Vietnamese multiple-choice questions require cross-linguistic reasoning to ensure accuracy. The context involves standardized testing or academic evaluation where precision and cultural awareness are critical.

- Task
The assistant should answer Vietnamese multiple-choice questions by executing a systematic bilingual reasoning process that validates conclusions across both Vietnamese and English analytical frameworks, then output only the single letter corresponding to the correct answer.

- Objective
Maximize answer accuracy by leveraging both English-language logical reasoning and Vietnamese cultural/contextual expertise, ensuring that answers reflect both universal logic and Vietnam-specific knowledge where applicable.

Knowledge
Vietnamese multiple-choice questions may contain:
- Cultural references specific to Vietnam
- Local conventions in mathematics, science, or social contexts
- Idiomatic expressions that don't translate literally
- Technical terminology that requires precise understanding
- Numerical values and measurements that must be preserved exactly during translation

The bilingual reasoning approach mitigates risks of:
- Translation errors that distort meaning
- Missing culturally-specific context
- Overlooking Vietnam-specific conventions
- Misinterpreting technical terms

Core Instructions
The assistant should execute the following five-step process for each question:
- English Translation Phase: Translate the Vietnamese question into clear, natural English while preserving all numerical values, technical terms, proper nouns, and contextual nuances exactly as presented.
- English Reasoning Phase: Apply chain-of-thought reasoning in English to analyze the question systematically. The assistant should evaluate each answer option, eliminate demonstrably incorrect choices, apply relevant domain knowledge, and identify the most logically defensible answer.
- Vietnamese Re-examination Phase: Translate the reasoning back to Vietnamese and re-analyze the question from the perspective of a Vietnam-based subject matter expert. The assistant should consider whether Vietnamese cultural context, local conventions, regional knowledge, or language-specific interpretations affect the answer.
- Consolidation Phase: Compare conclusions from both reasoning attempts. When both analyses converge on the same answer, confidence is high. When analyses diverge, the assistant should identify the source of disagreement and determine which reasoning path is more reliable based on the question type and context.
- Output Phase: After completing both analyses, output exactly one character (A, B, C, D, or E) with no additional text, explanations, punctuation, or formatting.

Decision Rules

- When facing uncertainty between two options, the assistant should default to the answer derived from English reasoning.
- When the question proves exceptionally difficult to answer with confidence after completing both reasoning phases, the assistant should output B.

Output Format
- The final response must be exactly one character from the set {A, B, C, D, E}. The assistant should never include explanations, reasoning, multiple letters, or any additional text in the final output
            """
        ),
        description="System-level instruction prepended to every request.",
    )
    preamble: str = Field(
        default=(
            "Chỉ đưa ra chữ cái đứng trước câu trả lời đúng "
            "(A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau:"
        ),
        description="User-visible instruction placed before the question.",
    )

    # ── Reasoning / generation ────────────────────────────────────────────
    reasoning: ReasoningMode = Field(
        default=ReasoningMode.OFF,
        description="'on' enables chain-of-thought via reasoning_budget.",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.  0 = greedy / deterministic.",
    )
    max_tokens: int = Field(
        default=262144,
        gt=0,
        description="Maximum tokens in the model response.",
    )

    # ── Data & I/O ────────────────────────────────────────────────────────
    input_file: str = Field(
        default="test.jsonl",
        description="Path to the VMLU JSONL file.",
    )
    output_dir: str = Field(
        default="results",
        description="Directory for all output artefacts.",
    )
    limit: Optional[int] = Field(
        default=None,
        description="Process only the first N records (for debugging).",
    )

    # ── Parallelism ───────────────────────────────────────────────────────
    workers: int = Field(
        default=16,
        gt=0,
        description="Number of concurrent API request threads.",
    )
    max_retries: int = Field(
        default=5,
        gt=0,
        description="Retry count for transient API failures.",
    )
    retry_delay: float = Field(
        default=10.0,
        ge=0.0,
        description="Seconds to wait between retries.",
    )

    def resolve_api_key(self) -> str:
        """Return the API key, falling back to the environment variable."""
        return self.api_key or os.environ.get("NVIDIA_API_KEY", "")


class VMLUQuestion(BaseModel):
    """One record from the VMLU test.jsonl file."""
    id: str
    question: str
    choices: list[str]


class VMLUResult(BaseModel):
    """The model's response for a single question."""
    id: str
    question: str
    choices: list[str]
    prompt_sent: str
    raw_answer: str = Field(description="Full text returned by the model.")
    reasoning: str = Field(default="", description="Chain-of-thought text (if reasoning=on).")
    parsed_answer: str = Field(description="Single letter a-e extracted from raw_answer.")


class RunSummary(BaseModel):
    """Aggregate statistics for the entire run."""
    model: str
    reasoning: str
    temperature: float
    max_tokens: int
    total_questions: int
    answered: int
    unanswered: int
    accuracy_proxy: float = Field(description="Fraction of questions that got a valid a-e letter.")
    elapsed_seconds: float
    throughput_qps: float = Field(description="Questions per second.")
    output_csv: str
    output_json: str


# ═══════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════


def load_questions(path: str, limit: Optional[int] = None) -> list[VMLUQuestion]:
    """
    Read the VMLU JSONL file and return typed question objects.

    >>> qs = load_questions("test.jsonl", limit=5)
    >>> print(qs[0].question)
    """
    questions: list[VMLUQuestion] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            questions.append(VMLUQuestion(**raw))
            if limit and len(questions) >= limit:
                break
    return questions


# ═══════════════════════════════════════════════════════════════════════════
# 3.  PROMPT BUILDING
# ═══════════════════════════════════════════════════════════════════════════


def build_prompt(question: VMLUQuestion, preamble: str) -> str:
    """
    Format a VMLU question into the user-message string.

    Structure:
        {preamble}
        {question text}

         A. ...
         B. ...
         C. ...
         D. ...
        Đáp án:
    """
    choices_block = "\n".join(f" {c}" for c in question.choices)
    return f"{preamble}\n\n{question.question}\n\n{choices_block}\nĐáp án:"


def build_messages(
    prompt: str,
    system_prompt: str,
) -> list[dict[str, str]]:
    """
    Assemble the chat-completion messages list.

    If ``system_prompt`` is non-empty it is sent as the system role,
    followed by the user prompt.
    """
    messages: list[dict[str, str]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


# ═══════════════════════════════════════════════════════════════════════════
# 4.  API CALL  (single question)
# ═══════════════════════════════════════════════════════════════════════════


def call_model(
    client: OpenAI,
    messages: list[dict[str, str]],
    cfg: InferenceConfig,
) -> tuple[str, str]:
    """
    Send one request to the Nemotron API with streaming.

    Returns
    -------
    (answer_text, reasoning_text)
        answer_text  : the visible completion content.
        reasoning_text : chain-of-thought content (empty when reasoning=off).
    """
    extra_body: dict = {}
    if cfg.reasoning == ReasoningMode.ON:
        extra_body = {
            "reasoning_budget": cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    for attempt in range(1, cfg.max_retries + 1):
        try:
            stream = client.chat.completions.create(
                model=cfg.model,
                messages=messages,
                temperature=cfg.temperature,
                top_p=1,
                max_tokens=cfg.max_tokens,
                stream=True,
                **({"extra_body": extra_body} if extra_body else {}),
            )

            answer_parts: list[str] = []
            reasoning_parts: list[str] = []

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # Chain-of-thought (reasoning_content is a NIM extension)
                cot = getattr(delta, "reasoning_content", None)
                if cot:
                    reasoning_parts.append(cot)

                # Normal answer content
                if delta.content is not None:
                    answer_parts.append(delta.content)

            return "".join(answer_parts), "".join(reasoning_parts)

        except Exception as exc:
            if attempt < cfg.max_retries:
                print(f"  [retry {attempt}/{cfg.max_retries}] {exc}")
                time.sleep(cfg.retry_delay)
            else:
                print(f"  [FAILED] {exc}")
                return "", ""

    return "", ""  # unreachable, but keeps mypy happy


# ═══════════════════════════════════════════════════════════════════════════
# 5.  ANSWER PARSING
# ═══════════════════════════════════════════════════════════════════════════


def parse_answer(raw: str) -> str:
    """
    Extract a single answer letter (a-e) from the model's raw output.

    Strategy (in priority order):
      1.  Last standalone A-E letter in the text (handles "The answer is D").
      2.  Last word's first character if it is a-e.
      3.  Empty string (unanswered).

    Returns lowercase a-e or "".
    """
    if not raw or not raw.strip():
        return ""

    # Find all standalone A-E letters; take the last one
    matches = re.findall(r"\b([A-Ea-e])\b", raw.strip())
    if matches:
        return matches[-1].lower()

    # Fallback: last token
    last_word = raw.strip().split()[-1]
    first_char = last_word[0].lower()
    return first_char if first_char in "abcde" else ""


# ═══════════════════════════════════════════════════════════════════════════
# 6.  PARALLEL RUNNER
# ═══════════════════════════════════════════════════════════════════════════


def run_inference(
    questions: list[VMLUQuestion],
    cfg: InferenceConfig,
) -> list[VMLUResult]:
    """
    Process all questions in parallel and return typed results (order preserved).
    """
    api_key = cfg.resolve_api_key()
    if not api_key:
        raise SystemExit(
            "ERROR: No API key.  Set NVIDIA_API_KEY or pass --api-key."
        )

    client = OpenAI(base_url=cfg.base_url, api_key=api_key)

    # Pre-allocate so results stay in order
    results: list[VMLUResult | None] = [None] * len(questions)
    lock = threading.Lock()
    done = 0

    def _worker(idx: int, q: VMLUQuestion) -> None:
        nonlocal done
        prompt = build_prompt(q, cfg.preamble)
        messages = build_messages(prompt, cfg.system_prompt)
        answer_text, reasoning_text = call_model(client, messages, cfg)

        results[idx] = VMLUResult(
            id=q.id,
            question=q.question,
            choices=q.choices,
            prompt_sent=prompt,
            raw_answer=answer_text,
            reasoning=reasoning_text,
            parsed_answer=parse_answer(answer_text),
        )

        with lock:
            done += 1
            if done % 200 == 0:
                print(f"  ... {done}/{len(questions)} done")

    with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
        futures = {
            pool.submit(_worker, i, q): i
            for i, q in enumerate(questions)
        }
        with tqdm(total=len(questions), desc="Inference", unit="q") as pbar:
            for fut in as_completed(futures):
                fut.result()  # propagate exceptions
                pbar.update(1)

    return results  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# 7.  SAVING RESULTS
# ═══════════════════════════════════════════════════════════════════════════


def save_results(
    results: list[VMLUResult],
    cfg: InferenceConfig,
    elapsed: float,
) -> RunSummary:
    """
    Write outputs and return a summary.

    Artefacts saved:
        {output_dir}/submission.csv      – id,answer (for leaderboard upload)
        {output_dir}/full_results.json   – every VMLUResult as JSON
        {output_dir}/run_summary.json    – RunSummary
    """
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── submission.csv ────────────────────────────────────────────────────
    csv_path = out / "submission.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("id,answer\n")
        for r in results:
            fh.write(f"{r.id},{r.parsed_answer}\n")

    # ── full_results.json ─────────────────────────────────────────────────
    json_path = out / "full_results.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            [r.model_dump() for r in results],
            fh,
            ensure_ascii=False,
            indent=2,
        )

    # ── summary ───────────────────────────────────────────────────────────
    answered = sum(1 for r in results if r.parsed_answer)
    summary = RunSummary(
        model=cfg.model,
        reasoning=cfg.reasoning.value,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        total_questions=len(results),
        answered=answered,
        unanswered=len(results) - answered,
        accuracy_proxy=answered / max(len(results), 1),
        elapsed_seconds=round(elapsed, 1),
        throughput_qps=round(len(results) / max(elapsed, 0.01), 2),
        output_csv=str(csv_path),
        output_json=str(json_path),
    )

    summary_path = out / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        fh.write(summary.model_dump_json(indent=2))

    return summary


# ═══════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ═══════════════════════════════════════════════════════════════════════════


def parse_cli() -> InferenceConfig:
    """Build an InferenceConfig from command-line arguments."""
    import argparse

    defaults = InferenceConfig()

    p = argparse.ArgumentParser(
        description="Run Nemotron inference on the VMLU benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model & endpoint
    p.add_argument("--model", default=defaults.model)
    p.add_argument("--base-url", default=defaults.base_url)
    p.add_argument("--api-key", default="")

    # Prompt engineering
    p.add_argument("--system-prompt", default=defaults.system_prompt,
                    help="System message (set to '' to disable).")
    p.add_argument("--preamble", default=defaults.preamble,
                    help="Instruction placed before each question.")

    # Generation
    p.add_argument("--reasoning", action="store_true",
                    help="Enable chain-of-thought reasoning.")
    p.add_argument("--temperature", type=float, default=defaults.temperature)
    p.add_argument("--max-tokens", type=int, default=defaults.max_tokens)

    # Data & I/O
    p.add_argument("--input", default=defaults.input_file)
    p.add_argument("--output-dir", default=defaults.output_dir)
    p.add_argument("--limit", type=int, default=None,
                    help="Only process the first N questions.")

    # Parallelism
    p.add_argument("--workers", type=int, default=defaults.workers)

    args = p.parse_args()

    return InferenceConfig(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        system_prompt=args.system_prompt,
        preamble=args.preamble,
        reasoning=ReasoningMode.ON if args.reasoning else ReasoningMode.OFF,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        input_file=args.input,
        output_dir=args.output_dir,
        limit=args.limit,
        workers=args.workers,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 9.  MAIN
# ═══════════════════════════════════════════════════════════════════════════


def main() -> None:
    cfg = parse_cli()

    # ── Print config ──────────────────────────────────────────────────────
    print("=" * 64)
    print("  Nemotron × VMLU Inference")
    print("=" * 64)
    print(f"  Model        : {cfg.model}")
    print(f"  Base URL     : {cfg.base_url}")
    print(f"  System prompt: {cfg.system_prompt[:60]}{'...' if len(cfg.system_prompt) > 60 else ''}")
    print(f"  Preamble     : {cfg.preamble[:60]}{'...' if len(cfg.preamble) > 60 else ''}")
    print(f"  Reasoning    : {cfg.reasoning.value}")
    print(f"  Temperature  : {cfg.temperature}")
    print(f"  Max tokens   : {cfg.max_tokens}")
    print(f"  Workers      : {cfg.workers}")
    print(f"  Input        : {cfg.input_file}")
    print(f"  Output dir   : {cfg.output_dir}")
    print(f"  Limit        : {cfg.limit or 'all'}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────
    questions = load_questions(cfg.input_file, cfg.limit)
    print(f"Loaded {len(questions)} questions from {cfg.input_file}")
    print()

    # ── Run inference ─────────────────────────────────────────────────────
    t0 = time.perf_counter()
    results = run_inference(questions, cfg)
    elapsed = time.perf_counter() - t0

    # ── Save & summarise ──────────────────────────────────────────────────
    summary = save_results(results, cfg, elapsed)

    print()
    print("=" * 64)
    print("  Run Summary")
    print("=" * 64)
    print(f"  Total questions : {summary.total_questions}")
    print(f"  Answered (a-e)  : {summary.answered}")
    print(f"  Unanswered      : {summary.unanswered}")
    print(f"  Answer rate     : {summary.accuracy_proxy:.1%}")
    print(f"  Elapsed         : {summary.elapsed_seconds:.1f}s")
    print(f"  Throughput      : {summary.throughput_qps:.1f} q/s")
    print(f"  Submission CSV  : {summary.output_csv}")
    print(f"  Full results    : {summary.output_json}")
    print("=" * 64)


if __name__ == "__main__":
    main()
