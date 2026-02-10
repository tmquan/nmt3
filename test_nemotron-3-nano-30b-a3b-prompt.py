import pandas as pd
import numpy as np
import json
import logging
import os
import argparse
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from string import Template
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

import warnings
warnings.simplefilter("ignore")

DEFAULT_MODEL = "nvidia/nemotron-3-nano-30b-a3b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def call_nemotron(
    client: OpenAI,
    prompt: str,
    model: str,
    enable_thinking: bool = False,
    temperature: float = 0.0,
    max_tokens: int = 128,
    max_retries: int = 5,
    retry_delay: float = 10.0,
) -> str:
    """
    Call the Nemotron model via NVIDIA API with streaming.
    Returns the full answer text.
    """
    messages = [{"role": "user", "content": prompt}]

    extra_body = {}
    if enable_thinking:
        extra_body = {
            "reasoning_budget": max_tokens,
            "chat_template_kwargs": {"enable_thinking": True},
        }

    for attempt in range(1, max_retries + 1):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                top_p=1,
                max_tokens=max_tokens,
                stream=True,
                **({"extra_body": extra_body} if extra_body else {}),
            )

            answer_parts: list[str] = []
            for chunk in completion:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content is not None:
                    answer_parts.append(delta.content)

            return "".join(answer_parts)

        except Exception as exc:
            print(f"  Attempt {attempt}/{max_retries} failed: {exc}")
            if attempt < max_retries:
                print(f"  Retrying in {retry_delay}s ...")
                time.sleep(retry_delay)
            else:
                print(f"  All retries exhausted.")
                return ""


def main(args):
    llm = args.llm
    folder = args.folder

    path = llm.replace('/', '-') + '-prompt'

    ## create directory
    directory_path = './logs'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    # Configure logging
    logging.basicConfig(filename=f"./logs/{path}.log", level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    logging.info(f'Model name: {llm}')

    # Setup API client
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        print("ERROR: Set NVIDIA_API_KEY environment variable.")
        print("  export NVIDIA_API_KEY='nvapi-...'")
        raise SystemExit(1)

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    # Create empty lists to store data
    ids = []
    questions = []
    choices_A = []
    choices_B = []
    choices_C = []
    choices_D = []
    choices_E = []

    # Read JSONL files
    data_path = Path(folder)
    jsonl_files = list(data_path.glob('test.jsonl'))

    for file in jsonl_files:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                ids.append(data["id"])
                questions.append(data["question"])
                choices = data["choices"]
                try:
                    choices_A.append(choices[0])
                except:
                    choices_A.append('')
                try:
                    choices_B.append(choices[1])
                except:
                    choices_B.append('')
                try:
                    choices_C.append(choices[2])
                except:
                    choices_C.append('')
                try:
                    choices_D.append(choices[3])
                except:
                    choices_D.append('')
                try:
                    choices_E.append(choices[4])
                except:
                    choices_E.append('')

    # Create a DataFrame
    df = pd.DataFrame({
        "id": ids,
        "prompt": questions,
        "A": choices_A,
        "B": choices_B,
        "C": choices_C,
        "D": choices_D,
        "E": choices_E
    })
    logging.info(df.head())

    preamble = \
"""- Situation
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
- The final response must be exactly one character from the set {A, B, C, D, E}. The assistant should never include explanations, reasoning, multiple letters, or any additional text in the final output."""

    template = Template('$preamble\n\n$prompt\n\n $a\n $b\n $c\n $d\n $e\nĐáp án:')

    def format_input(df, idx):
        prompt = df.loc[idx, 'prompt']
        a = df.loc[idx, 'A']
        b = df.loc[idx, 'B']
        c = df.loc[idx, 'C']
        d = df.loc[idx, 'D']
        e = df.loc[idx, 'E']

        input_text = template.substitute(
            preamble=preamble, prompt=prompt, a=a, b=b, c=c, d=d, e=e)

        return input_text

    # Test a toy example
    toy_input = format_input(df, 0)
    toy_answer = call_nemotron(
        client=client, prompt=toy_input, model=llm,
        enable_thinking=args.enable_thinking,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )
    logging.info('Contruct a toy eg')
    logging.info("Generated answer: %s", toy_answer)

    print(f"Model       : {llm}")
    print(f"Thinking    : {'ON' if args.enable_thinking else 'OFF'}")
    print(f"Workers     : {args.workers}")
    print(f"Records     : {len(df)}")
    print()

    # Pre-allocate answers list (indexed) so ordering is preserved
    answers = [None] * len(df)
    checkpoint_lock = threading.Lock()
    completed_count = 0

    def _process_one(idx: int) -> None:
        nonlocal completed_count
        input_text = format_input(df, idx)
        answer_decoded = call_nemotron(
            client=client, prompt=input_text, model=llm,
            enable_thinking=args.enable_thinking,
            temperature=args.temperature, max_tokens=args.max_tokens,
        )

        last_element = answer_decoded
        answer = last_element.split()[-1] if last_element.split() else ""
        answers[idx] = answer

        with checkpoint_lock:
            completed_count += 1
            if completed_count % 100 == 0:
                logging.info(f"Completed {completed_count}/{len(df)}")

    start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_one, idx): idx
            for idx in df.index
        }
        with tqdm(total=len(df), desc="Inference") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    end = time.time()
    duration = end - start
    print('Time taken for running inference: ', duration)

    df['answer'] = answers
    logging.info(df.head())

    # save the answer csv
    df[['id', 'answer']].to_csv(f"./logs/{path}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VMLU Benchmark with Nemotron via NVIDIA API (with bilingual prompt)")

    # Add command-line arguments
    parser.add_argument("--llm", type=str, default=DEFAULT_MODEL,
                        help=f"Specify the model (default: {DEFAULT_MODEL})")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL,
                        help=f"API base URL (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--folder", type=str, default="./",
                        help="Specify the folder data (default: ./)")
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Enable chain-of-thought reasoning")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature (default: 0.0)")
    parser.add_argument("--max-tokens", type=int, default=262144,
                        help="Max output tokens (default: 262144)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel API workers (default: 16)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
