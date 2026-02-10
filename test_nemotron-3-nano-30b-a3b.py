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

    path = llm.replace('/', '-')

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
        'Chỉ đưa ra chữ cái đứng trước câu trả lời đúng (A, B, C, D hoặc E) của câu hỏi trắc nghiệm sau: '

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
    parser = argparse.ArgumentParser(description="VMLU Benchmark with Nemotron via NVIDIA API")

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
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max output tokens (default: 128)")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel API workers (default: 16)")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
