"""
Example usage:
python scripts/test_api_empty_responses.py --concurrent 4 --queries 1000 --out test

Results will be stored in test/

Additional requirements:
pip install plotly kaleido
"""

import argparse
import asyncio
import csv
import random
import sys
import time
from pathlib import Path

import nltk
import openai
import pandas as pd
import plotly.graph_objects as go
from httpx import HTTPStatusError, Timeout
from nltk.corpus import words

nltk.download("words")
word_list = words.words()


def approximate_tokens(text: str) -> int:
    """Approximate the number of tokens from a given string.

    Approach:
      - Split by whitespace to get a word count.
      - Multiply by ~1.3 to get approximate amount of tokens.
    """
    words_in_text = text.split()
    return int(len(words_in_text) * 1.3)


async def make_completion(
    client: openai.AsyncOpenAI,
    prompt: str,
    stream: bool = True,
    seed: str = "1759348",
) -> dict:
    """Make a completion request to the API.

    Measures:
      - Time to first token.
      - Time to full response.
      - Approximate tokens received.
      - Status code.

    Returns a dictionary with measurements.
    """
    start_time = time.perf_counter()
    time_to_first_token = None
    total_approx_tokens = 0
    status_code = None

    try:
        result = await client.chat.completions.create(
            model=None,
            messages=[{"role": "user", "content": prompt}],
            stream=stream,
            extra_body={
                "seed": seed,
                "sampling_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 50,
                    "max_new_tokens": 256,
                    "do_sample": True,
                    "seed": None,
                },
                "task": "InferenceTask",
                "mixture": False,
            },
        )

        if not stream:
            # TODO: Non-streaming part is not tested.
            raise NotImplementedError("Implement non-streaming mode")
            text = result
            total_latency = time.perf_counter() - start_time
            total_approx_tokens = approximate_tokens(text)
            status_code = 200
        else:
            # If streaming, measure time-to-first-token and accumulate approximate tokens
            async for chunk in result:
                if hasattr(result, "response") and result.response is not None:
                    status_code = result.response.status_code

                delta_content = chunk.choices[0].delta.content
                if delta_content:
                    if time_to_first_token is None:
                        time_to_first_token = time.perf_counter() - start_time
                    total_approx_tokens += approximate_tokens(delta_content)

            total_latency = time.perf_counter() - start_time

        if time_to_first_token is None:
            time_to_first_token = total_latency

        if status_code is None:
            status_code = 200

        if total_latency > 0:
            tokens_per_second = total_approx_tokens / total_latency
        else:
            tokens_per_second = 0.0

        return {
            "time_to_first_token": time_to_first_token,
            "total_latency": total_latency,
            "tokens_per_second": tokens_per_second,
            "status_code": status_code,
            "success": True,
            "total_approx_tokens": total_approx_tokens,
        }

    except HTTPStatusError as e:
        return {
            "time_to_first_token": None,
            "total_latency": None,
            "tokens_per_second": 0,
            "status_code": e.response.status_code,
            "success": False,
            "total_approx_tokens": 0,
        }
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return {
            "time_to_first_token": None,
            "total_latency": None,
            "tokens_per_second": 0,
            "status_code": 0,
            "success": False,
            "total_approx_tokens": 0,
        }


async def run_stress_test(
    api_key: str, output_dir: str, concurrent: int = 4, queries: int = 1000, url: str = "http://0.0.0.0:8005/v1"
):
    """Run a stress test by sending concurrent API requests.

    Args:
        api_key (str): API key for authentication.
        output_dir (str): Directory to save outputs.
        concurrent (int): Number of concurrent workers.
        queries (int): Number of queries per worker.
        url (str): API endpoint URL.

    Measures:
        - Total successes and failures.
        - Success rate.
        - Cumulative failures over query execution time.
    Saves to result.csv and generates a Plotly fail rate chart.

    Additionally, prints the total number of empty or errored queries to the console.
    """
    client = openai.AsyncOpenAI(
        base_url=url,
        max_retries=0,
        timeout=Timeout(120, connect=120, read=120),
        api_key=api_key,
    )
    client._client.headers["api-key"] = api_key
    word = random.choice(word_list)
    prompt = f"Write a short story about {word}."

    total_queries = concurrent * queries
    success_count = 0
    fail_count = 0
    fail_list = []

    print(
        f"\nStarting stress test with {concurrent} concurrent workers, each performing {queries} queries (Total: {total_queries} queries)."
    )

    semaphore = asyncio.Semaphore(concurrent)

    async def worker(worker_id: int):
        nonlocal success_count, fail_count
        for i in range(queries):
            async with semaphore:
                if i % 10 == 0:
                    print(f"Worker {worker_id}: {i} / {queries}")
                response = await make_completion(client, prompt=prompt, stream=True)
                if response["success"] and response["total_approx_tokens"] > 0:
                    success_count += 1
                    fail_list.append(0)
                else:
                    fail_count += 1
                    fail_list.append(1)

    # Launch all workers
    workers = [asyncio.create_task(worker(w_id)) for w_id in range(concurrent)]
    await asyncio.gather(*workers)

    # Calculate success rate
    success_rate = (success_count / total_queries) * 100 if total_queries > 0 else 0.0

    # Save result.csv
    result_data = {
        "concurrent": concurrent,
        "queries": queries,
        "success": success_count,
        "fail": fail_count,
        "success_rate": round(success_rate, 2),
    }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / "result.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["concurrent", "queries", "success", "fail", "success_rate"])
        writer.writeheader()
        writer.writerow(result_data)

    print("\nSaved result.csv with the following data:")
    print(result_data)

    print(f"Total number of empty or errored queries: {fail_count}")

    # Create DataFrame for plotting
    fail_df = pd.DataFrame({"fail": fail_list})
    fail_df["cumulative_fail"] = fail_df["fail"].cumsum()
    fail_df["query_number"] = fail_df.index + 1

    # Generate the fail rate chart using Plotly with dark background
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fail_df["query_number"],
            y=fail_df["cumulative_fail"],
            mode="lines",
            name="Cumulative Failed/Empty Queries",
            line=dict(color="red"),
            hoverinfo="x+y",
        )
    )

    fig.update_layout(
        template="plotly_dark",
        title="Cumulative Failed/Empty Queries Over Time",
        xaxis_title="Query Number",
        yaxis_title="Number of Failed/Empty Queries",
        autosize=False,
        width=1200,
        height=600,
    )

    # Save the plot to the output directory
    plot_file = output_dir / "fail_rate_chart.png"
    try:
        fig.write_image(plot_file)
        print(f"Saved fail rate chart to {plot_file}")
    except Exception as e:
        print(f"Failed to save fail rate chart as image: {e}")

    # Optionally display the figure
    fig.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a stress test against the specified API endpoint.")

    parser.add_argument(
        "--key",
        type=str,
        # Specify your API key, current is left here just for local testings.
        default="0566dbe21ee33bba9419549716cd6f1f",
        help="API key for authentication (default: 0566dbe21ee33bba9419549716cd6f1f).",
    )

    parser.add_argument(
        "--url",
        type=str,
        default="http://0.0.0.0:8005/v1",
        help="URL of the API endpoint to test (default: http://0.0.0.0:8005/v1).",
    )

    parser.add_argument(
        "--out",
        type=str,
        default="stress_test",
        help="Output directory for storing test results (default: stress_test).",
    )

    parser.add_argument("--concurrent", type=int, default=4, help="Number of concurrent workers to query (default: 4).")

    parser.add_argument("--queries", type=int, default=1000, help="Number of queries per worker (default: 1000).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    asyncio.run(
        run_stress_test(
            api_key=args.key,
            url=args.url,
            output_dir=args.out,
            concurrent=args.concurrent,
            queries=args.queries,
        )
    )
