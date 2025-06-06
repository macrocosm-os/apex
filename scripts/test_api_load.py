"""
Example usage:
python scripts/test_api_load.py --levels 4 --out stress_test --key API_KEY

Results will be stored in stress_test/

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
import plotly.express as px
from httpx import HTTPStatusError, Timeout
from nltk.corpus import words

nltk.download("words")
word_list = words.words()


def approximate_tokens(text: str) -> int:
    """Approximate the number of tokens from a given string.

    Approach:
      - Split by whitespace to get a word count.
      - Multiply by ~1.3 to guess how GPT might break words into tokens.
      - This is a naive heuristic. For accuracy, consider tiktoken.
    """
    words = text.split()
    return int(len(words) * 1.3)


def get_color_for_code(code: int) -> str:
    """Return a color string for a given status code.

    - 200 -> green
    - 4xx -> crimson
    - 5xx -> darkred
    - 0  -> firebrick (means unknown error in this script)
    - else -> red
    """
    if code == 200:
        return "green"
    elif 400 <= code < 500:
        return "crimson"
    elif 500 <= code < 600:
        return "darkred"
    elif code == 0:
        return "firebrick"
    else:
        return "red"


async def make_completion(
    client: openai.AsyncOpenAI,
    prompt: str,
    stream: bool = True,
    seed: str = "1759348",
) -> dict:
    """Make a completion request to the API.

    Measures:
      - Time to first token
      - Time to full response
      - Approximate tokens received
      - Status code

    Returns a dictionary with measurements.
    """
    start_time = time.perf_counter()
    time_to_first_token = None
    total_approx_tokens = 0
    status_code = None

    try:
        result = await client.chat.completions.create(
            model=None,  # or your preferred model if required
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
            # If not streaming, we treat the entire response as one chunk
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


async def run_stress_test(api_key: str, output_dir: str, levels: int = 10, url: str = "http://0.0.0.0:8005/v1"):
    """Run a stress test by sending exponentially increasing amount of concurrent requests till `2**levels`.

    Measures:
      - Time to first token
      - Total response latency
      - Approx. tokens/second
      - Success/failure counts & status codes
      - Empty response counts
    Saves to stress_test.csv and generates five Plotly charts.
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
    concurrency_levels = [2**i for i in range(0, levels)]

    results = []
    for concurrency in concurrency_levels:
        print(f"\n=== Testing concurrency: {concurrency} ===")

        tasks = [make_completion(client, prompt=prompt, stream=True) for _ in range(concurrency)]
        responses = await asyncio.gather(*tasks)

        success_count = sum(r["success"] for r in responses)
        failure_count = concurrency - success_count

        status_code_counts = {}
        empty_response_count = 0  # Initialize counter for empty responses
        for r in responses:
            code = r["status_code"]
            status_code_counts[code] = status_code_counts.get(code, 0) + 1
            # Count as empty response if successful but no tokens received
            if r["success"] and r["total_approx_tokens"] == 0:
                empty_response_count += 1

        successful_responses = [r for r in responses if r["success"]]
        if successful_responses:
            avg_time_to_first = sum(r["time_to_first_token"] for r in successful_responses) / len(successful_responses)
            avg_total_latency = sum(r["total_latency"] for r in successful_responses) / len(successful_responses)
            avg_tokens_per_sec = sum(r["tokens_per_second"] for r in successful_responses) / len(successful_responses)
        else:
            avg_time_to_first = None
            avg_total_latency = None
            avg_tokens_per_sec = None

        row = {
            "concurrency": concurrency,
            "avg_time_to_first_token": avg_time_to_first,
            "avg_total_latency": avg_total_latency,
            "avg_tokens_per_second": avg_tokens_per_sec,
            "success_count": success_count,
            "failure_count": failure_count,
            "empty_response_count": empty_response_count,
            "status_code_counts": status_code_counts,
        }
        results.append(row)

    # Save results to CSV.
    all_status_codes = set()
    for row in results:
        all_status_codes.update(row["status_code_counts"].keys())
    all_status_codes = sorted(list(all_status_codes))

    csv_rows = []
    for row in results:
        base_data = {
            "concurrency": row["concurrency"],
            "avg_time_to_first_token": row["avg_time_to_first_token"] if row["avg_time_to_first_token"] else "",
            "avg_total_latency": row["avg_total_latency"] if row["avg_total_latency"] else "",
            "avg_tokens_per_second": row["avg_tokens_per_second"] if row["avg_tokens_per_second"] else "",
            "success_count": row["success_count"],
            "failure_count": row["failure_count"],
            "empty_response_count": row["empty_response_count"],
        }
        for code in all_status_codes:
            base_data[f"status_{code}"] = row["status_code_counts"].get(code, 0)
        csv_rows.append(base_data)

    field_names = list(csv_rows[0].keys())

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True, parents=True)

    with open(output_dir / "stress_test.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(csv_rows)

    print("\nSaved stress_test.csv.")

    # Generate and save charts.
    px.defaults.template = "plotly_dark"
    df = pd.DataFrame(csv_rows)

    # 1. Tokens/Second vs. Concurrency.
    fig1 = px.line(
        df,
        x="concurrency",
        y="avg_tokens_per_second",
        title="Tokens/Second vs. Concurrency (Approx.)",
        markers=True,
    )
    fig1.update_xaxes(title_text="Concurrent Queries")
    fig1.update_yaxes(title_text="Avg. Tokens/Second (Approx.)")
    fig1.write_image(output_dir / "stress_test_tokens_per_second.png")
    fig1.show()

    # 2. First-Token Latency vs. Concurrency.
    fig2 = px.line(
        df,
        x="concurrency",
        y="avg_time_to_first_token",
        title="First-Token Latency vs. Concurrency",
        markers=True,
    )
    fig2.update_xaxes(title_text="Concurrent Queries")
    fig2.update_yaxes(title_text="Avg. Latency to First Token (s)")
    fig2.write_image(output_dir / "stress_test_first_token_latency.png")
    fig2.show()

    # 3. Full Response Latency vs. Concurrency.
    fig3 = px.line(
        df,
        x="concurrency",
        y="avg_total_latency",
        title="Full Response Latency vs. Concurrency",
        markers=True,
    )
    fig3.update_xaxes(title_text="Concurrent Queries")
    fig3.update_yaxes(title_text="Avg. Total Latency (s)")
    fig3.write_image(output_dir / "stress_test_full_response_latency.png")
    fig3.show()

    # 4. Status Code Counts vs. Concurrency.
    # Prepare a long-form DataFrame for plotting
    status_data = []
    for _, row in df.iterrows():
        for code in all_status_codes:
            status_data.append(
                {
                    "concurrency": row["concurrency"],
                    "status_code": code,
                    "count": row[f"status_{code}"],
                }
            )
    status_df = pd.DataFrame(status_data)

    # Build a color map: 200 = green, others = red-like.
    color_map = {}
    for code in all_status_codes:
        color_map[str(code)] = get_color_for_code(code)

    fig4 = px.line(
        status_df,
        x="concurrency",
        y="count",
        color="status_code",
        title="Response Count by Status Code vs. Concurrency",
        markers=True,
        color_discrete_map=color_map,  # use our custom color map
    )
    fig4.update_xaxes(title_text="Concurrent Queries")
    fig4.update_yaxes(title_text="Count of Responses")
    fig4.write_image(output_dir / "stress_test_status_codes.png")
    fig4.show()

    # 5. Empty Responses vs. Concurrency.
    fig5 = px.line(
        df,
        x="concurrency",
        y="empty_response_count",
        title="Empty Responses vs. Concurrency",
        markers=True,
        line_shape="linear",
    )
    fig5.update_xaxes(title_text="Concurrent Queries")
    fig5.update_yaxes(title_text="Count of Empty Responses")
    fig5.write_image(output_dir / "stress_test_empty_responses.png")
    fig5.show()

    print("All plots saved to .png files and displayed.")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run a stress test against the specified API endpoint.")

    parser.add_argument(
        "--key",
        type=str,
        # Specify your API key, current is left here just for local testings.
        default="0566dbe21ee33bba9419549716cd6f1f",
        help="API key for authentication.",
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

    parser.add_argument("--levels", type=int, default=10, help="Number of stress test levels to execute (default: 10).")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    asyncio.run(
        run_stress_test(
            api_key=args.key,
            url=args.url,
            output_dir=args.out,
            levels=args.levels,
        )
    )
