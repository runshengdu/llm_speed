import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
MODELS_PATH = ROOT / "models.yaml"
PROMPTS_PATH = ROOT / "prompts.json"
OUTPUT_DIR = ROOT / "output"
RETRY_COUNT = 3
ENV_REF_RE = re.compile(r"^\$\{([A-Za-z_][A-Za-z0-9_]*)\}$")


class BenchmarkError(Exception):
    """Expected benchmark failure for one model or prompt."""


def load_models(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise BenchmarkError(f"{path.name} must contain a top-level 'models' list")

    return models


def load_prompts(path: Path) -> List[Tuple[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict):
        prompts = list(data.items())
    elif isinstance(data, list):
        prompts = [(f"prompt_{index + 1}", prompt) for index, prompt in enumerate(data)]
    else:
        raise BenchmarkError(f"{path.name} must be a JSON object or array")

    invalid = [name for name, prompt in prompts if not isinstance(prompt, str) or not prompt]
    if invalid:
        raise BenchmarkError(f"{path.name} contains empty or non-string prompts: {', '.join(invalid)}")

    if not prompts:
        raise BenchmarkError(f"{path.name} does not contain any prompts")

    return prompts


def resolve_api_key(raw_value: Any, model_name: str) -> str:
    if not isinstance(raw_value, str) or not raw_value:
        raise BenchmarkError(f"{model_name}: api_key is missing or not a string")

    match = ENV_REF_RE.match(raw_value)
    if not match:
        return raw_value

    env_name = match.group(1)
    value = os.getenv(env_name)
    if not value:
        raise BenchmarkError(f"{model_name}: environment variable {env_name} is not set")

    return value


def delta_has_generated_text(delta: Any) -> bool:
    if delta is None:
        return False

    if hasattr(delta, "model_dump"):
        data = delta.model_dump(exclude_none=True)
    elif isinstance(delta, dict):
        data = delta
    else:
        return False

    def has_value(key: str) -> bool:
        value = data.get(key)
        return bool(value) if isinstance(value, (str, list)) else False

    return any(has_value(key) for key in ("content", "reasoning_content", "reasoning", "thinking", "refusal"))


def completion_tokens_from_usage(usage: Any) -> Optional[int]:
    if usage is None:
        return None

    if isinstance(usage, dict):
        tokens = usage.get("completion_tokens")
    else:
        tokens = getattr(usage, "completion_tokens", None)

    return tokens if isinstance(tokens, int) else None


def build_request(model_config: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    model_name = model_config.get("name")
    if not isinstance(model_name, str) or not model_name:
        raise BenchmarkError("model config is missing a valid name")

    request: Dict[str, Any] = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    for key in ("temperature", "max_tokens", "top_p", "frequency_penalty", "presence_penalty"):
        if key in model_config:
            request[key] = model_config[key]

    if "extra_body" in model_config:
        request["extra_body"] = model_config["extra_body"]

    return request


def create_client(model_config: Dict[str, Any]) -> OpenAI:
    model_name = model_config.get("name")
    if not isinstance(model_name, str) or not model_name:
        raise BenchmarkError("model config is missing a valid name")

    base_url = model_config.get("base_url")
    if not isinstance(base_url, str) or not base_url:
        raise BenchmarkError(f"{model_name}: base_url is missing or not a string")

    api_key = resolve_api_key(model_config.get("api_key"), model_name)
    return OpenAI(api_key=api_key, base_url=base_url)


def benchmark_prompt(client: OpenAI, model_config: Dict[str, Any], prompt: str) -> Tuple[int, float, float]:
    model_name = model_config["name"]

    start_time: Optional[float] = None
    end_time: Optional[float] = None
    completion_tokens: Optional[int] = None

    stream = client.chat.completions.create(**build_request(model_config, prompt))
    for chunk in stream:
        now = time.perf_counter()

        usage_tokens = completion_tokens_from_usage(getattr(chunk, "usage", None))
        if usage_tokens is not None:
            completion_tokens = usage_tokens

        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if start_time is None and delta_has_generated_text(delta):
                start_time = now

        end_time = now

    if start_time is None or end_time is None:
        raise BenchmarkError(f"{model_name}: no generated token delta was received")

    if completion_tokens is None:
        raise BenchmarkError(f"{model_name}: streaming usage.completion_tokens was not returned")

    elapsed_seconds = end_time - start_time
    if elapsed_seconds <= 0:
        raise BenchmarkError(f"{model_name}: measured generation duration is not positive")

    tps = completion_tokens / elapsed_seconds
    return completion_tokens, elapsed_seconds, tps


def benchmark_prompt_with_retries(
    client: OpenAI,
    model_config: Dict[str, Any],
    prompt: str,
    retries: int = RETRY_COUNT,
) -> Tuple[int, float, float, int]:
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            tokens, seconds, tps = benchmark_prompt(client, model_config, prompt)
            return tokens, seconds, tps, attempt
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(min(2 ** (attempt - 1), 8))

    raise BenchmarkError(f"failed after {retries} attempts: {last_error}")


def average(values: List[float]) -> Optional[float]:
    if not values:
        return None
    return sum(values) / len(values)


def failed_prompt_result(prompt_name: str, error: Exception | str, attempts: int = RETRY_COUNT) -> Dict[str, Any]:
    return {
        "prompt": prompt_name,
        "status": "failed",
        "error": str(error),
        "attempts": attempts,
    }


def successful_prompt_result(
    prompt_name: str,
    tokens: int,
    seconds: float,
    tps: float,
    attempts: int,
) -> Dict[str, Any]:
    return {
        "prompt": prompt_name,
        "status": "success",
        "completion_tokens": tokens,
        "seconds": seconds,
        "tps": tps,
        "attempts": attempts,
    }


def benchmark_model(
    model_config: Dict[str, Any],
    prompts: List[Tuple[str, str]],
) -> Dict[str, Any]:
    model_name = model_config.get("name", "<unnamed>")
    prompt_results: List[Dict[str, Any]] = []
    model_tps: List[float] = []

    try:
        client = create_client(model_config)
    except Exception as exc:
        prompt_results = [failed_prompt_result(prompt_name, exc) for prompt_name, _ in prompts]
        return {
            "model": model_name,
            "average_tps": None,
            "successful_prompts": 0,
            "total_prompts": len(prompts),
            "prompts": prompt_results,
        }

    for prompt_name, prompt in prompts:
        try:
            tokens, seconds, tps, attempts = benchmark_prompt_with_retries(client, model_config, prompt)
        except Exception as exc:
            prompt_results.append(failed_prompt_result(prompt_name, exc))
            continue

        model_tps.append(tps)
        prompt_results.append(successful_prompt_result(prompt_name, tokens, seconds, tps, attempts))

    avg_tps = average(model_tps)
    return {
        "model": model_name,
        "average_tps": avg_tps,
        "successful_prompts": len(model_tps),
        "total_prompts": len(prompts),
        "prompts": prompt_results,
    }


def print_model_result(result: Dict[str, Any]) -> None:
    print(f"=== {result['model']} ===")

    for prompt_result in result["prompts"]:
        prompt_name = prompt_result["prompt"]
        if prompt_result["status"] != "success":
            print(f"{prompt_name}: FAILED - {prompt_result['error']}")
            continue

        print(
            f"{prompt_name}: tokens={prompt_result['completion_tokens']}, "
            f"seconds={prompt_result['seconds']:.3f}, "
            f"tps={prompt_result['tps']:.2f}, "
            f"attempts={prompt_result['attempts']}"
        )

    avg_tps = result["average_tps"]
    if avg_tps is None:
        print("average_tps=N/A (no successful prompts)")
    else:
        print(
            f"average_tps={avg_tps:.2f} "
            f"({result['successful_prompts']}/{result['total_prompts']} prompts succeeded)"
        )
    print()


def write_output(results: Dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"{timestamp}.json"
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=2)
        file.write("\n")
    return output_path


def main() -> int:
    try:
        models = load_models(MODELS_PATH)
        prompts = load_prompts(PROMPTS_PATH)
    except Exception as exc:
        print(f"配置读取失败: {exc}", file=sys.stderr)
        return 1

    run_started_at = datetime.now().astimezone().isoformat()
    max_workers = max(1, len(models))
    model_results_by_index: Dict[int, Dict[str, Any]] = {}

    print(f"Loaded {len(models)} models and {len(prompts)} prompts.")
    print("TPS = usage.completion_tokens / seconds from first generated delta to stream end")
    print(f"Retry count per prompt: {RETRY_COUNT}")
    print(f"Running models in parallel with max_workers={max_workers}")
    print()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(benchmark_model, model_config, prompts): (index, str(model_config.get("name", "<unnamed>")))
            for index, model_config in enumerate(models)
        }

        for future in as_completed(futures):
            model_index, model_name = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "model": model_name,
                    "average_tps": None,
                    "successful_prompts": 0,
                    "total_prompts": len(prompts),
                    "prompts": [
                        failed_prompt_result(prompt_name, exc)
                        for prompt_name, _ in prompts
                    ],
                }

            model_results_by_index[model_index] = result
            print_model_result(result)

    ordered_model_results = [
        model_results_by_index[index]
        for index in range(len(models))
    ]

    output = {
        "started_at": run_started_at,
        "finished_at": datetime.now().astimezone().isoformat(),
        "retry_count": RETRY_COUNT,
        "parallelism": {
            "scope": "models",
            "max_workers": max_workers,
        },
        "tps_formula": "usage.completion_tokens / seconds from first generated delta to stream end",
        "models": ordered_model_results,
    }
    output_path = write_output(output)
    print(f"Saved JSON results to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
