"""
LLM Fallback & Escalation Tool
===============================
When Cline gets stuck in a loop, this script sends the problem to an external
LLM (OpenAI API or any LM Studio server) for a fresh perspective.

Usage:
    python tools/llm_fallback.py discover
    python tools/llm_fallback.py escalate --problem "error description" --code "snippet"
    python tools/llm_fallback.py escalate --problem "error" --code "snippet" --server http://192.168.50.3:1234
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path

CONFIG_PATH = Path(__file__).parent / "fallback_config.json"
ENV_PATH = Path(__file__).parent.parent / ".env"


def load_dotenv():
    """Load variables from .env file into os.environ."""
    if not ENV_PATH.exists():
        return
    with open(ENV_PATH, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def load_config():
    if not CONFIG_PATH.exists():
        print(f"ERROR: Config not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)


# ── Discovery ────────────────────────────────────────────────────────────────

def probe_lm_studio(url, timeout=5):
    """Check if an LM Studio server is reachable and list its models."""
    models_url = f"{url.rstrip('/')}/v1/models"
    try:
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
            models = [m["id"] for m in data.get("data", [])]
            return {"online": True, "models": models}
    except (urllib.error.URLError, OSError, json.JSONDecodeError):
        return {"online": False, "models": []}


def discover(config):
    """Discover all configured LM Studio servers and their models."""
    print("=" * 60)
    print("LM Studio Server Discovery")
    print("=" * 60)

    for server in config.get("lm_studio_servers", []):
        name = server["name"]
        url = server["url"]
        desc = server.get("description", "")
        result = probe_lm_studio(url)

        status = "ONLINE" if result["online"] else "OFFLINE"
        print(f"\n[{status}] {name} — {url}")
        if desc:
            print(f"         {desc}")
        if result["online"]:
            if result["models"]:
                print(f"         Models: {', '.join(result['models'])}")
            else:
                print("         Models: (none loaded)")
        else:
            print("         Could not connect")

    # Check OpenAI
    openai_cfg = config.get("openai", {})
    api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        print(f"\n[CONFIGURED] OpenAI API — model: {openai_cfg.get('model', 'gpt-4o')}")
    else:
        print("\n[NOT CONFIGURED] OpenAI API — set api_key in config or OPENAI_API_KEY env var")

    print()


# ── Escalation ───────────────────────────────────────────────────────────────

def build_escalation_prompt(problem, code, attempts):
    return f"""You are a senior software engineer brought in to debug a problem that another
AI coding assistant has been unable to solve after {attempts} attempts.

## Problem Description
{problem}

## Relevant Code
```
{code}
```

## Instructions
1. Analyze the root cause — don't just treat symptoms.
2. If the code has a fundamental design issue, say so clearly.
3. Provide a CONCRETE fix with exact code changes (not just descriptions).
4. Explain WHY previous attempts likely failed.
5. If you need more context, list exactly what files/info would help.
"""


def call_lm_studio(url, prompt, timeout=120):
    """Send a chat completion request to an LM Studio server."""
    endpoint = f"{url.rstrip('/')}/v1/chat/completions"
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": "You are an expert debugging assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


def call_openai(config, prompt, timeout=120):
    """Send a chat completion request to the OpenAI API."""
    openai_cfg = config.get("openai", {})
    api_key = openai_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None

    base_url = openai_cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    model = openai_cfg.get("model", "gpt-4o")
    endpoint = f"{base_url}/chat/completions"

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert debugging assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }).encode()

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode())
        return data["choices"][0]["message"]["content"]


def pick_best_server(config, preferred_url=None):
    """Pick the best available server. Prefers the explicitly requested one,
    then the configured preferred server, then any online LM Studio, then OpenAI."""
    servers = config.get("lm_studio_servers", [])
    prefer_name = config.get("escalation", {}).get("prefer_server", "")

    # If a specific server URL was requested, try it first
    if preferred_url:
        result = probe_lm_studio(preferred_url)
        if result["online"]:
            return {"type": "lm_studio", "url": preferred_url, "name": "requested"}
        print(f"WARNING: Requested server {preferred_url} is offline", file=sys.stderr)

    # Try preferred server from config
    for s in servers:
        if s["name"] == prefer_name:
            result = probe_lm_studio(s["url"])
            if result["online"]:
                return {"type": "lm_studio", "url": s["url"], "name": s["name"]}
            break

    # Try any online LM Studio server
    for s in servers:
        result = probe_lm_studio(s["url"])
        if result["online"]:
            return {"type": "lm_studio", "url": s["url"], "name": s["name"]}

    # Fall back to OpenAI
    api_key = config.get("openai", {}).get("api_key") or os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        return {"type": "openai", "name": "OpenAI API"}

    return None


def escalate(config, problem, code, attempts, server_url=None):
    """Escalate a problem to the best available external LLM."""
    timeout = config.get("escalation", {}).get("timeout_seconds", 120)
    prompt = build_escalation_prompt(problem, code, attempts)

    target = pick_best_server(config, preferred_url=server_url)
    if target is None:
        print("ERROR: No LLM servers available. Check config and server status.",
              file=sys.stderr)
        print("Run: python tools/llm_fallback.py discover", file=sys.stderr)
        sys.exit(1)

    print(f"Escalating to: {target['name']} ({target['type']})")
    print(f"Problem: {problem[:100]}{'...' if len(problem) > 100 else ''}")
    print(f"Attempts so far: {attempts}")
    print("-" * 60)

    try:
        if target["type"] == "lm_studio":
            response = call_lm_studio(target["url"], prompt, timeout=timeout)
        else:
            response = call_openai(config, prompt, timeout=timeout)

        if response:
            print(response)
        else:
            print("ERROR: Empty response from LLM", file=sys.stderr)
            sys.exit(1)
    except (urllib.error.URLError, OSError) as e:
        print(f"ERROR: Failed to reach {target['name']}: {e}", file=sys.stderr)
        sys.exit(1)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LLM Fallback & Escalation Tool for Cline loop detection"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # discover
    sub.add_parser("discover", help="Discover available LLM servers and models")

    # escalate
    esc = sub.add_parser("escalate", help="Escalate a stuck problem to an external LLM")
    esc.add_argument("--problem", required=True, help="Description of the problem/error")
    esc.add_argument("--code", required=True, help="Relevant code snippet")
    esc.add_argument("--attempts", type=int, default=5,
                     help="Number of failed attempts so far")
    esc.add_argument("--server", default=None,
                     help="Specific LM Studio server URL to use")

    args = parser.parse_args()
    load_dotenv()
    config = load_config()

    if args.command == "discover":
        discover(config)
    elif args.command == "escalate":
        escalate(config, args.problem, args.code, args.attempts,
                 server_url=args.server)


if __name__ == "__main__":
    main()
