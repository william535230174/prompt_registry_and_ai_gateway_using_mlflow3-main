---
marp: true
theme: default
paginate: true
size: 16:9
title: Prompt Registry and AI Gateway using MLflow 3.13
description: A short guide to governing prompts and model access with MLflow 3.13
style: |
  :root {
    --ink: #1b2430;
    --muted: #5b6675;
    --accent: #0194e2;      /* MLflow blue */
    --accent-2: #43c9ed;
    --bg-soft: #f3f7fb;
    --bad: #c2410c;
  }
  section {
    font-family: "Inter", "Helvetica Neue", Arial, sans-serif;
    color: var(--ink);
    font-size: 26px;
    padding: 56px 72px;
    background: #ffffff;
  }
  h1 { color: var(--accent); font-size: 1.6em; margin-bottom: 0.3em; }
  h2 { color: var(--ink); font-size: 1.25em; border-bottom: 3px solid var(--accent-2); padding-bottom: 0.15em; }
  h3 { color: var(--accent); font-size: 0.95em; margin: 0.5em 0 0.2em; }
  strong { color: var(--ink); }
  em { color: var(--muted); }
  code { background: var(--bg-soft); color: #0b5d8c; border-radius: 4px; padding: 0 0.25em; }
  pre { background: #0f1722; border-radius: 8px; font-size: 0.62em; line-height: 1.4; }
  pre code { background: transparent; color: #e6edf3; padding: 0; }
  /* Dark-background syntax colors (GitHub Dark) */
  pre .hljs-keyword, pre .hljs-built_in, pre .hljs-meta { color: #ff7b72; }
  pre .hljs-string, pre .hljs-regexp { color: #a5d6ff; }
  pre .hljs-number, pre .hljs-literal, pre .hljs-attr, pre .hljs-variable { color: #79c0ff; }
  pre .hljs-title, pre .hljs-function, pre .hljs-section { color: #d2a8ff; }
  pre .hljs-comment { color: #8b949e; font-style: italic; }
  pre .hljs-params, pre .hljs-name, pre .hljs-bullet { color: #e6edf3; }
  pre .hljs-type, pre .hljs-symbol { color: #ffa657; }
  table { font-size: 0.74em; border-collapse: collapse; }
  th { background: var(--accent); color: #fff; }
  td, th { padding: 6px 12px; }
  blockquote { border-left: 5px solid var(--accent-2); background: var(--bg-soft); color: var(--ink); padding: 0.4em 0.9em; font-size: 0.8em; }
  section::after { color: var(--muted); font-size: 0.6em; }
  footer { color: var(--muted); font-size: 0.55em; }

  /* Title & section-divider slides */
  section.lead {
    background: linear-gradient(135deg, #0b3a5c 0%, #0194e2 100%);
    color: #fff; justify-content: center;
  }
  section.lead h1 { color: #fff; font-size: 2em; }
  section.lead h2 { color: #d8f1fb; border: none; font-weight: 400; }
  section.lead p, section.lead strong { color: #e6f6fd; }
  section.lead em { color: #ffffff; font-weight: 600; }
  section.lead::after { color: #cfe9f7; }

  /* Dense slides: smaller type so code and tables fit */
  section.dense { font-size: 22px; }
  section.dense pre { font-size: 0.66em; }

  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 32px; align-items: start; }
  .pill { display: inline-block; background: var(--accent); color: #fff; border-radius: 999px; padding: 2px 14px; font-size: 0.7em; }
  .big { font-size: 1.25em; font-weight: 600; }
footer: MLflow 3.13 · Prompt Registry & AI Gateway
---

<!-- _class: lead -->
<!-- _paginate: false -->
<!-- _footer: "" -->

# Prompt Registry & AI Gateway
## using MLflow 3.13

Govern **what you say** to the model — and **how you reach** it.

---

## Agenda

<div class="cols">
<div>

### Part 1 — Prompt Registry
- Why a registry? The story
- Register, load, version
- Aliases: promote & roll back
- Older MLflow compatibility
- Compare, evaluate, optimize

</div>
<div>

### Part 2 — AI Gateway
- Why a gateway? The story
- Endpoints & `config.yaml`
- Calling it: Deployments client, OpenAI Responses API, Pydantic AI
- Traffic routing & fallbacks
- Usage tracking

</div>
</div>

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Part 1
## Prompt Registry

Store, version, alias, compare, evaluate, and optimize prompts — the way a model registry manages models, or Git manages code.

---

## Before: the prompt lives next to the code

```python
SYSTEM_PROMPT = """
You are an expert summarizer. Condense the content into exactly {num_sentences}
clear, neutral sentences that capture the key points.
"""

def summarize(text: str, num_sentences: int) -> str:
    response = client.responses.create(
        model="gpt-6-luna",
        reasoning={"effort": "medium"},
        instructions=SYSTEM_PROMPT.format(num_sentences=num_sentences),
        input=text,
    )
    return response.output_text
```

Also common: a `prompts.py` / `.yaml` file, prompts inside notebooks, or copy-pasted over chat.
**Fine for a prototype. Breaks down in a real project.**

---

<!-- _class: dense -->

## The problems

| Problem | What it looks like in practice |
| --- | --- |
| **No version history** | You "improve" a prompt, quality drops, and you can't recover the old wording. |
| **Deployment coupling** | A one-word tweak needs a code change, PR, build, and redeploy. |
| **No safe rollback** | Reverting means reverting a commit, not flipping a pointer. |
| **Hard to compare** | "What changed between the version that worked and the one that didn't?" |
| **No lineage** | Which prompt version produced this trace, evaluation, or model? |
| **Poor collaboration** | PMs and domain experts can't iterate without touching code. |
| **No quality loop** | Changes are judged by vibes, not metrics on a dataset. |

> **Root cause:** prompts are treated as *incidental strings* instead of *versioned artifacts* with their own lifecycle.

---

## What a Prompt Registry offers

- **Version control** — Git-style commits; every version is **immutable** → reproducible
- **Aliasing** — mutable pointers (`production`, `champion`) to promote, A/B test, and roll back **without code changes**
- **Lineage** — tied into MLflow Tracing, Evaluation, and Models
- **Collaboration** — one central registry with a UI for the whole team
- **Quality loop** — plugs straight into evaluation and automated optimization

---

## Setup in MLflow 3.13

<div class="cols">
<div>

### Install
```bash
pip install --upgrade "mlflow>=3.13"
```

### Start tracking server + UI
```bash
mlflow server --host 127.0.0.1 --port 5000
```

### Point your code at it
```python
import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

</div>
<div>

### Core concepts
- **Prompt** — named, versioned template
- **Template** — text or chat messages with `{{ variables }}` (Jinja2 auto-detected)
- **Version** — sequential & immutable
- **Alias** — mutable pointer (`@production`)
- **URI** — `prompts:/name/1` or `prompts:/name@alias`

APIs live under **`mlflow.genai`**.

</div>
</div>

---

## Register a prompt

```python
initial_template = """\
Summarize the content you are provided with in {{ num_sentences }} sentences.

Content: {{ content }}
"""

prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=initial_template,
    commit_message="Initial summarization prompt",
    tags={"task": "summarization", "language": "en"},
)
print(f"Created '{prompt.name}' version {prompt.version}")
```

**Chat prompts** take a message list instead:

```python
mlflow.genai.register_prompt(name="support-assistant", template=[
    {"role": "system", "content": "You are a helpful {{ tone }} support agent."},
    {"role": "user", "content": "{{ question }}"},
])
```

---

## Load and use a prompt

```python
import openai

prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")

response = openai.OpenAI().responses.create(
    model="gpt-6-luna",
    reasoning={"effort": "medium"},
    input=prompt.format(num_sentences=2, content="..."),
)
print(response.output_text)
```

> **Lineage for free:** calling `load_prompt()` inside a traced function or logged model records the prompt version as a dependency.

Also useful: `model_config` (mutable model settings per version), `response_format` (structured output), and `mlflow.genai.search_prompts(...)`.

---

## New version = same name, register again

```python
improved_template = """\
You are an expert summarizer. Condense the following content into exactly
{{ num_sentences }} clear, informative sentences.

Content: {{ content }}

Your summary should:
- Contain exactly {{ num_sentences }} sentences
- Include only the most important information
- Use a neutral, objective tone
"""

updated = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=improved_template,
    commit_message="Add explicit constraints and tone guidance",
)
print(f"Now at version {updated.version}")  # -> 2
```

Version 1 stays intact and immutable.

---

## Aliases: promote & roll back without a redeploy

<div class="cols">
<div>

### Promote
```python
mlflow.genai.set_prompt_alias(
    "summarization-prompt",
    alias="production", version=2,
)
```

### App always loads by alias
```python
mlflow.genai.load_prompt(
    "prompts:/summarization-prompt@production"
)
```

### Roll back instantly
```python
mlflow.genai.set_prompt_alias(
    "summarization-prompt",
    alias="production", version=1,
)
```

</div>
<div>

### Ways to reference a prompt

| URI | Meaning |
| --- | --- |
| `prompts:/name/2` | Exact version |
| `prompts:/name@production` | Alias target |
| `prompts:/name@latest` | Newest version |
| `load_prompt("name")` | Latest version |

> **Caching:** version URIs cache forever; alias URIs use a 60 s TTL. Pass `cache_ttl_seconds=0` to always fetch fresh.

</div>
</div>

---

<!-- _class: dense -->

## Prompts from an earlier MLflow (e.g. 3.12)

<div class="cols">
<div>

### 1. Storage is compatible
Same tracking backend → no migration.

```python
mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
```

### 2. The API namespace moved
Top-level `mlflow.*` prompt functions still work but emit a `FutureWarning`.

</div>
<div>

### Migration cheat sheet

| Deprecated | MLflow 3.13 |
| --- | --- |
| `mlflow.register_prompt` | `mlflow.genai.register_prompt` |
| `mlflow.load_prompt` | `mlflow.genai.load_prompt` |
| `mlflow.search_prompts` | `mlflow.genai.search_prompts` |
| `mlflow.set_prompt_alias` | `mlflow.genai.set_prompt_alias` |
| `mlflow.delete_prompt_alias` | `mlflow.genai.delete_prompt_alias` |

> Mixing old and new clients? Always use the **same tracking server**.

</div>
</div>

---

## Compare versions like Git commits

**In the UI:** Prompts tab → **Compare** view for a side-by-side, highlighted diff.

**In Python:**

```python
import difflib

v1 = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
v2 = mlflow.genai.load_prompt("prompts:/summarization-prompt/2")

diff = difflib.unified_diff(
    v1.template.splitlines(), v2.template.splitlines(),
    fromfile="v1", tofile="v2", lineterm="",
)
print("\n".join(diff))
```

Review flow: **diff the wording → evaluate both on a dataset → promote the winner via alias.**

---

<!-- _class: dense -->

## Evaluate a prompt on a dataset

```python
from mlflow.genai.scorers import Correctness

def predict_fn(content: str, num_sentences: int) -> str:
    prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
    response = openai.OpenAI().responses.create(
        model="gpt-6-luna",
        reasoning={"effort": "medium"},
        input=prompt.format(content=content, num_sentences=num_sentences),
    )
    return response.output_text

eval_data = [{
    "inputs": {"content": "MLflow is an open source platform...", "num_sentences": 1},
    "expectations": {"expected_response": "MLflow is an open source ML/AI platform."},
}]

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[Correctness(model="openai:/gpt-6-luna")],
)
```

Run it for two versions and compare scores in the MLflow UI.

---

<!-- _class: dense -->

## Optimize a prompt automatically

`mlflow.genai.optimize_prompts()` rewrites a prompt to score higher and saves it as a **new version**.

<div class="cols">
<div>

### It needs four things
1. **Target prompt URI(s)**
2. **`predict_fn`** that uses `load_prompt(...).format(...)`
3. **Training data** — `inputs` + `expectations`
4. **An optimizer** — GEPA or MetaPrompt

</div>
<div>

```python
from mlflow.genai.optimize import GepaPromptOptimizer

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(
        reflection_model="openai:/gpt-6-luna",
        max_metric_calls=300,
    ),
    scorers=[Correctness(model="openai:/gpt-6-luna")],
)

optimized = result.optimized_prompts[0]
print(optimized.uri, result.initial_eval_score,
      "->", result.final_eval_score)
```

</div>
</div>

---

## Choosing an optimizer

| | GEPA | MetaPrompt (zero-shot) | MetaPrompt (few-shot) |
| --- | --- | --- | --- |
| Needs training data | Yes (100+ ideal) | No | Yes (a few examples) |
| Speed | Moderate | Fast | Fast–moderate |
| Approach | Iterative reflection on failures | Restructure to best practices | Data-aware restructuring |
| Best for | High-stakes tasks with clear metrics | Quick wins, no data | Targeted wins, limited data |

---

<!-- _class: dense -->

## Custom scorers: optimize for more than correctness

```python
from mlflow.genai.scorers import scorer

@scorer
def accuracy_scorer(outputs, expectations):
    return 1.0 if outputs.strip().upper() == expectations["expected_response"] else 0.0

@scorer
def brevity_scorer(outputs):
    return min(1.0, 50 / max(len(outputs), 1))

def weighted_objective(scores):
    return 0.7 * scores["accuracy_scorer"] + 0.3 * scores["brevity_scorer"]

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-6-luna"),
    scorers=[accuracy_scorer, brevity_scorer],
    aggregation=weighted_objective,
)
```

---

## The prompt lifecycle loop

<br>

<p class="big">register → alias to production → evaluate → optimize → compare → promote the winner → roll back if needed</p>

<br>

Optimized prompts land back in the registry as new versions — so the loop closes on itself.

### Quick reference
```python
mlflow.genai.register_prompt(name="my-prompt", template="Hi {{ name }}")
mlflow.genai.load_prompt("prompts:/my-prompt@production").format(name="Ada")
mlflow.genai.set_prompt_alias("my-prompt", alias="production", version=1)
mlflow.genai.search_prompts(filter_string="task='summarization'")
```

---

<!-- _class: lead -->
<!-- _paginate: false -->

# Part 2
## AI Gateway

One OpenAI-compatible entry point in front of every LLM provider.
The registry governs *what you say*; the gateway governs *how you reach* the model.

---

## Before: every service talks to every provider

```python
import os
import openai
import anthropic

# One client, key, and SDK per provider — repeated across services
openai_client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

def summarize(text: str) -> str:
    resp = openai_client.responses.create(
        model="gpt-6-luna",
        reasoning={"effort": "medium"},
        input=text,
    )
    return resp.output_text
```

Each new provider = a new SDK, a new key in every service, new request/response handling.

---

<!-- _class: dense -->

## The problems

| Problem | What it looks like in practice |
| --- | --- |
| **Secrets sprawl** | Provider keys copied into every service, notebook, and CI job. |
| **Provider lock-in** | OpenAI → Anthropic means rewriting SDK calls everywhere. |
| **No central control** | Rate limits and spend caps re-implemented per app. |
| **No A/B testing or migration** | Shifting a % of traffic needs custom code. |
| **No resilience** | An outage or 429 takes the app down; fallback logic is bespoke. |
| **Fragmented observability** | Tokens, latency, and cost scattered across provider dashboards. |
| **Config = code changes** | Changing model or params redeploys every consumer. |

> **Root cause:** each application owns its own direct, uncontrolled connection to every provider.

---

## What an AI Gateway offers

- **Unified, OpenAI-compatible API** — call by *endpoint name*, not provider SDK
- **Centralized credentials** — provider keys live only on the gateway
- **Config-driven** — swap models or params without redeploying apps (hot-reload)
- **Rate limiting & budget control** — per-endpoint limits in one place
- **Traffic routing & fallbacks** — weighted A/B splits and failover chains
- **Usage tracking** — one dashboard for requests, latency, errors, tokens, cost

---

<!-- _class: dense -->

## Setting up the gateway

<div class="cols">
<div>

### Install
```bash
pip install --upgrade "mlflow[gateway]>=3.13"
```

### Keys stay on the gateway
```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Start (hot-reloads on config changes)
```bash
mlflow gateway start --config-path config.yaml \
  --host 127.0.0.1 --port 5000
```

Endpoints can also be managed in the **AI Gateway** tab of the UI.

</div>
<div>

### `config.yaml`
```yaml
endpoints:
  - name: fast-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-6-luna
      config:
        openai_api_key: $OPENAI_API_KEY
    limit:
      renewal_period: minute
      calls: 100

  - name: quality-chat
    endpoint_type: llm/v1/chat
    model:
      provider: anthropic
      name: claude-3-5-sonnet-20241022
      config:
        anthropic_api_key: $ANTHROPIC_API_KEY
```

</div>
</div>

---

## Core concepts

<br>

| Concept | Meaning |
| --- | --- |
| **Endpoint** | A named model + provider configuration (`fast-chat`) that apps reference by name |
| **Endpoint type** | The interface contract: `llm/v1/chat`, `llm/v1/completions`, `llm/v1/embeddings` |
| **Route** | The URL the gateway exposes for each endpoint (auto-generated) |
| **Limit** | Per-endpoint rate limit (`calls` per `renewal_period`) |

---

## Option A — MLflow Deployments client

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("http://127.0.0.1:5000")

for ep in client.list_endpoints():          # discover what's available
    print(ep["name"])

response = client.predict(
    endpoint="fast-chat",
    inputs={
        "messages": [{"role": "user", "content": "What is MLflow AI Gateway?"}],
        "temperature": 0.7,
    },
)
print(response["choices"][0]["message"]["content"])

emb = client.predict(endpoint="embeddings", inputs={"input": ["hello", "world"]})
```

Handles auth, retries, and parsing. Works with every provider.

---

## Option B — OpenAI SDK, Responses API

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5000/gateway/openai/v1",
    api_key="unused",  # provider auth is handled by the gateway
)

response = client.responses.create(
    model="fast-chat",                 # gateway endpoint name, not provider model
    reasoning={"effort": "medium"},    # thinking effort: low / medium / high
    instructions="You are a concise assistant.",
    input="What is MLflow AI Gateway?",
)
print(response.output_text)
```

The app holds **no provider key**.

---

## Option B — two things to know

<div class="cols">
<div>

### Which base URL?
The Responses API is served on the **OpenAI passthrough** route:

`/gateway/openai/v1` ✅

The unified route only exposes `chat/completions`:

`/gateway/mlflow/v1` ❌ for `responses.create()`

</div>
<div>

### Which providers?
Passthrough → the endpoint must speak the Responses API: **OpenAI, Azure OpenAI / Databricks, LiteLLM**.

Anthropic-backed endpoints (`quality-chat`) → use Option A or the unified chat route.

### Thinking effort
`reasoning` is passed through unchanged — the backing model must support reasoning.

</div>
</div>

---

## Combine with the Prompt Registry

Governed prompt in, governed model out:

```python
import mlflow
from mlflow.deployments import get_deploy_client

mlflow.set_tracking_uri("http://127.0.0.1:5000")
gateway = get_deploy_client("http://127.0.0.1:5000")

prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
messages = [{"role": "user",
             "content": prompt.format(num_sentences=2, content="...")}]

response = gateway.predict(endpoint="fast-chat", inputs={"messages": messages})
print(response["choices"][0]["message"]["content"])
```

---

<!-- _class: dense -->

## Pydantic AI through the gateway

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

model = OpenAIResponsesModel(
    "fast-chat",  # MLflow gateway endpoint name
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:5000/gateway/openai/v1",
        api_key="unused",  # gateway holds the real provider key
    ),
    settings=OpenAIResponsesModelSettings(openai_reasoning_effort="medium"),
)

agent = Agent(model, system_prompt="You are a concise assistant.")
print(agent.run_sync("Explain the MLflow AI Gateway in one sentence.").output)

# Feed the agent a governed system prompt from the registry
sys_prompt = mlflow.genai.load_prompt("prompts:/support-assistant@production")
agent = Agent(model, system_prompt=sys_prompt.format(tone="friendly"))
```

Anthropic-backed endpoint? Use `OpenAIChatModel` with `.../gateway/mlflow/v1`. Add `mlflow.openai.autolog()` for traces.

---

## Traffic routing & fallbacks

<div class="cols">
<div>

<span class="pill">Priority 1</span>

### Traffic split
Several models on one endpoint, each with a weight.

- Weights 1–100%, **must sum to exactly 100%**
- e.g. 50% GPT-5 / 50% Claude 4.5 Sonnet to test Claude on real traffic

**Use for:** A/B tests, gradual migration, load distribution

</div>
<div>

<span class="pill">Priority 2</span>

### Fallbacks
Ordered models tried **top to bottom** when the primary errors or is rate-limited.

- e.g. primary → GPT-5 mini → Claude 4.5 Haiku → error
- First success wins

**Use for:** outages, throttling, regional failover

</div>
</div>

> **Pitfall:** fallbacks only activate once traffic-split weights total exactly 100%. Callers still use one endpoint name — app code never changes.

---

## Usage tracking

<div class="cols">
<div>

### Enable
Endpoint detail page → **Usage Tracking** → toggle **on** → **Save changes**

### View
**AI Gateway › Usage** (or the endpoint's **Usage** tab). Filter by endpoint, hour/day/week, and time range.

> Token and cost charts depend on provider support and pricing data — some may be empty.

</div>
<div>

### Dashboard
- **Requests** — volume over time
- **Latency** — p50 / p90 / p99
- **Errors** — rate & breakdown
- **Token usage** — input vs. output
- **Tokens per request**
- **Cost** — by model or provider, over time

</div>
</div>

---

## Recap

<div class="cols">
<div>

### Prompt Registry — *what you say*
- Prompts as immutable, versioned artifacts
- Aliases decouple prompts from deploys
- Diff → evaluate → optimize → promote
- `mlflow.genai.*` APIs

</div>
<div>

### AI Gateway — *how you reach the model*
- One endpoint name, any provider
- Keys, limits, budgets in one place
- Routing & fallbacks behind the endpoint
- Unified usage & cost tracking

</div>
</div>

---

## References

<div class="cols">
<div>

### Prompt Registry
- [Overview](https://mlflow.org/docs/latest/genai/prompt-registry/)
- [Use prompts in apps](https://mlflow.org/docs/latest/genai/prompt-registry/use-prompts-in-apps/)
- [Lifecycles with aliases](https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/)
- [Optimize prompts](https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/)

</div>
<div>

### AI Gateway
- [Overview](https://mlflow.org/docs/latest/genai/governance/ai-gateway/)
- [Setup](https://mlflow.org/docs/latest/genai/governance/ai-gateway/setup/)
- [Configuration](https://mlflow.org/docs/latest/genai/governance/ai-gateway/configuration/)
- [Usage](https://mlflow.org/docs/latest/genai/governance/ai-gateway/usage/)
- [Traffic routing & fallbacks](https://mlflow.org/docs/latest/genai/governance/ai-gateway/traffic-routing-fallbacks/)
- [Usage tracking](https://mlflow.org/docs/latest/genai/governance/ai-gateway/usage-tracking/)

</div>
</div>
