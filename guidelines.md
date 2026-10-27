# Prompt Registry and AI Gateway using MLflow 3.13: A Short Guidelines

A hands-on guide to managing prompts as first-class, versioned assets using the
**MLflow Prompt Registry** (MLflow 3.13).

---

## Prompt Registry

The Prompt Registry lets you store, version, alias, compare, evaluate, and
optimize the prompts that power your LLM applications and agents — the same way
a model registry manages models or Git manages code.

### Why This Matters (The Story)

Before diving into the API, it helps to understand *why* a registry exists at
all. The sections below walk from the naive approach, to its problems, to what a
registry fixes.

---

### Storing System Prompts without Prompt Registry

When you first build an LLM feature, the prompt usually lives right next to the
code that calls the model:

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

Common variations of "no registry" storage:

- Hardcoded string constants in application code.
- A `prompts.py` module or a `.txt` / `.yaml` / `.json` file in the repo.
- Prompts embedded in notebooks or scattered across microservices.
- Copy-pasted prompts shared over chat/docs between teammates.

This is fine for a prototype. It breaks down quickly in a real project.

---

### The Problems in Storing Prompts without Prompt Registry

| Problem | What it looks like in practice |
| --- | --- |
| **No version history** | You "improve" a prompt, quality drops, and you can't recover the previous wording. |
| **Deployment coupling** | Changing a prompt requires a code change, PR, build, and redeploy — even for a one-word tweak. |
| **No safe rollback** | Reverting a bad prompt means reverting a commit, not flipping a pointer. |
| **Hard to compare** | "What actually changed between the version that worked and the one that didn't?" is a manual diff hunt. |
| **No lineage** | You can't tell which prompt version produced a given trace, evaluation, or model. |
| **Poor collaboration** | Non-engineers (PMs, domain experts) can't iterate on prompts without touching code. |
| **No systematic quality loop** | Prompt changes are judged by vibes, not by evaluation metrics on a dataset. |

The root cause: prompts are treated as *incidental strings* instead of
*versioned artifacts* with their own lifecycle.

---

### What a Prompt Registry Offers

A Prompt Registry treats each prompt as a governed, versioned entity:

- **Version control** — Git-inspired, commit-based versioning. Every version is
  **immutable**, guaranteeing reproducibility.
- **Aliasing** — Mutable named pointers (e.g. `production`, `champion`) so you
  can promote, A/B test, and roll back **without changing application code**.
- **Lineage** — Prompts integrate with MLflow Tracing, Evaluation, and Models,
  so you always know which prompt version produced which result.
- **Collaboration** — A centralized registry (with UI) lets the whole team
  discover and build on each other's prompts.
- **Quality loop** — Prompts plug directly into evaluation and automated
  optimization.

---

### Prompt Registry in MLflow 3.13

In MLflow 3.x the Prompt Registry APIs live under the **`mlflow.genai`**
namespace (this changed from the old top-level `mlflow.*` prompt functions — see
the backward-compatibility section below).

**Install / upgrade:**

```bash
pip install --upgrade "mlflow>=3.13"
```

**Start a tracking server + UI** (the registry needs a backend to store
prompts):

```bash
mlflow server --host 127.0.0.1 --port 5000
```

**Point your code at it:**

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
```

Open the UI at `http://127.0.0.1:5000` and use the **Prompts** tab to browse
everything you register below.

#### Core concepts

- **Prompt** — A named, versioned template.
- **Template** — Either a **text** string or a **chat** message list, with
  variables in `{{double_brace}}` format. Jinja2 (`{% ... %}`) is auto-detected
  for conditionals/loops.
- **Version** — Sequential, immutable revision of the template.
- **Alias** — Mutable pointer to a version (`@production`, `@latest`).
- **URI** — `prompts:/<name>/<version>` or `prompts:/<name>@<alias>`.

---

### How to Register a New Prompt using MLflow 3.13 with Python

Use `mlflow.genai.register_prompt()`. Variables use double curly braces.

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

initial_template = """\
Summarize the content you are provided with in {{ num_sentences }} sentences.

Content: {{ content }}
"""

prompt = mlflow.genai.register_prompt(
    name="summarization-prompt",
    template=initial_template,
    commit_message="Initial summarization prompt",
    tags={"author": "author@example.com", "task": "summarization", "language": "en"},
)

print(f"Created '{prompt.name}' version {prompt.version}")
```

**Chat prompts** use a list of role/content messages:

```python
chat_prompt = mlflow.genai.register_prompt(
    name="support-assistant",
    template=[
        {"role": "system", "content": "You are a helpful {{ tone }} support agent."},
        {"role": "user", "content": "{{ question }}"},
    ],
    commit_message="Initial chat prompt",
)
```

**Loading and using a prompt** — load it, then fill variables with `.format()`:

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

> Tip: When `load_prompt()` is called inside a traced function or a logged
> model, MLflow automatically records the prompt version as a dependency, giving
> you end-to-end lineage.

**Optional extras worth knowing:**

- Attach model settings to a prompt version with `model_config` (model name,
  temperature, max_tokens, ...). Model config is *mutable* even though the
  template is not.
- Provide a `response_format` (a Pydantic model or JSON schema) to declare
  structured output.
- Discover prompts with `mlflow.genai.search_prompts(filter_string="task='summarization'")`.

---

### How to Version and Use Prompt Aliases using MLflow 3.13 with Python

#### Create a new version

Call `register_prompt()` again with the **same name**. MLflow auto-increments
the version; the old version stays intact and immutable.

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

#### Use aliases to decouple prompts from code

Hardcoding `prompts:/summarization-prompt/2` in production means every promotion
is a code change. Aliases fix this — the code references a *role*, not a number.

```python
# Point the "production" alias at version 2
mlflow.genai.set_prompt_alias("summarization-prompt", alias="production", version=2)
```

Your application always loads by alias:

```python
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
```

Promotions and rollbacks become a single, code-free operation:

```python
# Roll back to version 1 instantly — no redeploy
mlflow.genai.set_prompt_alias("summarization-prompt", alias="production", version=1)
```

#### Ways to reference a prompt

| URI | Meaning |
| --- | --- |
| `prompts:/name/2` | Exact, immutable version (safe to cache forever) |
| `prompts:/name@production` | Whatever version the alias points to |
| `prompts:/name@latest` | Reserved alias: newest version automatically |
| `mlflow.genai.load_prompt("name")` | Name only → latest version |

> **Caching note:** version URIs are cached indefinitely (they're immutable);
> alias URIs default to a 60-second TTL so updates propagate. Override per call
> with `load_prompt(..., cache_ttl_seconds=0)` to always fetch fresh.

---

### How to Access a Prompt Registered in an Earlier MLflow (e.g. 3.12) Prompt Registry with Python

Two things to keep straight: **where the prompt is stored** and **which API you
call**.

**1. Storage is compatible.** Prompts registered with an earlier MLflow version
live in the same tracking backend. As long as you point at the same tracking
URI, MLflow 3.13 reads them with no migration step:

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")  # same backend as before

# Works regardless of which MLflow version originally registered it
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
```

**2. The API namespace moved.** Older code and tutorials used **top-level**
functions like `mlflow.load_prompt()` / `mlflow.register_prompt()`. In MLflow 3.x
these moved under `mlflow.genai.*`. The old names still work but emit a
`FutureWarning`, so update your imports:

```python
# Deprecated (emits FutureWarning) — old 2.x / early-3.x style
prompt = mlflow.load_prompt("prompts:/summarization-prompt/1")

# Current (MLflow 3.13)
prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
```

Migration cheat sheet:

| Deprecated | MLflow 3.13 |
| --- | --- |
| `mlflow.register_prompt(...)` | `mlflow.genai.register_prompt(...)` |
| `mlflow.load_prompt(...)` | `mlflow.genai.load_prompt(...)` |
| `mlflow.search_prompts(...)` | `mlflow.genai.search_prompts(...)` |
| `mlflow.set_prompt_alias(...)` | `mlflow.genai.set_prompt_alias(...)` |
| `mlflow.delete_prompt_alias(...)` | `mlflow.genai.delete_prompt_alias(...)` |

> If you register prompts from a mix of old and new client versions, always talk
> to the **same tracking server** so versions and aliases stay consistent.

---

### How to Compare Prompts Registered in the Prompt Registry

Because every version is immutable and stored, you can diff versions like Git
commits.

**In the UI (recommended for reviews):** open the prompt in the **Prompts** tab
and use the **Compare** view for side-by-side, diff-highlighted comparison of two
versions.

**In Python** — load two versions and inspect them:

```python
import difflib
import mlflow

v1 = mlflow.genai.load_prompt("prompts:/summarization-prompt/1")
v2 = mlflow.genai.load_prompt("prompts:/summarization-prompt/2")

print("v1 commit:", v1.commit_message)
print("v2 commit:", v2.commit_message)

diff = difflib.unified_diff(
    v1.template.splitlines(),
    v2.template.splitlines(),
    fromfile="v1",
    tofile="v2",
    lineterm="",
)
print("\n".join(diff))
```

Comparing by **behavior** (not just text) is the job of evaluation — covered
next. The typical review flow is: diff the wording → evaluate both versions on a
dataset → promote the winner via alias.

---

### How to Evaluate & Optimize Prompts

Diffs tell you *what changed*; evaluation and optimization tell you *whether it's
better*.

#### Evaluate a prompt on a dataset

Use `mlflow.genai.evaluate()` with a prediction function that loads the prompt
from the registry, a dataset of inputs + expectations, and one or more scorers.

```python
import mlflow
import openai
from mlflow.genai.scorers import Correctness

def predict_fn(content: str, num_sentences: int) -> str:
    prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
    response = openai.OpenAI().responses.create(
        model="gpt-6-luna",
        reasoning={"effort": "medium"},
        input=prompt.format(content=content, num_sentences=num_sentences),
    )
    return response.output_text

eval_data = [
    {
        "inputs": {"content": "MLflow is an open source platform...", "num_sentences": 1},
        "expectations": {"expected_response": "MLflow is an open source ML/AI platform."},
    },
    # ... more rows ...
]

results = mlflow.genai.evaluate(
    data=eval_data,
    predict_fn=predict_fn,
    scorers=[Correctness(model="openai:/gpt-6-luna")],
)
```

Run this for two prompt versions and compare their scores in the MLflow UI to
pick a winner objectively.

#### Optimize a prompt automatically

`mlflow.genai.optimize_prompts()` (requires **MLflow >= 3.5.0**) automatically
rewrites a registered prompt to score higher on your metrics, then saves the
result as a **new version** in the registry.

It needs four things:

1. **Target prompt URI(s)** — which prompt(s) to improve.
2. **`predict_fn`** — must load the prompt via `load_prompt(...).format(...)`.
3. **Training data** — rows of `inputs` + `expectations`.
4. **An optimizer** — `GepaPromptOptimizer` (iterative reflection, needs data)
   or `MetaPromptOptimizer` (fast restructuring, works zero- or few-shot).

```python
import mlflow
import openai
from mlflow.genai.optimize import GepaPromptOptimizer
from mlflow.genai.scorers import Correctness

prompt = mlflow.genai.register_prompt(
    name="medical_section_classifier",
    template=(
        "Classify this medical research paper sentence into one of: "
        "CONCLUSIONS, RESULTS, METHODS, OBJECTIVE, BACKGROUND.\n\n"
        "Sentence: {{ sentence }}"
    ),
)

def predict_fn(sentence: str) -> str:
    p = mlflow.genai.load_prompt("prompts:/medical_section_classifier/1")
    response = openai.OpenAI().responses.create(
        model="gpt-6-luna",
        reasoning={"effort": "medium"},
        input=p.format(sentence=sentence),
    )
    return response.output_text

train_data = [
    {"inputs": {"sentence": "This study is a randomised controlled trial..."},
     "expectations": {"expected_response": "METHODS"}},
    {"inputs": {"sentence": "Both groups showed improvement in symptoms..."},
     "expectations": {"expected_response": "RESULTS"}},
    # ... 100+ rows recommended for GEPA ...
]

result = mlflow.genai.optimize_prompts(
    predict_fn=predict_fn,
    train_data=train_data,
    prompt_uris=[prompt.uri],
    optimizer=GepaPromptOptimizer(reflection_model="openai:/gpt-6-luna", max_metric_calls=300),
    scorers=[Correctness(model="openai:/gpt-6-luna")],
)

optimized = result.optimized_prompts[0]
print(f"Optimized -> {optimized.uri} (version {optimized.version})")
print(f"Score: {result.initial_eval_score} -> {result.final_eval_score}")
```

**Choosing an optimizer:**

| | GEPA | Metaprompting (zero-shot) | Metaprompting (few-shot) |
| --- | --- | --- | --- |
| Needs training data | Yes (100+ ideal) | No | Yes (a few examples) |
| Speed | Moderate | Fast | Fast–moderate |
| Approach | Iterative reflection on failures | Restructure to best practices | Data-aware restructuring |
| Best for | High-stakes tasks with clear metrics | Quick wins, no data | Targeted wins with limited data |

**Custom scorers** let you optimize for more than correctness (e.g. brevity,
safety), optionally combined with an aggregation function:

```python
from typing import Any
from mlflow.genai.scorers import scorer

@scorer
def accuracy_scorer(outputs: Any, expectations: dict[str, Any]):
    return 1.0 if outputs.strip().upper() == expectations["expected_response"] else 0.0

@scorer
def brevity_scorer(outputs: Any):
    return min(1.0, 50 / max(len(outputs), 1))

def weighted_objective(scores: dict[str, Any]):
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

Because optimized prompts land back in the registry as new versions, you close
the loop: **register → alias to production → evaluate → optimize → compare →
promote the winner → (roll back instantly if needed).**

---

## Quick Reference

```python
import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Register / new version (same name auto-increments)
p = mlflow.genai.register_prompt(name="my-prompt", template="Hi {{ name }}",
                                 commit_message="init")

# Load
mlflow.genai.load_prompt("my-prompt")                    # latest
mlflow.genai.load_prompt("prompts:/my-prompt/1")         # by version
mlflow.genai.load_prompt("prompts:/my-prompt@production") # by alias

# Format
mlflow.genai.load_prompt("my-prompt").format(name="Ada")

# Aliases
mlflow.genai.set_prompt_alias("my-prompt", alias="production", version=1)
mlflow.genai.delete_prompt_alias("my-prompt", alias="production")

# Search
mlflow.genai.search_prompts(filter_string="task='summarization'")
```

### References

- Prompt Registry overview: https://mlflow.org/docs/latest/genai/prompt-registry/
- Use prompts in apps: https://mlflow.org/docs/latest/genai/prompt-registry/use-prompts-in-apps/
- Manage lifecycles with aliases: https://mlflow.org/docs/latest/genai/prompt-registry/manage-prompt-lifecycles-with-aliases/
- Optimize prompts: https://mlflow.org/docs/latest/genai/prompt-registry/optimize-prompts/

## AI Gateway

The **MLflow AI Gateway** is a single, OpenAI-compatible entry point that sits
in front of every LLM provider you use (OpenAI, Anthropic, Bedrock, self-hosted,
...). Your application talks to *one* endpoint; the gateway handles provider
authentication, model selection, rate limits, routing, fallbacks, and usage
tracking.

It pairs naturally with the Prompt Registry: the registry governs *what you
say* to the model, the gateway governs *how you reach* the model.

### Why This Matters (The Story)

As with the Prompt Registry, it helps to see the naive approach and its
problems before the API.

---

### Accessing LLMs without an AI Gateway

Without a gateway, every service calls each provider's SDK directly, with
provider keys and model names hardcoded throughout the codebase:

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

Each new provider means a new SDK, a new key distributed to every service, and
new provider-specific request/response handling.

---

### The Problems of Accessing LLMs without an AI Gateway

| Problem | What it looks like in practice |
| --- | --- |
| **Secrets sprawl** | Provider API keys are copied into every service, notebook, and CI job — hard to rotate, easy to leak. |
| **Provider lock-in** | Switching OpenAI → Anthropic means rewriting SDK calls and payload shapes in many places. |
| **No central control** | Rate limits, spend caps, and access policies must be re-implemented per app. |
| **No A/B testing or migration** | Shifting a % of traffic to a new model requires custom code. |
| **No resilience** | A provider outage or 429 takes your app down; fallback logic is bespoke and duplicated. |
| **Fragmented observability** | Token usage, latency, and cost are scattered across provider dashboards, not unified. |
| **Config = code changes** | Changing model or params requires a redeploy of every consumer. |

The root cause: each application owns its own direct, uncontrolled connection to
every provider.

---

### What an AI Gateway Offers to Solve These Problems

- **Unified, OpenAI-compatible API** — one interface for every provider; call by
  *endpoint name*, not provider SDK.
- **Centralized credentials** — provider keys live only on the gateway; apps
  hold no provider secrets.
- **Config-driven** — swap models/providers or tune params without touching or
  redeploying application code (hot-reload supported).
- **Rate limiting & budget control** — enforce per-endpoint call limits and
  spend policies in one place.
- **Traffic routing & fallbacks** — weighted A/B splits and automatic failover
  chains for high availability.
- **Usage tracking** — unified dashboards for requests, latency, errors, tokens,
  and cost across all providers.

---

### AI Gateway in MLflow 3.13

The gateway runs as part of your MLflow server and exposes an OpenAI-compatible
endpoint for any provider.

**Install with gateway extras:**

```bash
pip install --upgrade "mlflow[gateway]>=3.13"
```

**Store provider keys as environment variables** (the gateway reads these; your
apps never see them):

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

**Define endpoints in `config.yaml`.** Each endpoint has a `name`, an
`endpoint_type` (`llm/v1/chat`, `llm/v1/completions`, or `llm/v1/embeddings`),
and a `model` block. `$VAR` values are read from the environment:

```yaml
endpoints:
  # Cost-effective chat endpoint
  - name: fast-chat
    endpoint_type: llm/v1/chat
    model:
      provider: openai
      name: gpt-6-luna
      config:
        openai_api_key: $OPENAI_API_KEY
    # Optional: per-endpoint rate limit
    limit:
      renewal_period: minute
      calls: 100

  # High-quality chat endpoint (different provider, same interface)
  - name: quality-chat
    endpoint_type: llm/v1/chat
    model:
      provider: anthropic
      name: claude-3-5-sonnet-20241022
      config:
        anthropic_api_key: $ANTHROPIC_API_KEY

  # Embeddings endpoint
  - name: embeddings
    endpoint_type: llm/v1/embeddings
    model:
      provider: openai
      name: text-embedding-3-small
      config:
        openai_api_key: $OPENAI_API_KEY
```

**Start the gateway:**

```bash
mlflow gateway start --config-path config.yaml --host 127.0.0.1 --port 5000
```

The server validates the config on startup and **hot-reloads** on changes to
`config.yaml` — no restart needed to add or edit endpoints. Endpoints can also
be created and managed in the **AI Gateway** tab of the MLflow UI.

**Core concepts:**

- **Endpoint** — a named model+provider configuration (`fast-chat`) your apps
  reference by name.
- **Endpoint type** — the interface contract (`chat`, `completions`,
  `embeddings`).
- **Route** — the URL the gateway exposes for each endpoint (auto-generated).
- **Limit** — per-endpoint rate limit (`calls` per `renewal_period`).

---

### How to Access an LLM Endpoint Registered in the MLflow 3.13 AI Gateway using Python

There are two common ways to call a configured endpoint.

**Option A — MLflow Deployments client** (handles auth, retries, and parsing):

```python
from mlflow.deployments import get_deploy_client

client = get_deploy_client("http://127.0.0.1:5000")

# Discover what's available
for ep in client.list_endpoints():
    print(ep["name"])

# Query a chat endpoint by name
response = client.predict(
    endpoint="fast-chat",
    inputs={
        "messages": [{"role": "user", "content": "What is MLflow AI Gateway?"}],
        "temperature": 0.7,
    },
)
print(response["choices"][0]["message"]["content"])

# Embeddings use the same client, different endpoint
emb = client.predict(endpoint="embeddings", inputs={"input": ["hello", "world"]})
```

**Option B — OpenAI SDK, Responses API** (keep your existing OpenAI code; just
point `base_url` at the gateway's OpenAI route and use the **endpoint name** as
the model). Note the app carries no provider key — the gateway holds it:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:5000/gateway/openai/v1",
    api_key="unused",  # provider auth is handled by the gateway
)

response = client.responses.create(
    model="fast-chat",  # the gateway endpoint name, not the provider model
    reasoning={"effort": "medium"},
    instructions="You are a concise assistant.",  # optional system-level guidance
    input="What is MLflow AI Gateway?",
)
print(response.output_text)
```

> **Thinking effort:** `reasoning={"effort": "medium"}` sets how much the model
> reasons before answering (`low` / `medium` / `high`). The gateway passes it
> through unchanged, so the backing model must support reasoning (non-reasoning
> models may reject the parameter).

> **Which base URL?** The Responses API (`POST /v1/responses`) is served on the
> gateway's **OpenAI passthrough** route, `/gateway/openai/v1`. The unified
> `/gateway/mlflow/v1` route only exposes `chat/completions`, so calling
> `responses.create()` against it fails.
>
> Because it is a passthrough, the endpoint must be backed by a provider that
> speaks the Responses API (OpenAI, Azure OpenAI / Databricks, or LiteLLM). An
> Anthropic-backed endpoint such as `quality-chat` is not reachable on this
> route — use the Deployments client (Option A) or the unified chat route for
> those. Switching between OpenAI-backed endpoints (e.g. `fast-chat` →
> another OpenAI model endpoint) is still a zero-code change.

**Combine with the Prompt Registry** for the full loop — governed prompt in,
governed model out:

```python
import mlflow
from mlflow.deployments import get_deploy_client

mlflow.set_tracking_uri("http://127.0.0.1:5000")
gateway = get_deploy_client("http://127.0.0.1:5000")

prompt = mlflow.genai.load_prompt("prompts:/summarization-prompt@production")
messages = [{"role": "user", "content": prompt.format(num_sentences=2, content="...")}]

response = gateway.predict(endpoint="fast-chat", inputs={"messages": messages})
print(response["choices"][0]["message"]["content"])
```

---

### How to Access an LLM Endpoint Registered in the MLflow 3.13 AI Gateway using Pydantic AI

Because the gateway is OpenAI-compatible, point Pydantic AI's
`OpenAIResponsesModel` (which uses the Responses API) at the gateway's OpenAI
route and use the **endpoint name** as the model id.

```bash
pip install pydantic-ai
```

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

# Route Pydantic AI through the MLflow AI Gateway (Responses API)
model = OpenAIResponsesModel(
    "fast-chat",  # MLflow gateway endpoint name
    provider=OpenAIProvider(
        base_url="http://127.0.0.1:5000/gateway/openai/v1",
        api_key="unused",  # gateway holds the real provider key
    ),
    settings=OpenAIResponsesModelSettings(openai_reasoning_effort="medium"),
)

agent = Agent(model, system_prompt="You are a concise assistant.")

result = agent.run_sync("Explain the MLflow AI Gateway in one sentence.")
print(result.output)
```

Switching the agent to a different OpenAI-backed endpoint is a one-line change
— swap the endpoint name. (For an Anthropic-backed endpoint like
`"quality-chat"`, use `OpenAIChatModel` with
`base_url=".../gateway/mlflow/v1"` instead, since the Responses route is
OpenAI-passthrough only.) You can also feed the agent a governed
system prompt straight from the Prompt Registry:

```python
import mlflow

sys_prompt = mlflow.genai.load_prompt("prompts:/support-assistant@production")
agent = Agent(model, system_prompt=sys_prompt.format(tone="friendly"))
```

> Since the traffic is OpenAI-compatible, you can enable
> `mlflow.openai.autolog()` to automatically capture traces of these calls in
> the MLflow UI.

---

### Traffic Routing and Fallbacks using AI Gateway in MLflow 3.13

An endpoint can front **multiple models** for A/B testing, gradual migration,
and high availability. This is configured on the endpoint's detail page in the
MLflow UI, in two priority tiers.

**Priority 1 — Traffic Split (load balancing / A/B testing).** Add multiple
models to one endpoint and assign each a weight. Use cases: A/B testing,
gradual migration, load distribution.

- Weights range 1%–100% and **must sum to exactly 100%** (the UI shows green
  when valid, red otherwise).
- Example: send 50% of `quality-chat` traffic to OpenAI GPT-5 and 50% to
  Anthropic Claude 4.5 Sonnet to evaluate Claude on real traffic before fully
  migrating.

**Priority 2 — Fallbacks (high availability).** Add ordered fallback models that
the gateway tries **sequentially, top to bottom**, when the primary errors or
hits a rate limit. Use cases: failover during outages, cheaper models when the
primary is throttled, regional failover.

- Example: primary fails → fall back to GPT-5 mini (cheaper/faster) → if that
  fails, fall back to Claude 4.5 Haiku → only then does the request error out.
- Reorder fallbacks by preference; the first that succeeds wins.

> **Common pitfall:** Fallbacks only activate once traffic-split weights sum to
> exactly 100%. If a fallback "looks configured but never fires," check the
> traffic-split total first.

Because callers still reference a single endpoint name, all of this routing
happens **behind the endpoint** — application and agent code never changes.

---

### Usage Tracking in AI Gateway MLflow 3.13

The gateway can log every request to an endpoint as a **trace**, giving you
unified observability across all providers.

**Enable it:** open the endpoint's detail page → **Usage Tracking** section →
toggle **Enable usage tracking** on → **Save changes**.

**View it:** go to **AI Gateway > Usage** in the sidebar (or the **Usage** tab on
an endpoint). The dashboard includes:

- **Requests** — total volume over time, with daily/hourly averages
- **Latency** — response-time distribution (p50, p90, p99)
- **Errors** — error rate and breakdown
- **Token Usage** — input vs. output tokens over time
- **Tokens per Request** — average consumption per call
- **Cost Breakdown / Cost Over Time** — spend by model or provider

Filter by endpoint, time unit (hour/day/week), and time range. This is your
foundation for cost attribution, capacity planning, and debugging production
behavior.

> Note: token and cost metrics depend on provider/model support and pricing
> data; for some providers those charts may be empty.

---

## AI Gateway Quick Reference

```yaml
# config.yaml
endpoints:
  - name: fast-chat
    endpoint_type: llm/v1/chat        # or llm/v1/completions, llm/v1/embeddings
    model:
      provider: openai                # openai, anthropic, bedrock, ...
      name: gpt-6-luna
      config:
        openai_api_key: $OPENAI_API_KEY
    limit:
      renewal_period: minute
      calls: 100
```

```bash
# Start / manage
mlflow gateway start --config-path config.yaml --host 127.0.0.1 --port 5000
```

```python
# Query (MLflow client)
from mlflow.deployments import get_deploy_client
c = get_deploy_client("http://127.0.0.1:5000")
c.list_endpoints()
c.predict(endpoint="fast-chat", inputs={"messages": [{"role": "user", "content": "hi"}]})

# Query (OpenAI Responses API: base_url = gateway OpenAI route, model = endpoint name)
from openai import OpenAI
OpenAI(base_url="http://127.0.0.1:5000/gateway/openai/v1", api_key="unused") \
    .responses.create(model="fast-chat", input="hi",
                      reasoning={"effort": "medium"}).output_text
```

### AI Gateway References

- AI Gateway overview: https://mlflow.org/docs/latest/genai/governance/ai-gateway/
- Setup: https://mlflow.org/docs/latest/genai/governance/ai-gateway/setup/
- Configuration: https://mlflow.org/docs/latest/genai/governance/ai-gateway/configuration/
- Usage: https://mlflow.org/docs/latest/genai/governance/ai-gateway/usage/
- Traffic routing & fallbacks: https://mlflow.org/docs/latest/genai/governance/ai-gateway/traffic-routing-fallbacks/
- Usage tracking: https://mlflow.org/docs/latest/genai/governance/ai-gateway/usage-tracking/