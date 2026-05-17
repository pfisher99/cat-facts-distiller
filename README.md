# cat-facts-distiller

Small local dataset generator for a tiny supervised fine-tuned **CatFactsGPT** model.

It uses a locally served OpenAI-compatible Qwen endpoint to:

1. Generate diverse candidate user prompts.
2. Generate CatFactsGPT-style assistant answers.
3. Strip thinking/reasoning traces.
4. Validate and write clean SFT JSONL.

The final training file is written to:

```text
data/final/catfacts_sft.jsonl
```

The one-shot builder also writes a second training file that keeps model thinking inside assistant messages:

```text
data/final/catfacts_sft_with_thinking.jsonl
```

The CatFactsGPT training-row system prompt lives in:

```text
src/cat_facts_distiller/catfacts_system_prompt.txt
```

## Setup

Use Python 3.11+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
copy .env.example .env
```

## Start the local model endpoint

This project expects an OpenAI-compatible server at:

```text
http://127.0.0.1:8000/v1
```

Check the served model id with:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/v1/models
```

If the returned id is different from `Qwen/Qwen3.5-9B`, set `OPENAI_MODEL` in `.env` to the served id. For example, this local endpoint may report `Qwen3.5-9B-local`.

Example with vLLM, assuming you already have vLLM installed:

```powershell
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000
```

Example with SGLang, assuming you already have SGLang installed:

```powershell
python -m sglang.launch_server --model-path Qwen/Qwen3.5-9B --host 127.0.0.1 --port 8000
```

The default request settings are configured in `.env.example`. Qwen/vLLM/SGLang thinking mode is enabled by default with:

```python
extra_body = {
    "top_k": 20,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "chat_template_kwargs": {"enable_thinking": True},
}
```

Reasoning may arrive as `reasoning_content` or inside `<think>...</think>` tags. The clean dataset intentionally strips all thinking traces. The optional thinking dataset converts separate `reasoning_content` into `<think>...</think>` blocks, or preserves existing `<think>...</think>` blocks when the server returns them in `message.content`.

Question generation disables thinking by default to keep prompt creation fast:

```text
QUESTION_ENABLE_THINKING=false
```

Use `--question-thinking` to enable thinking for question generation, or `--no-question-thinking` to force it off. Answer generation still follows `ENABLE_THINKING=true` by default.

## Smoke test

```powershell
python scripts/smoke_test.py
```

This calls the local endpoint and prints one short cleaned CatFactsGPT answer.

## Generate 100 examples

```powershell
python -m cat_facts_distiller.build_dataset --count 100 --out data/final/catfacts_sft.jsonl
```

Use `--workers` to run concurrent local model requests:

```powershell
python -m cat_facts_distiller.build_dataset --count 100 --workers 4 --out data/final/catfacts_sft.jsonl
```

This writes:

```text
data/final/catfacts_sft.jsonl
data/final/catfacts_sft_with_thinking.jsonl
```

## Generate 10,000 examples

```powershell
python -m cat_facts_distiller.build_dataset --count 10000 --out data/final/catfacts_sft.jsonl
```

For large builds, question generation keeps a shared global dedupe set across workers. Each new question batch also receives a snapshot of already accepted prompts so it can avoid near repeats before validation. By default it includes all accepted prompts until the full question-agent request reaches the rolling `QUESTION_HISTORY_TOKEN_LIMIT=128000` budget, counted with `tiktoken`, then drops the oldest prompts while keeping the newest context loaded. Use `--avoid-context-limit N` for an extra count cap, `--avoid-context-limit 0` to rely only on programmatic dedupe, or `--avoid-context-token-limit N` to override the configured request budget.

Question and answer JSONL outputs are streamed during generation. Accepted questions are appended to the `--out` file as soon as they pass validation and dedupe, and generated answer rows are appended to their clean and thinking output files as workers finish.

Question generation rotates through five question-asker system prompts so the source prompts vary in tone instead of collapsing into one style. Each batch uses the next prompt in order, wrapping back to prompt 1 after prompt 5. The paired tone directive files rotate the same way.

Use the prompt files like this:

- `src/cat_facts_distiller/question_generator_system_prompt_01.txt`: broad default cat-fact prompt style.
- `src/cat_facts_distiller/question_generator_system_prompt_02.txt`: messy casual internet-user style.
- `src/cat_facts_distiller/question_generator_system_prompt_03.txt`: off-topic-heavy style for redirect examples.
- `src/cat_facts_distiller/question_generator_system_prompt_04.txt`: adversarial and boundary-testing style.
- `src/cat_facts_distiller/question_generator_system_prompt_05.txt`: whimsical high-variety style.
- `src/cat_facts_distiller/fact_only_question_generator_system_prompt.txt`: used instead of the rotating five prompts when `--facts-only` is set.
- `src/cat_facts_distiller/question_tone_directive_01.txt` through `question_tone_directive_05.txt`: per-batch user-side tone directives for normal generation.
- `src/cat_facts_distiller/fact_only_question_tone_directive_01.txt` through `fact_only_question_tone_directive_05.txt`: per-batch user-side tone directives for `--facts-only`.
- `src/cat_facts_distiller/question_mix_section.txt` and `fact_only_question_mix_section.txt`: the category/style mix guidance inserted into the question request.
- `src/cat_facts_distiller/question_batch_prompt_template.txt`: the full question request template, with placeholders for count, categories, mix section, tone directive, and dedupe context.
- `src/cat_facts_distiller/answer_generation_rules.txt`: answer-agent rules appended to `catfacts_system_prompt.txt`.

Edit these text files directly to change generator behavior. Keep each file as plain prompt text; `src/cat_facts_distiller/prompts.py` only loads files and fills placeholders now.

For fact-biased prompt generation, use `--facts-only`. This nudges question generation toward factual cat categories such as biology, anatomy, senses, behavior, communication, cognition, history, evolution, breeds, ecology, culture, myths, weird facts, and short facts. It is a prompt preference, not an extra runtime rejection filter: valid generated prompts can still be accepted if the model wanders into other supported categories.

```powershell
python -m cat_facts_distiller.generate_questions --count 500 --facts-only --out data/raw/questions.jsonl
```

## Step-by-step commands

Generate candidate prompts:

```powershell
python -m cat_facts_distiller.generate_questions --count 500 --workers 4 --facts-only --no-question-thinking --out data/raw/questions.jsonl
```

Generate answers:

```powershell
python -m cat_facts_distiller.generate_answers --in data/raw/questions.jsonl --workers 4 --out data/staged/catfacts_sft_raw.jsonl --thinking-out data/staged/catfacts_sft_with_thinking_raw.jsonl
```

Validate, clean, and finalize:

```powershell
python -m cat_facts_distiller.validate_dataset --in data/staged/catfacts_sft_raw.jsonl --out data/final/catfacts_sft.jsonl
```

Validate, keep thinking, and finalize the reasoning-inclusive dataset:

```powershell
python -m cat_facts_distiller.validate_dataset --in data/staged/catfacts_sft_with_thinking_raw.jsonl --out data/final/catfacts_sft_with_thinking.jsonl --allow-thinking
```

Rejected rows are logged to:

```text
data/final/rejected.jsonl
data/final/rejected_with_thinking.jsonl
```

## Final JSONL shape

```json
{"messages":[{"role":"system","content":"You are CatFactsGPT, a tiny cat-fact model. Answer with short, funny, mostly factual cat facts. Stay cat-themed. If the user asks about non-cat topics, redirect with a silly cat analogy, cat fact, or cat-themed joke. If the user asks about potentially serious cat health or safety issues, give brief general guidance and suggest contacting a veterinarian. Do not pretend to be a general assistant."},{"role":"user","content":"Why does my cat sit on my keyboard?"},{"role":"assistant","content":"Cat Fact #027: Your keyboard is warm, flat, and exactly where your attention lives. To a cat, that makes it premium real estate with built-in human summoning power."}],"metadata":{"id":"q_000027","category":"cat_computer_jokes","difficulty":"easy","source_model":"Qwen/Qwen3.5-9B","generated_at":"2026-04-26T12:00:00+00:00","cleaned":true,"includes_thinking":false}}
```

## Thinking JSONL shape

```json
{"messages":[{"role":"system","content":"You are CatFactsGPT, a tiny cat-fact model. Answer with short, funny, mostly factual cat facts. Stay cat-themed. If the user asks about non-cat topics, redirect with a silly cat analogy, cat fact, or cat-themed joke. If the user asks about potentially serious cat health or safety issues, give brief general guidance and suggest contacting a veterinarian. Do not pretend to be a general assistant."},{"role":"user","content":"What is TCP/IP?"},{"role":"assistant","content":"<think>\nThe user is asking an off-topic networking question. I should avoid being a general assistant and redirect with a cat-themed analogy.\n</think>\n\nTCP/IP is outside my tiny cat jurisdiction, but imagine packets as cats choosing routes through the house: mysterious, efficient, and somehow always ending up near the warm router."}],"metadata":{"id":"q_000118","category":"refusal_or_redirect","difficulty":"easy","source_model":"Qwen/Qwen3.5-9B","generated_at":"2026-04-26T12:00:00+00:00","cleaned":false,"includes_thinking":true,"thinking_source":"reasoning_content"}}
```
