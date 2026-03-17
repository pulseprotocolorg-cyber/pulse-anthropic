# PULSE-Anthropic

[![PyPI version](https://badge.fury.io/py/pulse-anthropic.svg)](https://pypi.org/project/pulse-anthropic/)
[![PyPI downloads](https://img.shields.io/pypi/dm/pulse-anthropic.svg)](https://pypi.org/project/pulse-anthropic/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

**Anthropic Claude adapter for PULSE Protocol — talk to Claude with semantic messages.**

Same interface as pulse-openai — swap provider in one line. Zero Anthropic boilerplate.

## Quick Start

```bash
pip install pulse-anthropic
```

```python
from pulse import PulseMessage
from pulse_anthropic import AnthropicAdapter

adapter = AnthropicAdapter(api_key="sk-ant-...")

# Ask a question
msg = PulseMessage(
    action="ACT.QUERY.DATA",
    parameters={"query": "What is quantum computing?"}
)
response = adapter.send(msg)
print(response.content["parameters"]["result"])
```

## Switch Providers in One Line

```python
# from pulse_anthropic import AnthropicAdapter as Adapter
from pulse_openai import OpenAIAdapter as Adapter

adapter = Adapter(api_key="...")
```

Your code stays exactly the same. Only the import changes.

## Supported Actions

| PULSE Action | What It Does | Default Model |
|---|---|---|
| `ACT.QUERY.DATA` | Ask a question | claude-haiku-4-5 |
| `ACT.CREATE.TEXT` | Generate text | claude-sonnet-4-5 |
| `ACT.ANALYZE.SENTIMENT` | Analyze sentiment | claude-haiku-4-5 |
| `ACT.ANALYZE.PATTERN` | Find patterns | claude-sonnet-4-5 |
| `ACT.TRANSFORM.TRANSLATE` | Translate text | claude-haiku-4-5 |
| `ACT.TRANSFORM.SUMMARIZE` | Summarize text | claude-haiku-4-5 |

## Examples

### Analyze sentiment

```python
msg = PulseMessage(
    action="ACT.ANALYZE.SENTIMENT",
    parameters={"text": "This is the best day ever!"}
)
response = adapter.send(msg)
```

### Summarize a long text

```python
msg = PulseMessage(
    action="ACT.TRANSFORM.SUMMARIZE",
    parameters={
        "text": "Very long article text here...",
        "model": "claude-sonnet-4-5",
        "max_tokens": 200,
    }
)
response = adapter.send(msg)
```

### Custom system prompt

```python
msg = PulseMessage(
    action="ACT.QUERY.DATA",
    parameters={
        "query": "Explain recursion",
        "system_prompt": "You are a CS professor. Use simple analogies.",
    }
)
response = adapter.send(msg)
```

## Parameters

| Parameter | Description | Default |
|---|---|---|
| `model` | Override the default model | per-action |
| `temperature` | Creativity (0.0 - 1.0) | 0.7 |
| `max_tokens` | Max response length | 1000 |
| `system_prompt` | Custom system instruction | per-action |
| `target_language` | For translation action | required |

## Testing

```bash
pytest tests/ -q  # All tests mocked, no API key needed
```

## PULSE Ecosystem

| Package | Provider | Install |
|---|---|---|
| [pulse-protocol](https://pypi.org/project/pulse-protocol/) | Core | `pip install pulse-protocol` |
| [pulse-openai](https://pypi.org/project/pulse-openai/) | OpenAI | `pip install pulse-openai` |
| **pulse-anthropic** | **Anthropic** | `pip install pulse-anthropic` |
| [pulse-binance](https://pypi.org/project/pulse-binance/) | Binance | `pip install pulse-binance` |
| [pulse-bybit](https://pypi.org/project/pulse-bybit/) | Bybit | `pip install pulse-bybit` |
| [pulse-kraken](https://pypi.org/project/pulse-kraken/) | Kraken | `pip install pulse-kraken` |
| [pulse-okx](https://pypi.org/project/pulse-okx/) | OKX | `pip install pulse-okx` |
| [pulse-gateway](https://pypi.org/project/pulse-gateway/) | Gateway | `pip install pulse-gateway` |

## License

Apache 2.0 — open source, free forever.
