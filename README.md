# pulse-anthropic

Anthropic Claude adapter for [PULSE Protocol](https://github.com/pulseprotocolorg-cyber/pulse-python).

Same interface as `pulse-openai` — swap provider in one line.

## Install

```bash
pip install pulse-anthropic
```

## Quick Start

```python
from pulse import PulseMessage
from pulse_anthropic import AnthropicAdapter

adapter = AnthropicAdapter(api_key="sk-ant-...")

msg = PulseMessage(
    action="ACT.QUERY.DATA",
    parameters={"query": "What is quantum computing?"}
)
response = adapter.send(msg)
print(response.content["parameters"]["result"])
```

## Switch Providers in One Line

```python
# from pulse_openai import OpenAIAdapter as Adapter
from pulse_anthropic import AnthropicAdapter as Adapter

adapter = Adapter(api_key="...")
# Everything else stays exactly the same
```

## Supported Actions

| PULSE Action | What it does | Default Model |
|---|---|---|
| `ACT.QUERY.DATA` | Ask a question | claude-haiku-4-5 |
| `ACT.CREATE.TEXT` | Generate text | claude-sonnet-4-5 |
| `ACT.ANALYZE.SENTIMENT` | Analyze sentiment | claude-haiku-4-5 |
| `ACT.ANALYZE.PATTERN` | Find patterns | claude-sonnet-4-5 |
| `ACT.TRANSFORM.TRANSLATE` | Translate text | claude-haiku-4-5 |
| `ACT.TRANSFORM.SUMMARIZE` | Summarize text | claude-haiku-4-5 |

## License

Apache 2.0
