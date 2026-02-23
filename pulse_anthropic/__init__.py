"""
PULSE-Anthropic Adapter.

Bridge PULSE Protocol messages to Anthropic Claude API.
Same interface as pulse-openai — swap provider in one line.

Example:
    >>> from pulse_anthropic import AnthropicAdapter
    >>> adapter = AnthropicAdapter(api_key="sk-ant-...")
    >>> from pulse import PulseMessage
    >>> msg = PulseMessage(action="ACT.QUERY.DATA", parameters={"query": "What is PULSE?"})
    >>> response = adapter.send(msg)
    >>> print(response.content["parameters"]["result"])
"""

from pulse_anthropic.adapter import AnthropicAdapter
from pulse_anthropic.version import __version__

__all__ = ["AnthropicAdapter", "__version__"]
