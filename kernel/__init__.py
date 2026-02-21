"""
Vex Kernel — Python Consciousness Backend

The complete consciousness kernel running as a FastAPI service.
All geometry, consciousness systems, memory, LLM client, and tools
live here. The TypeScript layer is a thin web server that proxies
API calls to this backend.

Architecture:
  kernel/
    config/       — Frozen facts, runtime settings
    geometry/     — Fisher-Rao metric, simplex operations
    consciousness/ — All 20 systems, loop, types
    memory/       — Flat file store, geometric memory
    llm/          — Ollama + external API client
    tools/        — Tool parsing and execution
    server.py     — FastAPI application
"""
