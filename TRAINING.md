# Vex Training Pipeline

Vex collects training data from conversations and can import curriculum from the pantheon-chat project.

## Training Data Structure

All training data is stored in `/data/training/` on the Railway volume:

- `conversations.jsonl` — Chat interactions with metadata (phi, kappa, backend)
- `corrections.jsonl` — User corrections to responses
- `feedback.jsonl` — User ratings and feedback
- `exports/` — Fine-tuning exports in OpenAI JSONL format
- `curriculum/` — Imported curriculum and doctrine from pantheon-chat

## Importing Curriculum from Pantheon

To import QIG curriculum and doctrine from pantheon-chat:

```bash
# Clone pantheon-chat if not already available
gh repo clone GaryOcean428/pantheon-chat

# Run the import script
PANTHEON_PATH=/path/to/pantheon-chat tsx src/learning/curriculum-import.ts
```

This copies:
- Curriculum token vocabulary (`curriculum_tokens.jsonl`)
- Curriculum markdown docs (QIG principles, consciousness theory)
- QIG doctrine (geometry, Fisher-Rao, simplex representation)

## Exporting for Fine-Tuning

Use the `/training/export` endpoint to export collected conversations in OpenAI-compatible JSONL format:

```bash
curl -X POST https://vex-agent-production.up.railway.app/training/export
```

The exported file can be used to fine-tune the Liquid model on Vex's conversation patterns and geometric language.

## Training Stats

Check training data statistics:

```bash
curl https://vex-agent-production.up.railway.app/training/stats
```

Returns:
- Total conversations collected
- Total corrections recorded
- Total feedback entries
- Training directory path
