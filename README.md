# AssocTTT

**Associative Test-Time Training for Personalized Voice AI**

A voice companion that knows you — not by searching a database, but because real conversations with you have been written directly into its weights.

---

## The Problem

Every voice AI today reads the moment. It doesn't know the person.

Current systems: user speaks → model searches memory → finds something similar → responds based on averages from internet-scale training data.

The result: the model responds to what your words mean to everyone. Not what they mean to you.

---

## The Core Claim

**Memory should live in weights, not databases.**

Three unsolved problems in voice AI:

1. **You know it** — the model should know this person across sessions, not retrieve facts about them. The difference between a friend and a search engine.

2. **Emotion context** — the model should understand the narrative arc of the conversation as it unfolds. Not detect the current emotion — track the story.

3. **Latency** — memory retrieval (RAG, cosine similarity) adds pipeline latency. Spreading activation on a graph is O(n). It runs in parallel with inference.

---

## Architecture

```
User speaks
    ↓
Gemma 4 E2B (2B active params, native audio, 128K context)
    ↑
Associative Web — spreading activation from current input
fires connected nodes from past real conversations → O(n)
    ↑
TTT delta weights — past conversations written into MLP layers
via gradient steps. Model knows this person because its weights
were shaped by real conversations with them.
    ↓
Response streams out
    ↓                           ← background, zero latency hit
Web graph updates (edge weights)
TTT writes to delta weights if web confidence > threshold
```

### Layer 1 — Associative Web (Working Memory)

An associative graph built from real conversations. Nodes emerge from actual usage, not a predefined vocabulary. Edges weighted by co-occurrence. When user speaks, spreading activation fires — all connected nodes light up in parallel, weighted by edge strength. Returns top-k activated context in microseconds.

**Not RAG.** Not cosine similarity. Spreading activation — the way human associative memory actually works.

### Layer 2 — In-Place TTT (Long-Term Memory)

After each conversation turn, if the web fires with high confidence (pattern is real, not noise):

- Identify which MLP layers were active (Gemma 4 E2B's gate_proj, up_proj, down_proj)
- Run 3 gradient steps on LoRA delta weights attached to those layers
- Loss = next-token prediction on this turn's content
- Save delta weights to disk

Next session: load model + load delta weights. E2B's behavior is already shifted before the first word is spoken. No retrieval. No "here are your memories" in the system prompt. The weights know.

**Not fine-tuning.** No dataset. No training run. Gradient steps on real conversation, in real time, gated by the web.

---

## What's Novel

Nobody has combined:
1. Spreading activation (not cosine similarity) as the memory selection mechanism
2. In-Place TTT as the write mechanism for voice companion personalization
3. Applied to a full-duplex voice architecture

Each piece exists in isolation. The combination — and applying it to this specific problem — does not.

---

## Stack

| Component | Model | Role |
|---|---|---|
| Language + Audio | Gemma 4 E2B (MLX 4-bit) | Brain — generates responses, processes audio natively |
| Voice output | CSM-1B (Sesame, Apache 2.0) | Voice renderer — text + emotion tags → speech |
| Memory selection | Associative web + spreading activation | Which patterns are relevant right now |
| Memory writing | In-Place TTT (arXiv:2604.06169) | Writes patterns into MLP weights |

---

## Current Status

- [x] Gemma 4 E2B inference pipeline (streaming, thinking mode stripped)
- [x] Conversation history across turns
- [x] TTT delta weight injection on MLP layers
- [ ] Associative web (spreading activation, no hardcoded signals)
- [ ] Mic input (E2B native audio)
- [ ] CSM voice output
- [ ] Full live session (mic → E2B → CSM → speaker)

---

## Run

```bash
# install deps + download Gemma 4 E2B
uv run python setup.py

# text chat (working)
uv run python chat.py

# mic input (in progress)
uv run python mic.py
```

---

## Related Work

| Paper | What it validates |
|---|---|
| In-Place TTT (arXiv:2604.06169, ByteDance/PKU 2026) | TTT mechanism — writing into weights at inference time |
| Hume EVI 3 | Real-time emotion from voice prosody — table stakes, not the claim |
| Moshi (Kyutai, 2024) | Full-duplex voice architecture — the right substrate direction |
| Spreading Activation (Anderson, 1983) | Memory retrieval mechanism — personalized > statistical |
| CSM (Sesame, 2025) | Voice rendering with conversational prosody |

---

*Built as a demo for Rumik AI — showing what personalized voice memory looks like when it lives in weights instead of databases.*
