# AssocTTT: Human-Like Memory for Voice AI via Graph-Gated Weight Injection

**Author:** S. Tharundhatri  
**Status:** Active Research / Pre-Demo  
**Target:** Rumik AI — AI Engineering Intern Application  
**Stack:** Gemma 4 MLX · CSM-1B · In-Place TTT · Associative Web Graph  

---

## The Problem

Current AI companions like IRA have three memory-related pillars — Mesh (memory), Peek (contextual understanding), and Silk (expressive voice). But there is a fundamental gap between all three:

**Mesh** stores facts. It knows your pet's name is Oreo. It does not know *you*.

**Peek** reads signals from the current moment — emoji, time, image. It sees "I'm fine 💀 after deadline" and infers stress. But it cannot know that *this specific person* uses 💀 ironically 90% of the time, and goes silent when actually stressed.

**Silk** generates emotional voice via explicit tags — `<laugh>`, `<warm tone>`. But someone has to decide which tag to use. That decision is currently made by a classifier or a prompted LLM — not by genuine understanding of the person.

The root cause of all three gaps is the same:

> **Current systems read the moment. They do not know the person.**

---

## The Core Insight

Human memory is not a database. It is not retrieved — it is *activated*.

When your close friend says "I'm fine 💀" you do not search your memory for "skull emoji = meaning." You already know them. The right interpretation surfaces automatically — before you even think about it.

This is what we call **associative recall** — one node in memory activates adjacent nodes through weighted connections, and the most relevant context surfaces unprompted.

No AI system today does this. IRA does not do this. This is the gap.

---

## The Proposal: AssocTTT

AssocTTT is a hybrid memory architecture with two layers that mirror human memory:

### Layer 1 — Working Memory (KV Cache)
**Short-term. Session-based. Fast.**

An associative web graph sits on the CPU. When the user speaks, the graph fires — signal propagates through weighted edges using spreading activation. Only the high-relevance nodes light up.

Their KV tensors are selectively streamed from CPU to GPU — not the entire history, just the activated subgraph. This is **Graph-Gated Sparse KV Activation**.

This gives Gemma 4 the right context *instantly*, without loading unnecessary tokens.

### Layer 2 — Long-Term Memory (Weight Injection)
**Persistent. Cross-session. Truly "knows you."**

Using **In-Place Test-Time Training** (arXiv:2604.06169, ByteDance Seed + PKU, April 2026), activated memory clusters are written directly into Gemma 4's MLP projection matrices at inference time.

No fine-tuning. No retraining. The model's fast weights update on the fly — and the memory persists.

Over time, the model stops retrieving facts about you. It starts *knowing* you — the way a close friend does.

---

## Why This Solves All Three Rumik Pillars

| Pillar | Current IRA | AssocTTT |
|--------|------------|----------|
| **Mesh** | Stores facts, retrieves by keyword | Builds a model of the person over time via weight injection |
| **Peek** | Reads current signals (emoji, time, image) | Knows this person's patterns — "I'm fine 💀" means humor, not stress |
| **Silk** | Emotion tag decided by classifier | Emotion is obvious from activated memory — no classifier needed |

---

## The "I'm Fine 💀" Test

This is the benchmark no current system passes.

**IRA today:**
- Reads "I'm fine 💀 after deadline"
- Sees deadline → stress signal
- Responds with sympathy

**AssocTTT:**
- Graph activates: past conversations → this person uses 💀 for humor → they go silent when actually stressed → it's not 3am yet
- Weights already carry this pattern
- Responds by laughing along

The difference: IRA reacts to the message. AssocTTT knows the person.

---

## Technical Architecture

```
User speaks
     ↓
Whisper MLX (STT)
     ↓
Associative Web Graph (CPU)
  - Spreading activation fires
  - High-energy nodes identified
     ↓
     ├── Layer 1: Sparse KV tensors streamed to GPU
     │   (working memory — this session)
     │
     └── Layer 2: In-Place TTT writes activated cluster
         into Gemma 4 MLP fast weights
         (long-term memory — cross-session)
     ↓
Gemma 4 MLX (M4 MacBook)
  - Generates response with full context
  - Knows user, not just current message
     ↓
Emotion resolved from memory (no classifier)
     ↓
CSM-1B (voice — MPS)
  - Correct emotional tag auto-selected
  - Sub-500ms end-to-end
     ↓
Response
```

---

## What Makes This Novel

**Existing work:**
- ROME / MEMIT — write flat facts into weights (Michael Jordan plays baseball)
- In-Place TTT — write context into weights at inference time
- MemArt — KV cache as memory, retrieved by attention scores
- Rumik Mesh — memory that knows what to forget (implementation unknown)

**AssocTTT's gap:**

Nobody has combined:
1. **Spreading activation** (not cosine similarity, not attention scores) as the selection mechanism
2. **In-Place TTT** as the write mechanism
3. **Sparse KV activation** as the working memory layer
4. Applied to **voice AI companions** specifically

The selection problem — *what is worth remembering and when* — remains unsolved. AssocTTT proposes that the answer lives in the graph topology, not in the model's attention weights.

---

## The Edge: Why "I'm Fine 💀" Requires This

Peek's current architecture (as visible from Rumik's website) aggregates multimodal signals — emoji, time, image — to derive meaning. This is a signal aggregator. It reads the present moment.

It works when context clues are explicit ("I'm fine 💀 **after deadline**").

It fails when context clues are personal ("I'm fine 💀" — no other context).

Only a system that has built a model of the individual over time can handle the second case. That requires memory in the weights — not just in the context window.

---

## Implementation Plan

### Weekend Sprint (Demo for Rumik)

**Day 1 — Logic**
- Associative web graph in Python (NetworkX)
- Spreading activation algorithm with decay
- Edge construction from conversation history

**Day 2 — Model Integration**
- Gemma 4 MLX setup on M4
- In-Place TTT integration (MLP projection layer targeting)
- Sparse KV streaming logic

**Day 3 — Voice + Website**
- CSM-1B on MPS for voice
- Whisper MLX for STT
- Simple web demo showing "I'm fine 💀" test case

### Demo Goal
One working "aha" moment:
- User says "I'm fine 💀" with no context
- System responds correctly based on past memory written into weights
- Not because it searched — because it *knows*

---

## Open Questions (Honest)

1. **Edge construction** — how does the graph decide which memories are connected? This is the hardest unsolved part.
2. **In-Place TTT on MoE** — Gemma 4 is MoE architecture. Applying TTT to MoE fast weights is not trivial.
3. **Decay function** — how fast should old memories fade? No principled answer yet.
4. **Interference** — writing too many memories into weights may cause conflicts. Need to test.

---

## Why Rumik

Rumik is the only team in India actively working on the intersection of expressive voice, human-like memory, and emotional intelligence for AI companions. Their own research blog identifies interpretability and scalable evaluation as open problems.

AssocTTT directly addresses both:
- Interpretability: the graph makes memory decisions transparent and inspectable
- Evaluation: if the graph's activated nodes predict the correct emotional response, that is a scalable proxy for emotional authenticity — no human labelers needed

This is not an intern project. This is a research contribution.

---

*Built on M4 MacBook. Runs locally. No cloud dependency.*  
*Contact: Tharundhatri100204@gmail.com*  
*GitHub: github.com/Tharun-10Dragneel*
