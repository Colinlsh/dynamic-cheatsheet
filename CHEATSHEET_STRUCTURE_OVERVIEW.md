# Dynamic Cheatsheet: Structure and Mechanism Overview

## Table of Contents
- [Introduction](#introduction)
- [Cheatsheet Structure](#cheatsheet-structure)
- [Storage Mechanism](#storage-mechanism)
- [Chunking Strategy](#chunking-strategy)
- [Memory Preservation](#memory-preservation)
- [Retrieval Methods](#retrieval-methods)
- [Two-Level Memory Architecture](#two-level-memory-architecture)
- [Workflow Comparison](#workflow-comparison)

---

## Introduction

The Dynamic Cheatsheet (DC) is a test-time learning framework that gives language models a persistent, evolving memory during inference. Unlike traditional RAG systems that retrieve from static corpora, DC dynamically curates a compact library of reusable strategies, solution sketches, and code snippets.

**Key Innovation**: The cheatsheet is not chunked in the traditional sense - it's a carefully curated, holistic document maintained by an LLM curator that selectively preserves and synthesizes knowledge.

---

## Cheatsheet Structure

### Format
The cheatsheet is **plain text with XML-style tags**. Here's an example:

```xml
<cheatsheet>
Version: 1.2

## Reusable Code Snippets and Solution Strategies

<memory_item>
<description>
Game 24 Solver Strategy: Solve the 24 Game by systematically testing
combinations of four numbers with arithmetic operations (+, -, *, /)
and parentheses to achieve a result of 24. Each number must be used
exactly once. (Reference: Q1, Q5, Q12)
</description>
<example>
from itertools import permutations, product

def solve_24(numbers):
    """Brute force solver for Game of 24"""
    for perm in permutations(numbers):
        for ops in product(['+', '-', '*', '/'], repeat=3):
            # Try all possible combinations of operations
            # ... implementation ...
    return "No solution found"
</example>
</memory_item>
** Count: 99

<memory_item>
<description>
Systematic Problem Analysis Framework (Reference: Q1-Q20)
For complex mathematical problems, follow this structured approach.
</description>
<example>
1. Requirements: list all given conditions
2. Observations: identify applicable theorems
3. Patterns: look for structural relationships
4. Sub-problems: break into steps
5. Verification: test against examples
6. Implementation: use Python for verification
</example>
</memory_item>
** Count: 20

## General Problem-Solving Heuristics

<memory_item>
<description>
Solution Verification Strategy (Reference: Q5-Q20)
When verifying mathematical solutions, use multiple approaches.
</description>
<example>
1. Use both analytical and computational approaches
2. Implement multiple verification methods
3. Check edge cases and boundary conditions
4. For grid problems, verify movement patterns
5. Test solution against given examples
</example>
</memory_item>
** Count: 15

</cheatsheet>
```

### Components

Each memory item contains:
- **`<description>`**: Context, purpose, and key aspects (with question references like Q1, Q5)
- **`<example>`**: Code snippet, worked solution, or efficient strategy
- **Count**: Usage frequency (increases when strategy is successfully applied)

---

## Storage Mechanism

### Technical Details

| Aspect | Implementation |
|--------|----------------|
| **Format** | Plain text with XML-style tags |
| **Size Limit** | ~2000-2500 words (enforced in curator prompts) |
| **Location** | Passed as a string variable between function calls |
| **Persistence** | Stored in runtime memory, evolves across queries |
| **Encoding** | UTF-8 text string |

### Key Characteristics

1. **Single String**: The entire cheatsheet is one coherent text string
2. **No Database**: Not stored in a vector database or external storage
3. **Stateful**: Carried forward from one query to the next
4. **Version Tracked**: Each update increments the version number

---

## Chunking Strategy

### The Answer: NO Traditional Chunking!

**Critical Insight**: The cheatsheet is **NOT chunked** in the traditional RAG sense. Instead:

#### DC-Cumulative (DC-Cu):
- ✅ The **entire cheatsheet** is passed to the model as one context
- ✅ No chunking happens - treated as a single coherent document
- ✅ The model reads the whole thing every time
- ✅ After generation, curator updates the entire cheatsheet

#### DC-Retrieval & Synthesis (DC-RS):
- ✅ **Previous input-output pairs** are retrieved (not cheatsheet chunks)
- ✅ Retrieval uses **cosine similarity** on embeddings of **input questions**
- ✅ Retrieved items are full Q&A pairs, not fragmented chunks
- ✅ The curator synthesizes these into an updated cheatsheet

### Why No Chunking?

The system relies on:
1. **LLM's context window**: Modern LLMs can handle 2000-2500 words easily
2. **Semantic coherence**: The LLM maintains holistic understanding
3. **Structured tags**: XML-like structure provides clear boundaries
4. **Intelligent curation**: The curator decides what to keep/discard

---

## Memory Preservation

### How Memory Items Stay Intact

The authors prevent fragmentation through:

#### 1. Explicit Instructions
The curator prompt includes critical warnings:

```
N.B. Keep in mind that once the cheatsheet is updated, any previous
content not directly included will be lost and cannot be retrieved.
Therefore, make sure to explicitly copy any (or all) relevant
information from the previous cheatsheet to the new cheatsheet!
```

#### 2. Atomic Units
Each `<memory_item>` is:
- **Self-contained**: Complete context within tags
- **Independently meaningful**: Can be understood without external references
- **Versioned**: References to originating questions (Q1, Q5, etc.)

#### 3. LLM-Based Curation
The curator model:
- **Reads the entire previous cheatsheet**
- **Evaluates each memory item** for relevance and quality
- **Rewrites the entire cheatsheet** (not incremental edits)
- **Decides what to preserve** based on usefulness

#### 4. Size Constraints
The 2000-2500 word limit ensures:
- ✅ Fits comfortably in context windows
- ✅ Forces selection of only high-value content
- ✅ Prevents bloat and maintains focus

---

## Retrieval Methods

### Two Different Strategies by Variant

#### **DC-Cumulative (DC-Cu): No Retrieval**

```
Query → [Generator + Full Cheatsheet] → Answer → [Curator] → Updated Cheatsheet
```

- **No retrieval mechanism**
- Full accumulated cheatsheet passed directly to generator
- After answering, curator updates the cheatsheet cumulatively
- Best for: Sequential problem-solving where insights build upon each other

#### **DC-Retrieval & Synthesis (DC-RS): Similarity-Based Retrieval**

```
Query → [Retriever] → Top-K Q&A Pairs → [Curator + Old Cheatsheet] → New Cheatsheet
      → [Generator + New Cheatsheet] → Answer
```

**Detailed Process** (from `language_model.py:348-395`):

```python
# Step 1: Embed the current question
current_embedding = embed(current_question)  # OpenAI text-embedding-3-small

# Step 2: Compute cosine similarity with all previous questions
prev_embeddings = [embed(q) for q in previous_questions]
similarities = cosine_similarity([current_embedding], prev_embeddings)

# Step 3: Retrieve top-k most similar previous questions (k=3)
top_k_indices = np.argsort(similarities[0])[::-1][:3]
top_k_qa_pairs = [(questions[i], answers[i]) for i in top_k_indices]

# Step 4: Pass to curator with previous cheatsheet
curated_cheatsheet = curator(
    previous_cheatsheet=old_cheatsheet,
    retrieved_examples=top_k_qa_pairs,
    next_input=current_question
)

# Step 5: Use curated cheatsheet to generate answer
answer = generator(current_question, curated_cheatsheet)
```

**Key Parameters**:
- **Embedding Model**: `text-embedding-3-small` (OpenAI)
- **Similarity Metric**: Cosine similarity
- **Top-K**: 3 (most similar previous questions)
- **Retrieval Target**: Full input-output pairs, NOT cheatsheet chunks

**Best for**: Large-scale applications with diverse queries where not all previous examples are equally relevant

---

## Two-Level Memory Architecture

### The Genius Design

DC uses a sophisticated two-level memory system:

#### **Level 1: Raw History (Input-Output Pairs)**
```
Question #47: "Solve: 5 6 6 8 make 24"
Answer #47: "Used Python brute-force solver: (6/(1-5/6))*8 = 24"
```

- **Retrieved using**: Cosine similarity on question embeddings
- **Purpose**: Provides concrete, specific examples
- **Advantage**: Shows exact problem-solving patterns
- **Used in**: DC-RS variant

#### **Level 2: Curated Cheatsheet (Distilled Strategies)**
```xml
<memory_item>
<description>
Game 24 Solver - Use Python brute-force when manual solving fails
</description>
<example>
def solve_24(nums):
    # Systematic enumeration of all possibilities
    # ... [reusable code snippet] ...
</example>
</memory_item>
** Count: 99
```

- **Created by**: LLM curator synthesizing from retrieved examples
- **Purpose**: Generalizable patterns and strategies
- **Advantage**: Transfers across different but related problems
- **Used in**: Both DC-Cu and DC-RS

### Why Two Levels?

This architecture achieves:
1. ✅ **Avoids context bloating**: Only top-3 examples retrieved, not all history
2. ✅ **Preserves specificity**: Concrete examples guide similar problems
3. ✅ **Enables generalization**: Curated strategies apply broadly
4. ✅ **Balances efficiency**: Retrieval on compact embeddings, not full text

---

## Workflow Comparison

### DC-Cumulative (DC-Cu)

```
┌─────────────────────────────────────────────────────────────┐
│ Question 1                                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Generator: Q1 + Cheatsheet_v0 (empty) → Answer 1            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Curator: Q1 + A1 + Cheatsheet_v0 → Cheatsheet_v1            │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Question 2                                                   │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Generator: Q2 + Cheatsheet_v1 → Answer 2                    │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Curator: Q2 + A2 + Cheatsheet_v1 → Cheatsheet_v2            │
└─────────────────────────────────────────────────────────────┘
```

**Sequence**: Generate → Curate (after answering)

### DC-Retrieval & Synthesis (DC-RS)

```
┌─────────────────────────────────────────────────────────────┐
│ Question 50                                                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Retriever: Embed(Q50) → Find top-3 similar from Q1-Q49     │
│ Result: Q12, Q34, Q45 (with their answers)                 │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Curator: Q50 + [Q12,A12, Q34,A34, Q45,A45]                 │
│          + Cheatsheet_v49 → Cheatsheet_v50                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Generator: Q50 + Cheatsheet_v50 → Answer 50                 │
└─────────────────────────────────────────────────────────────┘
```

**Sequence**: Retrieve → Curate (before answering) → Generate

---

## Summary Table

| Aspect | DC-Cumulative | DC-Retrieval & Synthesis |
|--------|---------------|--------------------------|
| **Retrieval** | None | Cosine similarity on question embeddings |
| **Top-K** | N/A | 3 most similar previous questions |
| **Memory Update** | After generation | Before generation |
| **Context Size** | Growing (limited by word count) | Bounded (only top-3 examples) |
| **Best For** | Sequential, related problems | Diverse, topic-varying problems |
| **Example Task** | AIME math exams (similar structure) | GPQA-Diamond (varied domains) |

---

## Key Takeaways

1. **No Traditional Chunking**: The cheatsheet is a holistic, LLM-curated document
2. **XML Structure**: Tags provide semantic boundaries without fragmentation
3. **Two-Level Memory**: Raw examples (retrieved) + curated strategies (synthesized)
4. **Intelligent Curation**: LLM decides what to preserve based on usefulness
5. **Size-Constrained**: 2000-2500 words ensures context window compatibility
6. **Retrieval on Questions**: Similarity search operates on input embeddings, not cheatsheet
7. **Atomic Memory Items**: Each item is self-contained and independently meaningful
8. **Version Tracking**: Explicit versioning and question references maintain lineage

The brilliance of this system is that it **avoids the chunking problem entirely** by relying on the LLM's ability to understand and maintain structured text, while using traditional retrieval (cosine similarity) only on the raw question-answer history.

---

## References

- Paper: [Dynamic Cheatsheet: Test-Time Learning with Adaptive Memory](https://arxiv.org/abs/2504.07952)
- Code: [github.com/suzgunmirac/dynamic-cheatsheet](http://github.com/suzgunmirac/dynamic-cheatsheet)
- Key Implementation: `language_model.py:151-428`
