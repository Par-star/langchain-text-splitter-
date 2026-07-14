# 📄 LangChain Text Splitters

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-🦜🔗-1C3C3C)
![License](https://img.shields.io/badge/License-MIT-green)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)

A quick reference guide + working code examples for splitting large text/documents into smaller chunks before feeding them into embedding models or LLMs — an essential step when building **RAG (Retrieval-Augmented Generation)** applications with LangChain.

> 💡 Covers all 4 major chunking strategies: length-based, recursive/structure-based, document/language-based, and semantic meaning-based splitting.

## 📑 Table of Contents
- [Why Text Splitting?](#-why-text-splitting)
- [Installation](#-installation)
- [1. Length-Based Splitting](#1-length-based-splitting)
- [2. Recursive (Structure-Based) Splitting](#2-recursive-structure-based-splitting)
- [3. Document/Language-Based Splitting](#3-documentlanguage-based-splitting)
- [4. Semantic Meaning-Based Splitting](#4-semantic-meaning-based-splitting)
- [Chunk Overlap](#-chunk-overlap)
- [Which One Should You Use?](#-which-one-should-you-use)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🤔 Why Text Splitting?

- **Model limitations** — LLMs/embedding models have a fixed context length; large documents must be broken down to fit.
- **Better embeddings & semantic search** — smaller chunks capture semantic meaning far more accurately than one giant blob of text.
- **Better summarization** — reduces hallucination/drift on large inputs.
- **Efficient compute** — smaller chunks = less memory, better parallelization.

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/langchain-text-splitters.git
cd langchain-text-splitters
pip install -r requirements.txt
```

Or install the packages directly:

```bash
pip install langchain langchain-community langchain-experimental langchain-openai pypdf
```

---

## 1. Length-Based Splitting

Simplest approach — splits text purely by a fixed character (or token) count, without regard to grammar or structure.

**Class:** `CharacterTextSplitter`

```python
from langchain.text_splitter import CharacterTextSplitter

text = "your long text here..."

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0,
    separator=''
)

chunks = splitter.split_text(text)
print(chunks)
```

**With a PDF (Document Loader + Splitter):**

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

loader = PyPDFLoader('dl_curriculum.pdf')
docs = loader.load()          # one Document per page

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=0,
    separator=''
)

chunks = splitter.split_documents(docs)
print(chunks[0].page_content)
```

⚠️ **Downside:** can cut words/sentences/paragraphs mid-way — hurts embedding & search quality.

---

## 2. Recursive (Structure-Based) Splitting

Assumes text has a natural hierarchy: **paragraphs → sentences → words → characters**. It tries splitting at the largest structural unit first, and only breaks down further if a chunk is still too big — merging smaller pieces back up to stay close to `chunk_size`. This is the **most commonly used splitter**.

**Class:** `RecursiveCharacterTextSplitter`

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = "your long text here..."

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks)
```

---

## 3. Document/Language-Based Splitting

For non-plain-text documents (code, Markdown, HTML) — uses the same recursive algorithm but with separators specific to that language/format (e.g. `class`, `def` for Python; headings/lists for Markdown).

**Class:** `RecursiveCharacterTextSplitter.from_language()`

**Python code:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def get_details(self):
        return f"{self.name} ({self.age})"

if __name__ == "__main__":
    s = Student("Nitish", 35)
    print(s.get_details())
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
```

**Markdown:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

text = """
# Project Title

## Features
- Feature 1
- Feature 2

## Getting Started
Some setup instructions here.
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN,
    chunk_size=200,
    chunk_overlap=0
)

chunks = splitter.split_text(text)
print(len(chunks))
print(chunks[0])
```

Other supported languages: `Language.JS`, `Language.JAVA`, `Language.CPP`, `Language.PHP`, `Language.HTML`, etc.

---

## 4. Semantic Meaning-Based Splitting

Splits based on **meaning**, not length or structure. Sentences are embedded, cosine similarity between consecutive sentences is computed, and a chunk boundary is placed wherever similarity drops sharply (a topic change).

**Class:** `SemanticChunker` (experimental)

```python
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

text = "your long text with multiple topics..."

splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="standard_deviation",  # or: percentile, interquartile, gradient
    breakpoint_threshold_amount=1
)

docs = splitter.create_documents([text])
print(len(docs))
for d in docs:
    print(d.page_content)
    print("---")
```

⚠️ Still experimental — accuracy is inconsistent; expected to improve as embedding models get better.

---

## 🔗 Chunk Overlap

`chunk_overlap` controls how many characters repeat between consecutive chunks, to preserve context that would otherwise be lost at a cut point.

- Higher overlap → better context retention, but more chunks → higher compute cost.
- Rule of thumb: overlap ≈ **10–20%** of `chunk_size` (e.g., overlap of 10–20 for a chunk size of 100).

---

## 🧭 Which One Should You Use?

| Splitter | Splits By | Best For |
|---|---|---|
| `CharacterTextSplitter` | Fixed length | Quick/simple splitting |
| `RecursiveCharacterTextSplitter` | Paragraph → sentence → word → char | **General-purpose (most used)** |
| `RecursiveCharacterTextSplitter.from_language()` | Language-specific syntax | Code, Markdown, HTML |
| `SemanticChunker` | Embedding similarity | Topic-based splitting (experimental) |

> ✅ **Default recommendation:** `RecursiveCharacterTextSplitter` for most RAG pipelines.

---

## 🤝 Contributing

Contributions are welcome! Feel free to:
1. Fork this repo
2. Create a feature branch (`git checkout -b feature/new-example`)
3. Commit your changes
4. Open a Pull Request

---

## 📚 References
- [LangChain Text Splitters Documentation](https://python.langchain.com/docs/concepts/text_splitters/)
- [LangChain Text Splitter Visualizer](https://chunkviz.up.railway.app/)

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
