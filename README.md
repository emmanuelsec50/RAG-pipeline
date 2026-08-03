# Vixxon AI

A Retrieval-Augmented Generation (RAG) chatbot built with Django, featuring real-time streaming responses, semantic retrieval over a custom knowledge base, and a fully custom branded chat interface. I built the project to help university students go through processes such as HELB applications, admission processes and university student portal access.

**Live demo:** [ai.vixxon.online](https://ai.vixxon.online)

---

## Features

- **Retrieval-Augmented Generation** — answers are grounded in a custom knowledge base via dense embedding retrieval, not just raw model knowledge.
- **Real-time streaming** — responses stream token-by-token over Server-Sent Events (SSE), including a separate live stream of the model's reasoning trace.
- **Markdown-aware rendering** — streamed responses are parsed and rendered as proper markdown (lists, bold text, code blocks, links) in real time, sanitized client-side to prevent injection.
- **Light & dark themes** — full theme toggle with a palette derived from Vixxon's existing brand identity.
- **Rich link previews** — links returned in an answer are automatically expanded into preview cards (title, description, image) fetched server-side.
- **Shareable, legit-looking links** — Open Graph and Twitter Card metadata so the app itself previews properly when shared on WhatsApp, X, Slack, or LinkedIn.
- **Mobile-first UI** — proper viewport handling, iOS zoom-prevention on inputs, safe-area padding for notched devices.
- **Rate limiting** — request throttling via `django-ratelimit` to protect the underlying LLM/embedding API costs.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend framework | Django 6, Gunicorn |
| Retrieval | PyTorch (CPU-only) — dense vector similarity over pre-computed embeddings |
| Embeddings & generation | NVIDIA-hosted API (`nvidia/nemotron-3-embed-1b` for embeddings, `Deepseek` for generation) |
| Streaming | Server-Sent Events (SSE) over Django `StreamingHttpResponse` |
| Frontend | Vanilla HTML/CSS/JS — no framework, no build step |
| Markdown rendering | marked.js + DOMPurify (client-side, sanitized) |
| Link previews | BeautifulSoup (server-side OpenGraph scraping) |
| Rate limiting | django-ratelimit |
| Hosting | Render (custom domain via Cloudflare DNS) |

---

## How Retrieval Works

1. A knowledge base is pre-processed offline into sentence-level chunks and embedded via the NVIDIA embeddings API.
2. Embeddings are stored as a single `.pt` tensor file, loaded into memory once at application startup (not per-request).
3. At query time, the user's question is embedded the same way, and top-k most relevant chunks are retrieved via dot-product similarity computed directly in PyTorch.
4. Retrieved chunks are injected into a prompt template alongside the user's question.
5. The final prompt is sent to the generation model, and the response is streamed back to the client as it's generated.

---

## Getting Started

### Prerequisites

- Python 3.11+
- An NVIDIA API key with access to embedding and chat completion endpoints

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/emmanuelsec50/RAG-pipeline.git
   cd RAG-pipeline
   ```

2. **Create a virtual environment and install dependencies**

   PyTorch must be installed separately as a CPU-only build to avoid pulling in unnecessary CUDA dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```

3. **Configure environment variables**

   Create a `.env` file in the project root:
   ```
   LLM_API_KEY=your_nvidia_api_key
   EMBED_API_KEY=your_nvidia_api_key
   DJANGO_SECRET_KEY=your_secret_key
   ```

4. **Add your knowledge base files**

   Place your pre-computed `embeddings.pt` and `pages_and_chunks.pkl` files in the `ragapp/` directory.

5. **Run migrations and start the server**
   ```bash
   python manage.py migrate
   python manage.py runserver
   ```

6. Visit `http://127.0.0.1:8000` in your browser.

---

## Deployment

This project is deployed on [Render](https://render.com) with the following considerations:

- **Build command** installs the CPU-only PyTorch wheel explicitly before the rest of the requirements:
  ```
  pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
  ```
- **`CSRF_TRUSTED_ORIGINS`** and **`ALLOWED_HOSTS`** must include the deployed domain.
- **Custom domain** (`ai.vixxon.online`) is routed via a Cloudflare CNAME record, set to DNS-only during initial certificate verification.
- **Keep-alive**: a scheduled GitHub Actions workflow pings a lightweight health endpoint every 13 minutes to prevent Render's free-tier instance from spinning down.

---

## Roadmap / Known Trade-offs

- No persistent chat history yet — each session starts fresh.
- Semantic caching (returning cached answers for near-duplicate queries) is under consideration but not yet implemented.
- Rate limiting currently uses Django's default in-memory cache — fine for a single worker, would need Redis for multi-worker deployments.

---

## License

Distributed under the MIT License.
