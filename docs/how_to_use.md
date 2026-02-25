# How to Use PrERT-CNM

This document provides a comprehensive guide on utilizing the deliverables produced over the four-month lifecycle of the PrERT-CNM project. It outlines expected results, execution commands, and local deployment strategies.

## Deployment Strategy and Environment Setup

The PrERT-CNM engine is designed as a modular Python package and includes a FastAPI backend for the showcase application. Below are the steps to deploy the system both locally for development and on a production server.

### 1. Initialize Environment

It is highly recommended to isolate dependencies using a virtual environment.

```bash
python3 -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

### 2. Install Dependencies

Install the required machine learning, probabilistic, data processing, and web server libraries.

```bash
pip install -r requirements.txt
```

### 3. Configuration (.env)

The ingestion layer relies on a Vector Database (ChromaDB) for Context Memory. Create a `.env` file in the root directory to configure the environment variables:

```env
CHROMADB_COLLECTION_NAME=prert_memory
CHROMADB_API_KEY=your_api_key_here
CHROMADB_TENANT=default_tenant
CHROMADB_DATABASE=default_database
```

### 4. Running Locally (Development)

To run the evaluation engine and showcase API locally with hot-reloading:

```bash
uvicorn api.showcase_server:app --reload --host localhost --port 8000
```

The API will be available at `http://127.0.0.1:8000`. You can verify the server is running by visiting the health check endpoint:

```bash
curl http://127.0.0.1:8000/health
```

When using the Interactive Showcase HTML frontend, set the **Backend API Server** field to `http://127.0.0.1:8000`.

### 5. Server Deployment (Production)

For production environments, it is recommended to run the server using `gunicorn` with `uvicorn` workers to handle concurrent requests robustly.

Ensure HuggingFace caching is properly scoped to avoid system-wide permission errors:

```bash
export HF_HOME="$(pwd)/.hf_cache"
```

Start the production server:

```bash
gunicorn api.showcase_server:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 6. Exposing the Local Server with Cloudflare Tunnel

When running the PrERT-CNM server on your local PC and needing to expose it to the internet (e.g. for the hosted GitHub Pages frontend to reach the backend), you can use **Cloudflare Tunnel** (`cloudflared`). This avoids the **524 timeout error** that occurs when intermediate proxies or tunnelling protocols are not configured to allow long-running AI inference requests.

#### Why This Is Needed

The PrERT-CNM pipeline runs DeBERTa encoding, attention rollout, and CNM agent reasoning sequentially. For larger documents this process can exceed the default timeout thresholds of many reverse proxies and hosting platforms, resulting in a **HTTP 524 (A Timeout Occurred)** error. Running the server locally and tunnelling it via `cloudflared` bypasses these third-party timeout limits.

#### Install Cloudflared

Download and install `cloudflared` for your platform:

- **macOS (Homebrew):**

  ```bash
  brew install cloudflared
  ```

- **Windows (winget):**

  ```bash
  winget install --id Cloudflare.cloudflared
  ```

- **Linux (Debian/Ubuntu):**

  ```bash
  curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb -o cloudflared.deb
  sudo dpkg -i cloudflared.deb
  ```

#### Start the Tunnel

First, make sure the uvicorn server is running locally:

```bash
uvicorn api.showcase_server:app --reload --host localhost --port 8000
```

Then, in a **separate terminal**, start the Cloudflare quick tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

`cloudflared` will output a public URL (e.g. `https://<random-subdomain>.trycloudflare.com`). Copy this URL and paste it into the **Backend API Server** field in the Interactive Showcase frontend, or use it as the base URL for API requests.

#### Verifying the Tunnel

Confirm the tunnel is working by calling the health endpoint through the public URL:

```bash
curl https://<random-subdomain>.trycloudflare.com/health
```

You should receive `{"status":"ok","pipeline":"initialized"}`.

#### Troubleshooting the 524 Timeout

If you still encounter a 524 error:

1. **Ensure the local server is running** — confirm `uvicorn` is active and responding on `http://localhost:8000/health`.
2. **Check cloudflared is connected** — the terminal running `cloudflared` should show an active connection with no errors.
3. **Reduce document size** — very large documents increase inference time. Try a shorter policy text first to confirm the pipeline works end-to-end.
4. **Check firewall rules** — ensure your local firewall is not blocking connections on port 8000.

---

## Month-by-Month Usage Guide

### Month 1: Standards Mapping

**Component:** `config/` layer.

**How to Use:**
The primary deliverable is the deterministic mapping of privacy regulations to software configurations. You can interact with the current state of mappings by editing `config/privacy_indicators.json`.

**Expected Result:**
When executed through the loader, the system rigidly enforces GDPR/NIST structural integrity using Pydantic models. Any malformed mappings will crash the pipeline immediately, protecting the Bayesian Engine from processing invalid data.

**Commands:**
To verify the configuration logic holds true against the mappings:

```bash
pytest tests/test_pipeline.py -k "test_config_validation"
```

### Month 2: Metrics Definition & Synthetic Data

**Component:** `data/` layer.

**How to Use:**
The goal is to provide the models with data to learn from. The downloaded OPP-115 alternative public mirror is cached as the definitive baseline truth.

**Expected Result:**
Running the dataset download script caches the Parquet binary files into `data/raw/`. Data loading scripts will interface strictly with this local cache to avoid external HTTP dependencies during offline training.

**Commands:**
To pull and persist the data locally:

```bash
python data/download.py
```

### Month 3: AI Prototype Development

**Components:** `models/` and `engine/` layers.

**How to Use:**
This phase activates the core AI capabilities. You provide raw, unstructured privacy text to the `PrivacyFeatureExtractor` (transformer model). The logits generated by the model are passed to the `BayesianRiskEngine` as evidence nodes.

**Expected Result:**
The transformer layer extracts latent features representing clauses (e.g., data minimization principles). The Bayesian Network takes those features, updates its Variable Elimination distributions, and returns a concrete, auditable probability scale representing overall standard non-compliance risk.

**Commands:**
To run logic checks proving the topological mapping dynamically aligns the JSON config points to mathematically queryable DAG structures:

```bash
pytest tests/test_pipeline.py -k "test_dynamic_topology_generation"
```

### Month 4: Testing & Final Validation

**Component:** `tests/` and Benchmarking.

**How to Use:**
Execution of the complete CI/CD test suite simulating adversarial compliance anomalies and ensuring probabilistic limits.

**Expected Result:**
All unit tests should pass indicating that the risk indicators are mapped perfectly, uncertainty bounds are correctly mathematically restricted, and the HuggingFace trainers correctly compute tokenized arrays.

**Commands:**
Run the complete testing suite ensuring no module caching conflicts:

```bash
pytest tests/test_pipeline.py -v -p no:cacheprovider
```
