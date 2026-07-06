Server deployment (Python + MusicGen)

This document describes how to deploy the `musicserver.py` (Flask) application which wraps the MusicGen model and exposes a /generate endpoint.

Prerequisites
- A Linux server (Ubuntu 20.04+ recommended) with at least one GPU for reasonable generation performance.
- Python 3.10+ installed. If using GPU, install a PyTorch build that supports your CUDA version.
- A domain name and (recommended) HTTPS certificate (nginx + certbot).

Quick local run (development)
1. Create a virtualenv and activate it:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install requirements:

```powershell
pip install -r requirements.txt
```

3. Set required environment variables:

- GEMINI_API_KEY: your LLM/gemini API key (optional — server falls back to a simple prompt builder if not set).
- PORT (optional): port to listen on (default 5000)

```powershell
$env:GEMINI_API_KEY = "your_key"
$env:PORT = "5000"
```

4. Start the server (dev):

```powershell
python musicserver.py
```

Production run (recommended)
- Use a process manager (systemd) + reverse proxy (nginx) with TLS. Example with gunicorn:

1. Install gunicorn in the same venv if not installed:

```powershell
pip install gunicorn
```

2. Run with gunicorn (Linux):

```bash
# from project root
# Run with 1 worker (ML inference is CPU/GPU bound, tune workers carefully)
gunicorn --bind 0.0.0.0:8000 --workers 1 --threads 2 "musicserver:app"
```

3. Put nginx in front to terminate TLS and forward / to gunicorn. Use certbot to obtain certificates.

Notes and recommendations
- For production use HTTPS. Android blocks cleartext connections by default on newer Android versions.
- Configure the Android app's `SERVER_BASE_URL` to point at the public HTTPS URL (e.g. https://yourdomain.com).
- Monitor disk usage — generated files are stored to `static/music/` and the server includes a background cleaner to remove files older than 1 hour.
- If running on GPU, install a matching `torch` binary built for your CUDA version.

Docker (optional)
- Create a Dockerfile that installs the dependencies and runs the server. Expose 5000/8000 and run behind an external reverse proxy.

Troubleshooting
- Model loading can be slow and memory-hungry. Make sure the server has enough RAM and GPU memory.
- If the LLM prompt API fails or the GEMINI_API_KEY is missing, the server will revert to a simple prompt builder.
- If you want streaming audio instead of file URLs, consider switching to an async server (FastAPI + Uvicorn) and streaming the bytes — this requires client support.
