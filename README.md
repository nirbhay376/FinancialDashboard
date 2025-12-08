# FinanceHub Pro — Advanced Financial Analysis Dashboard

A beautiful, responsive Dash application for real-time stock analysis, technical indicators, valuation metrics, and financial data visualization.

## Features

- **Real-time Stock Data**: powered by yfinance
- **Advanced Analytics**: technical indicators, valuation models, financial statement analysis
- **Responsive Design**: works on desktop, tablet, and mobile devices
- **Dark/Light Theme Toggle**: user preference persisted
- **Multi-stock Comparison**: analyze and compare up to 3 stocks side-by-side
- **Performance Optimized**: LRU caching for yfinance calls, responsive UI with modern CSS

## Quick Start (Local)

### Prerequisites
- Python 3.9+
- pip or poetry

### Install & Run Locally

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app in development mode
python3 finance.py
# App runs on http://localhost:8050

# 3. (Optional) Run with Gunicorn (production-like)
gunicorn finance:server --workers 1 --worker-class gevent --bind 0.0.0.0:8050
```

## Deployment

### Option 1: Cloud Run (Free Tier) ⭐ Recommended

Deploy your app to Google Cloud Run with free-tier settings. The app will be publicly accessible from any device.

**Prerequisites:**
- Google Cloud account (free tier eligible)
- `gcloud` CLI installed ([install here](https://cloud.google.com/sdk/docs/install))
- Active GCP project

**Deploy:**
```bash
bash deploy-cloud-run.sh
```

Or manually:
```bash
# Authenticate
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# Enable APIs
gcloud services enable run.googleapis.com cloudbuild.googleapis.com

# Build and push
gcloud builds submit --tag gcr.io/YOUR_GCP_PROJECT_ID/financehub

# Deploy (free-tier)
gcloud run deploy financehub \
  --image gcr.io/YOUR_GCP_PROJECT_ID/financehub \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --concurrency 10 \
  --min-instances 0
```

**Access:**
After deployment, you'll get a public HTTPS URL. Share it with anyone — they can access it from any browser, any device.

**Cost:**
- Free for low traffic (within free tier quota).
- First request after idle may take 10–30 seconds (cold start with `min-instances 0`).
- Set budget alerts in Google Cloud Console to stay within free limits.

### Option 2: Docker (Local or VPS)

**Build locally:**
```bash
docker build -t financehub:local .
docker run -p 8050:8050 -e PORT=8050 financehub:local
```

**Deploy to a VPS (DigitalOcean, Linode, etc.):**
```bash
# On your VPS, after installing Docker:
docker pull YOUR_IMAGE_NAME:latest
docker run -d --restart unless-stopped -p 80:8050 -e PORT=8050 YOUR_IMAGE_NAME:latest
# Configure nginx and certbot for HTTPS (not covered here)
```

### Option 3: Fly.io

Create and deploy a new Fly app:
```bash
flyctl launch
flyctl deploy
```

Choose a machine size with ≥512Mi memory to avoid OOMs.

## Architecture

- **Backend:** Dash (Python web framework) + Gunicorn (WSGI server)
- **Frontend:** Plotly (interactive charts), Dash components (UI)
- **Data:** yfinance (stock data), pandas (analysis), numpy (calculations)
- **Caching:** LRU cache on yfinance calls to reduce API load and latency
- **Styling:** responsive CSS with Inter font, light/dark theme support

## Configuration

### Environment Variables

- `PORT`: port to listen on (default: 8050)
- `DEBUG`: enable debug mode (set to "true" to enable; default: false)

### Gunicorn Tuning

In `procfile` or CLI:
```
gunicorn finance:server --workers 1 --worker-class gevent --bind 0.0.0.0:$PORT
```

**Workers:**
- Use `--workers 1` for low-memory machines (256–512 MiB).
- Increase to 2–4 for multi-core instances if memory permits.

**Worker Class:**
- `gevent`: good for I/O-bound tasks (network calls); requires `gevent` package.
- `sync` (default): simpler, good for CPU-bound tasks.

## File Structure

```
.
├── finance.py              # Main Dash app (1000+ lines)
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container build config
├── procfile               # Gunicorn startup command
├── fly.toml               # Fly.io config (if deploying there)
├── deploy-cloud-run.sh    # Cloud Run deploy automation
├── assets/
│   └── style.css          # Responsive global styles
└── README.md              # This file
```

## Troubleshooting

### Cold Start Delays (Cloud Run)
If the first request is slow (10–30 seconds), this is expected with `min-instances 0`. Options:
- Accept the delay (free tier trade-off).
- Upgrade to `--min-instances 1` (costs ~$5–10/month).
- Set up a background job to periodically ping the URL to keep an instance warm.

### Out of Memory (OOM)
If the app crashes with OOM errors:
1. **Cloud Run:** increase `--memory` to `1Gi` and redeploy.
2. **Docker/VPS:** ensure the container has ≥1 GiB RAM allocated.
3. **Reduce workers:** set `--workers 1` to lower memory usage.

### yfinance API Rate Limits
If you see rate-limit errors:
- The app caches yfinance calls (LRU cache with maxsize=128).
- Rate-limiting is rare for typical usage; if it occurs, yfinance may have temporary issues.

### Port Already in Use
If port 8050 is occupied:
```bash
# Find and kill the process
lsof -i :8050
kill -9 <PID>

# Or run on a different port
gunicorn finance:server --bind 0.0.0.0:8051
```

## Performance Tips

1. **Caching:** The app uses `@lru_cache` for yfinance calls. Restart the app to clear cache if needed.
2. **Lazy Loading:** Tabs are rendered on-demand; data fetches happen when you click a tab.
3. **Mobile:** Use responsive CSS; tested on mobile browsers.
4. **Concurrency:** Cloud Run's `--concurrency 10` allows multiple requests per instance; for CPU-heavy work, reduce to 1–2.

## Development

### Adding New Features
1. Edit `finance.py` to add callbacks or visualizations.
2. Test locally: `python3 finance.py`
3. Commit and push to GitHub.
4. Redeploy: `bash deploy-cloud-run.sh`

### Dependencies
- Core: `dash`, `plotly`, `pandas`, `numpy`, `yfinance`
- Production: `gunicorn`, `gevent`
- Install all: `pip install -r requirements.txt`

## License & Credits

This is a custom financial analysis dashboard. Use and modify freely.

---

**Questions?** Check the deployment script output or Google Cloud Console logs for errors.

Happy analyzing! 📊
