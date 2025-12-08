#!/bin/bash
set -e

# FinanceHub Pro — Cloud Run Free-Tier Deployment Script
# This script builds and deploys your Dash app to Google Cloud Run
# Usage: bash deploy-cloud-run.sh

echo "=========================================="
echo "FinanceHub Pro — Cloud Run Deployment"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "❌ gcloud CLI not found. Please install it:"
    echo "   macOS: brew install google-cloud-sdk"
    echo "   https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project)
if [ -z "$PROJECT_ID" ]; then
    echo "❌ No GCP project configured."
    echo "Please run: gcloud auth login && gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

echo "✅ Using GCP Project: $PROJECT_ID"
echo ""

# Authenticate (optional — will prompt if needed)
echo "🔐 Authenticating with Google Cloud..."
gcloud auth login || true

# Enable required APIs
echo "🔧 Enabling Cloud Run and Cloud Build APIs..."
gcloud services enable run.googleapis.com cloudbuild.googleapis.com --quiet

# Build the Docker image
echo "🏗️ Building Docker image..."
gcloud builds submit --tag gcr.io/$PROJECT_ID/financehub --quiet

# Deploy to Cloud Run
echo "🚀 Deploying to Cloud Run (free-tier settings)..."
gcloud run deploy financehub \
  --image gcr.io/$PROJECT_ID/financehub \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 512Mi \
  --cpu 1 \
  --concurrency 10 \
  --min-instances 0 \
  --quiet

# Get the service URL
echo ""
echo "=========================================="
echo "✅ Deployment Complete!"
echo "=========================================="
SERVICE_URL=$(gcloud run services describe financehub --platform managed --region us-central1 --format 'value(status.url)' 2>/dev/null || echo "https://financehub-REGION.run.app")
echo "📱 Your app is live at:"
echo "   $SERVICE_URL"
echo ""
echo "ℹ️  Notes:"
echo "   • First request may take 10–30s (cold start)"
echo "   • Subsequent requests will be fast"
echo "   • Cost: free for low traffic (within free tier)"
echo "   • Monitor: gcloud run services describe financehub --region us-central1"
echo ""
