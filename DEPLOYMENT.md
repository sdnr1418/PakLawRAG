# Streamlit Deployment Guide

## Prerequisites

1. **GitHub Personal Access Token** with access to GitHub Models API

## Local Testing

1. Create/edit `.streamlit/secrets.toml`:
   ```toml
   GITHUB_TOKEN = "your_github_token_here"
   ```

2. Run locally:
   ```bash
   streamlit run app.py
   ```

## Deploy to Streamlit Cloud

### Step 1: Get GitHub Token

1. Go to [GitHub Settings → Developer settings → Personal access tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Fill in:
   - **Note**: "Streamlit PakLawRAG"
   - **Expiration**: 90 days (or your preference)
   - **Scopes**: Select `repo` (or `public_repo` if using public models)
4. Click "Generate token" and **copy it** (you won't see it again)

### Step 2: Add to Streamlit Cloud Secrets

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click your app → **Settings** (gear icon)
3. Go to **Secrets** tab
4. Paste this in the editor:
   ```toml
   GITHUB_TOKEN = "ghp_xxxxxxxxxxxx"
   ```
5. Click "Save"

### Step 3: Verify Models are Available

The app uses:
- **Embedding Model**: `BAAI/bge-large-en-v1.5` (HuggingFace - free)
- **LLM Model**: `openai/gpt-4.1-mini` (via GitHub Models)

Both should work with a valid GitHub token.

### Step 4: Redeploy

Streamlit will automatically restart your app. If you still see errors:

1. Check **Manage app** → **Logs** for error details
2. Verify token has correct scopes
3. Ensure token hasn't expired

## Environment Variables

Optional overrides via Secrets or `.env`:

```toml
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"
LLM_MODEL = "openai/gpt-4.1-mini"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"
```

## Troubleshooting

| Error | Solution |
|-------|----------|
| "Missing GitHub Models token" | Add `GITHUB_TOKEN` to Streamlit Secrets |
| "Model not found" | Verify model name is correct and token has access |
| "401 Unauthorized" | Token may have expired or insufficient scopes |
| "Rate limit exceeded" | GitHub Models has usage limits; upgrade GitHub account if needed |

## Security Notes

- ✅ `secrets.toml` is in `.gitignore` — never committed
- ✅ Streamlit Cloud stores secrets encrypted
- ✅ Token only used for API calls within Streamlit servers
- 🔄 Rotate tokens monthly for best practices
