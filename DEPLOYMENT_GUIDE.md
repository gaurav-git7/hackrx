# HackRx API Deployment Guide for Render

This guide will help you deploy your HackRx Document Q&A API to Render.

## Prerequisites

1. A GitHub account
2. A Render account (free tier available)
3. Your API keys:
   - `GEMINI_API_KEY` (from Google AI Studio)
   - `HF_TOKEN` (from Hugging Face)

## Step 1: Prepare Your Repository

1. Make sure your code is in a GitHub repository
2. Ensure all the following files are in your repository root:
   - `main.py` (your FastAPI application)
   - `requirements.txt` (updated with FastAPI dependencies)
   - `render.yaml` (Render configuration)
   - `llm_query_retrieval_enhanced.py` (your core functionality)
   - `.gitignore` (to exclude unnecessary files)

## Step 2: Deploy to Render

### Option A: Using render.yaml (Recommended)

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" and select "Blueprint"
3. Connect your GitHub repository
4. Render will automatically detect the `render.yaml` file
5. Click "Apply" to deploy

### Option B: Manual Deployment

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" and select "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `hackrx-api`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
   - **Plan**: Free

## Step 3: Configure Environment Variables

In your Render service dashboard:

1. Go to "Environment" tab
2. Add the following environment variables:
   - `GEMINI_API_KEY`: Your Google AI Studio API key
   - `HF_TOKEN`: Your Hugging Face API token

## Step 4: Test Your Deployment

Once deployed, your API will be available at:
`https://your-service-name.onrender.com`

### Test Endpoints:

1. **Health Check**: `GET /health`
2. **Main Endpoint**: `POST /hackrx/run`

### Example Request:

```bash
curl -X POST "https://your-service-name.onrender.com/hackrx/run" \
  -H "Authorization: Bearer 02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0" \
  -H "Content-Type: application/json" \
  -d '{
    "documents": "https://example.com/sample-policy.pdf",
    "questions": ["What is the waiting period for maternity coverage?"]
  }'
```

## Important Notes

1. **Free Tier Limitations**: 
   - Services may sleep after 15 minutes of inactivity
   - First request after sleep may take 30-60 seconds
   - Limited to 750 hours per month

2. **Environment Variables**: 
   - Never commit API keys to your repository
   - Use Render's environment variable feature

3. **Dependencies**: 
   - All dependencies are listed in `requirements.txt`
   - Render will automatically install them during build

4. **Port Configuration**: 
   - Render automatically sets the `$PORT` environment variable
   - Your app uses this in the start command

## Troubleshooting

### Common Issues:

1. **Build Failures**: Check that all dependencies are in `requirements.txt`
2. **Import Errors**: Ensure all Python files are in the repository
3. **API Key Errors**: Verify environment variables are set correctly
4. **Timeout Issues**: Free tier has limitations; consider upgrading for production

### Logs:
- Check Render logs in the dashboard for debugging
- Monitor the "Logs" tab for any errors

## Security Notes

- Your authentication token is hardcoded in the application
- For production, consider using environment variables for the token
- The current token: `02b1ad646a69f58d41c75bb9ea5f78bbaf30389258623d713ff4115b554377f0`

## Next Steps

After successful deployment:
1. Test all endpoints
2. Monitor performance and logs
3. Consider setting up custom domain (paid feature)
4. Set up monitoring and alerts 