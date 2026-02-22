# Deploy RAG Chatbot on Render

## What’s in this repo for Render

- **`Dockerfile`** – build and run the Streamlit app in a container.
- **`render.yaml`** – optional Blueprint; you can also create the service by hand.

## Steps to deploy

### 1. Push code to GitHub

Make sure this repo (including `Dockerfile`, `render.yaml`, `streamlit_app.py`, `requirements.txt`) is pushed to a GitHub repository.

### 2. Create a Render account

Go to [render.com](https://render.com) and sign up (e.g. with GitHub).

### 3. Create a Web Service

1. In the [Render Dashboard](https://dashboard.render.com), click **New +** → **Web Service**.
2. Connect your **GitHub** account if needed, then select the **repository** that contains this app.
3. Configure the service:
   - **Name:** e.g. `rag-chatbot`
   - **Region:** pick one close to you
   - **Root Directory:** leave blank if the app is in the repo root; otherwise set it to the folder that contains `streamlit_app.py` and `Dockerfile`
   - **Runtime:** **Docker**
   - **Instance type:** Free (or a paid plan if you need more memory for the embedding model)

4. **Environment variables** (required):
   - **Key:** `OPENROUTER_API_KEY`  
     **Value:** your OpenRouter API key (e.g. `sk-or-v1-...`)  
     → Click **Add** and mark it as **Secret**.
   - (Optional) **Key:** `OPENROUTER_MODEL`  
     **Value:** e.g. `openai/gpt-4o-mini`

5. Click **Create Web Service**.

Render will build the Docker image and start the app. The first deploy may take a few minutes (embedding model download). When it’s ready, Render will show a URL like `https://rag-chatbot-xxxx.onrender.com`.

### 4. Using the Blueprint (optional)

Instead of configuring the Web Service by hand, you can use the Blueprint:

1. **New +** → **Blueprint**.
2. Connect the same GitHub repo.
3. Render will read `render.yaml` and create the Web Service.
4. You still must set **OPENROUTER_API_KEY** in the service’s **Environment** tab (as a secret); it is not stored in `render.yaml`.

---

## After deploy

- **Cold starts:** On the free tier the service may sleep after inactivity; the first request can take 30–60 seconds.
- **API key:** Never commit `OPENROUTER_API_KEY` to the repo; always set it in Render’s Environment (secret).
- **Memory:** If the app crashes or runs out of memory, try a paid instance or a smaller embedding model.
