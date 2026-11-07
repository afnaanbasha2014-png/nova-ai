# NOVA AI — beginner starter project

A browser-based AI studio with three parts:

1. **Chat:** a ChatGPT-style interface backed by a hosted open-weight language model through Hugging Face Inference Providers.
2. **Image studio:** text-to-image generation through a separate hosted image model.
3. **LLM lab:** `train_minigpt.py` trains a tiny GPT-style transformer from randomly initialized weights on Tiny Shakespeare.

## Important: what is—and is not—trained from scratch

The chat and image features call hosted models. They are not trained by this project. The separate `train_minigpt.py` script really does initialize and train a small transformer from scratch, but it learns to continue Shakespeare-like text. It is a learning model, not a ChatGPT-level assistant. Training a capable general-purpose assistant from scratch requires much larger datasets, extensive evaluation, and substantial GPU compute.

## Part A — create the hosted web app

You will need accounts on GitHub, Hugging Face, and Streamlit Community Cloud. The basic web app can be hosted on Streamlit Community Cloud and opened from a phone, tablet, or computer. After deployment, Streamlit gives it a `streamlit.app` URL that you can share.

### 1. Download and unzip

Download the ZIP file, then extract the `nova_ai_starter` folder.

### 2. Create a Hugging Face token

1. Sign in at <https://huggingface.co/>.
2. Open <https://huggingface.co/settings/tokens>.
3. Create a **fine-grained** token and give it the **Make calls to Inference Providers** permission.
4. Keep the token private. Do not paste it into `app.py`, chat messages, or a public GitHub file.

The app uses these hosted model IDs:

- Chat: `openai/gpt-oss-120b:cheapest`
- Images: `black-forest-labs/FLUX.1-dev`

The Hugging Face model/provider may ask you to review terms or enable billing. Check the current price and usage before enabling it. Free Hugging Face accounts currently do not include monthly Inference Provider credits, so chat and image requests may require purchased credits. Ask a parent or guardian before enabling any paid service, and set a small usage limit if the provider offers one.

### 3. Put the project on GitHub

1. Sign in at <https://github.com/>.
2. Create a new repository named `nova-ai`.
3. Upload the files and folders from `nova_ai_starter`.
4. Confirm that `.streamlit/secrets.toml` is **not** in the repository. The example file is safe because it contains only a placeholder.

For an easy first launch, a public repository is simplest. You can also use a private repository if your hosting account supports it.

### 4. Deploy it on Streamlit Community Cloud

1. Open <https://share.streamlit.io/> and sign in using GitHub.
2. Choose **Create app**.
3. Select your `nova-ai` repository, branch `main`, and main file path `app.py`.
4. Open **Advanced settings → Secrets** and paste:

   ```toml
   HF_TOKEN = "hf_your_real_token_here"
   APP_PASSWORD = "choose_a_password_for_people_you_share_with"
   ```

   `APP_PASSWORD` is optional. If set, visitors must enter that shared password. For stronger access control, use Streamlit's own private-app sharing settings and invite specific people.
5. Click **Deploy**. When it starts, Streamlit will show your app URL, such as `https://your-name-nova-ai.streamlit.app`.
6. Open that URL on your phone or another computer. Use the Share controls to copy the link.

**Public sharing warning:** anyone who can use the app may trigger requests using the app owner's Hugging Face token. Keep the link/password limited to people you trust, and monitor usage. Do not publish the token itself.

### 5. Updating the app

Edit a project file, upload the changed file to the same GitHub repository, and Streamlit Community Cloud will redeploy the app.

## Part B — train your own tiny GPT from scratch

This is a separate learning experiment. It starts with random weights and learns patterns from text; it does not download a pre-trained language model.

### Beginner route: Google Colab

1. Open <https://colab.research.google.com/> and create a new notebook.
2. In Colab, choose **Runtime → Change runtime type → T4 GPU**, if that option is available.
3. Upload `train_minigpt.py` using the Files panel on the left.
4. Run this cell:

   ```python
   !python train_minigpt.py --steps 2500
   ```

5. When training finishes, run this cell to download your model checkpoint:

   ```python
   from google.colab import files
   files.download("mini_gpt.pt")
   ```

The script also prints a short text sample from your newly trained model. If you do not have a GPU runtime, training may be much slower. If a step fails, read the complete error message and check that `train_minigpt.py` was uploaded.

### What you have built

- A character tokenizer (each character gets an integer ID).
- Token and position embeddings.
- Causal self-attention transformer blocks.
- A next-character prediction training loop.
- A checkpoint file named `mini_gpt.pt`.

The small model is intended to teach the mechanics. It is not a factual assistant and should not be used for important decisions.

## Run the web app on your own Windows PC (optional)

1. Install Python 3.11 or 3.12 from <https://www.python.org/downloads/>. During setup, enable **Add Python to PATH**.
2. Open Command Prompt inside the extracted `nova_ai_starter` folder.
3. Run:

   ```bat
   py -m venv .venv
   .venv\Scripts\activate
   py -m pip install -r requirements.txt
   ```

4. Create `.streamlit\secrets.toml` and put your private token in it:

   ```toml
   HF_TOKEN = "hf_your_real_token_here"
   APP_PASSWORD = "choose_a_password"
   ```

5. Run:

   ```bat
   streamlit run app.py
   ```

6. Your browser opens the local app. Press `Ctrl+C` in Command Prompt to stop it.

Never upload the real `secrets.toml` file to GitHub.

## Current limitations

- This starter app does not have individual user accounts, a database, or cross-device chat-history syncing. Chat history is held in the active browser session.
- Image creation is a separate image-generation model, not a feature created by the tiny GPT training script.
- Hosted provider access, model availability, speed, and price can change. A failed request may mean the model is unavailable from a provider, the token lacks permission, or the account has no credits.
- The Streamlit starter is suitable for a learning/demo project, not a production service with guaranteed uptime or strong per-user billing controls.

## Useful official documentation

- Streamlit deployment: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy>
- Streamlit secrets: <https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management>
- Streamlit app sharing: <https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app>
- Hugging Face Inference Providers: <https://huggingface.co/docs/inference-providers>
- Hugging Face Inference Providers pricing: <https://huggingface.co/docs/inference-providers/pricing>
