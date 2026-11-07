"""NOVA AI — a beginner-friendly hosted AI web app.

The chat and image features call hosted Hugging Face Inference Providers.
The separate train_minigpt.py script teaches how to train a tiny GPT-style
transformer from random weights. The tiny training model is not the hosted
assistant used in this app.
"""

from __future__ import annotations

import hmac
import os
from io import BytesIO
from typing import Any

import streamlit as st
from huggingface_hub import InferenceClient


# These model IDs are configurable constants. The :cheapest suffix asks the
# Hugging Face router to select its lowest-price provider for chat.
CHAT_MODEL = "openai/gpt-oss-120b:cheapest"
IMAGE_MODEL = "black-forest-labs/FLUX.1-dev"
SYSTEM_PROMPT = (
    "You are NOVA, a helpful AI assistant. Be clear, friendly, and accurate. "
    "Explain difficult ideas in beginner-friendly steps. If you are uncertain, "
    "say so. Do not claim to have performed actions you cannot perform."
)

st.set_page_config(
    page_title="NOVA AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container {max-width: 1000px; padding-top: 2rem;}
      .nova-heading {font-size: 2.4rem; font-weight: 800; letter-spacing: -1px;}
      .nova-subtitle {color: #8b949e; margin-top: -12px;}
      div[data-testid="stChatMessage"] {border-radius: 14px;}
    </style>
    """,
    unsafe_allow_html=True,
)


def read_secret(name: str) -> str:
    """Read a Streamlit secret first, then an environment variable."""
    try:
        value = st.secrets.get(name, "")
        if value:
            return str(value).strip()
    except Exception:
        # A local secrets.toml file is optional.
        pass
    return os.getenv(name, "").strip()


def password_gate() -> None:
    """Optional shared password. Set APP_PASSWORD in Streamlit secrets."""
    expected = read_secret("APP_PASSWORD")
    if not expected or st.session_state.get("nova_authenticated", False):
        return

    st.title("NOVA AI")
    st.write("This app is password-protected.")
    guess = st.text_input("App password", type="password")
    if st.button("Unlock", type="primary"):
        if hmac.compare_digest(guess, expected):
            st.session_state["nova_authenticated"] = True
            st.rerun()
        else:
            st.error("That password did not match. Try again.")
    st.stop()


password_gate()

HF_TOKEN = read_secret("HF_TOKEN")


def make_client() -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not configured in the app's secrets.")
    return InferenceClient(api_key=HF_TOKEN)


def ask_hosted_assistant(history: list[dict[str, str]]) -> str:
    """Send recent chat turns to a hosted open-weight language model."""
    client = make_client()
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # Keep a short recent history to control response size and usage.
    for item in history[-12:]:
        role = item.get("role")
        content = item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            messages.append({"role": role, "content": content})

    result: Any = client.chat_completion(
        model=CHAT_MODEL,
        messages=messages,
        max_tokens=700,
        temperature=0.7,
    )
    answer = result.choices[0].message.content
    if not isinstance(answer, str) or not answer.strip():
        raise RuntimeError("The model returned an empty response. Please try again.")
    return answer.strip()


def generate_picture(prompt: str):
    """Generate one image with a hosted text-to-image model."""
    client = make_client()
    return client.text_to_image(prompt=prompt, model=IMAGE_MODEL)


# Sidebar
with st.sidebar:
    st.markdown("## ✨ NOVA AI")
    st.caption("Your own shareable AI web app")
    if HF_TOKEN:
        st.success("AI provider connected")
    else:
        st.warning("AI provider not configured")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state["nova_messages"] = []
        st.rerun()
    st.divider()
    st.caption("**Privacy and usage**")
    st.write("Chat and image prompts are sent to the hosted AI provider. Do not enter passwords, private records, or sensitive information.")
    st.caption("The app remembers chat only in the current browser session.")
    st.divider()
    st.caption("Built with Python · Streamlit · Hugging Face")

st.markdown('<div class="nova-heading">✨ NOVA AI</div>', unsafe_allow_html=True)
st.markdown('<div class="nova-subtitle">A small AI studio you can open from a browser and share.</div>', unsafe_allow_html=True)

if not HF_TOKEN:
    st.info(
        "To activate chat and image generation, add your Hugging Face token as "
        "`HF_TOKEN` in Streamlit Community Cloud → App settings → Secrets. "
        "See the README in the project ZIP for the steps."
    )

tab_chat, tab_images, tab_lab = st.tabs(["💬 Chat", "🎨 Image studio", "🧠 My LLM lab"])

with tab_chat:
    st.subheader("Chat with NOVA")
    st.caption("ChatGPT-style interface · powered by a hosted open-weight model")

    if "nova_messages" not in st.session_state:
        st.session_state["nova_messages"] = []

    for message in st.session_state["nova_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input("Message NOVA…")
    if prompt:
        st.session_state["nova_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            if not HF_TOKEN:
                st.error("The app owner must add HF_TOKEN in the app's Secrets before chat can work.")
            else:
                with st.spinner("NOVA is thinking…"):
                    try:
                        answer = ask_hosted_assistant(st.session_state["nova_messages"])
                        st.markdown(answer)
                        st.session_state["nova_messages"].append(
                            {"role": "assistant", "content": answer}
                        )
                    except Exception as exc:
                        st.error(
                            "The chat request failed. Check that the token has "
                            "Inference Providers permission, that the selected model "
                            "is available, and that the account has provider credits. "
                            f"Details: {str(exc)[:350]}"
                        )

with tab_images:
    st.subheader("Create an image")
    st.caption("Describe the image you want. A separate image-generation model creates it.")
    image_prompt = st.text_area(
        "Image prompt",
        placeholder="A tiny robot exploring a glowing library in a cinematic 3D illustration style…",
        height=110,
    )
    generate_clicked = st.button("✨ Generate image", type="primary", disabled=not bool(HF_TOKEN))
    if generate_clicked:
        if not image_prompt.strip():
            st.warning("Write a description first.")
        else:
            with st.spinner("Creating your image…"):
                try:
                    picture = generate_picture(image_prompt.strip())
                    output = BytesIO()
                    picture.save(output, format="PNG")
                    image_bytes = output.getvalue()
                    st.image(image_bytes, caption=image_prompt.strip(), use_container_width=True)
                    st.download_button(
                        "Download image (PNG)",
                        data=image_bytes,
                        file_name="nova_image.png",
                        mime="image/png",
                    )
                except Exception as exc:
                    st.error(
                        "Image generation failed. Check model access, Inference Providers "
                        "permission, and available credits. "
                        f"Details: {str(exc)[:350]}"
                    )
    if not HF_TOKEN:
        st.info("Image generation becomes available after the app owner adds HF_TOKEN to Secrets.")

with tab_lab:
    st.subheader("Build a language model from scratch")
    st.write(
        "The project ZIP includes `train_minigpt.py`. It creates a small GPT-style "
        "transformer with randomly initialized weights, then trains it on the "
        "Tiny Shakespeare text dataset. This is your own trained model—not a "
        "downloaded pre-trained model."
    )
    st.warning(
        "Reality check: this first model learns to continue Shakespeare-like text. "
        "It will not answer general questions like NOVA or ChatGPT. A useful "
        "assistant requires much larger datasets, more training, and substantial computing."
    )
    st.markdown("**Beginner training path**")
    st.markdown(
        "1. Open Google Colab and create a new notebook.\n"
        "2. Set Runtime → Change runtime type → T4 GPU, if available.\n"
        "3. Upload `train_minigpt.py` from this project ZIP.\n"
        "4. Run `!python train_minigpt.py --steps 2500`.\n"
        "5. Download the generated `mini_gpt.pt` checkpoint."
    )
    st.caption(
        "Training is separate from the hosted NOVA chat. The app's chat feature uses "
        "a hosted model so the web app can respond more capably while you learn how "
        "to train your own small transformer."
    )
    st.code("!python train_minigpt.py --steps 2500", language="bash")

st.divider()
st.caption(
    "NOVA is a learning project, not an official OpenAI product. Hosted model availability "
    "and costs depend on the AI provider and its current terms."
)
