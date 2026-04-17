import gradio as gr
import torch
import kagglehub
import re
import json
import os
import requests
from bs4 import BeautifulSoup
from threading import Thread
from transformers import AutoProcessor, AutoModelForCausalLM, TextIteratorStreamer
from duckduckgo_search import DDGS

# --- Setup & Model Loading ---
print("Downloading and loading model (this may take a minute)...")
# Note: In a real environment, you might use Ollama or a lighter quantization for speed.
# We are using the exact model from your Kaggle notebook.
MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e2b-it")

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto"
)

# --- Prompts ---
HEGELIAN_SYSTEM_PROMPT = """Follow this exact structure in your reasoning before responding. Do not skip tags.
<reason> State initial position + core assumption. </reason>
<critique> Attack the weakest logical/empirical link. Do not nitpick. </critique>
<respond> Concede at least one valid critique point, then defend/adjust. </respond>
<synthesis> Fuse reason & critique into a new framing that resolves the tension without averaging them. </synthesis>
<final> Clear, actionable conclusion. Note remaining uncertainty if any. </final>"""

TETRALEMMA_SYSTEM_PROMPT = """Follow this exact structure. Do not skip tags.
<reason> State the baseline rule/assumption. </reason>
<exception> Present a valid counterfactual or edge case that breaks the rule. </exception>
<tension> Describe the systemic friction when rule & exception coexist in practice. </tension>
<categorization> Clearly separate domains: where the rule holds, where the exception applies, and why. </categorization>
<deconstruction> Refute the false dichotomy. Reveal the higher-order framing, interdependence, or contextual mechanism that transcends both. </deconstruction>
<conclusion> Actionable takeaway + explicit boundary conditions. Note what would shift the conclusion if new context emerges. </conclusion>"""

# --- Processing Logic ---

def save_chat_to_json(query, framework, search_context, reasoning, conclusion):
    """Saves the interaction to a JSON audit log."""
    log_file = "chat_history.json"
    entry = {
        "query": query,
        "framework": framework,
        "search_context": search_context,
        "reasoning_scaffolding": reasoning,
        "final_conclusion": conclusion
    }

    data = []
    if os.path.exists(log_file):
        with open(log_file, "r") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                pass

    data.append(entry)

    with open(log_file, "w") as f:
        json.dump(data, f, indent=4)

    return log_file

def perform_search(query):
    """
    Acts as an advanced RAG retrieval step using deep web scraping.
    It fetches top search results and then scrapes the actual webpage content.
    """
    try:
        results = list(DDGS().text(query, max_results=3))
        context = ""

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }

        for idx, r in enumerate(results):
            url = r.get('href')
            snippet = r.get('body', '')

            scraped_text = ""
            if url:
                try:
                    # Attempt to fetch the actual webpage content
                    response = requests.get(url, headers=headers, timeout=5)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')

                        # Extract text from paragraphs
                        paragraphs = soup.find_all('p')
                        raw_text = " ".join([p.get_text(strip=True) for p in paragraphs])

                        # Clean and truncate the scraped text to avoid blowing up the context window
                        clean_text = re.sub(r'\s+', ' ', raw_text)
                        scraped_text = clean_text[:800] # Limit to ~800 chars per source
                except Exception as scrape_e:
                    # If scraping fails, just fallback to the DuckDuckGo snippet
                    scraped_text = f"Scrape failed. Snippet: {snippet}"
            else:
                scraped_text = snippet

            # If we scraped successfully but got nothing, fallback to snippet
            if not scraped_text.strip():
                scraped_text = snippet

            context += f"Source {idx+1} ({r.get('title', 'Unknown')}): {scraped_text}...\n\n"

        return context.strip() if context else "No factual context found."
    except Exception as e:
        return f"Search failed: {str(e)}"

def parse_output(response_text, framework):
    """
    Splits the raw LLM output into the 'hidden reasoning' (XML tags)
    and the 'final conclusion' robustly for streaming.
    """
    # For streaming, if we don't have the closing tags yet, just show everything in the reasoning block
    # and say "Thinking..." in the conclusion block.

    if framework == "Hegelian Dialectic":
        match = re.search(r"(<reason>.*?</synthesis>)\s*<final>(.*?)</final>", response_text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        elif "<final>" in response_text:
            parts = response_text.split("<final>")
            return parts[0].strip(), parts[1].replace("</final>", "").strip()

    elif framework == "Tetralemma (Systemic)":
        match = re.search(r"(<reason>.*?</deconstruction>)\s*<conclusion>(.*?)</conclusion>", response_text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        elif "<conclusion>" in response_text:
            parts = response_text.split("<conclusion>")
            return parts[0].strip(), parts[1].replace("</conclusion>", "").strip()

    # Fallback for mid-stream
    return response_text, "Thinking..."

def analyze_query(query, framework, enable_search=True):
    # Yield initial empty states for streaming
    yield "", "Initializing Agentic RAG...", gr.update(visible=False)

    # Select prompt based on dropdown
    system_prompt = HEGELIAN_SYSTEM_PROMPT if framework == "Hegelian Dialectic" else TETRALEMMA_SYSTEM_PROMPT

    # RAG / Search Step
    search_context = ""
    if enable_search:
        yield "", "Searching DuckDuckGo and scraping URLs for factual context...", gr.update(visible=False)
        search_context = perform_search(query)
        augmented_query = f"Background Fact-Check Context:\n{search_context}\n\nBased on this context and your knowledge, address the following query:\n{query}"
    else:
        augmented_query = query

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": augmented_query},
    ]

    # Process input
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True
    )

    inputs = processor(text=text, return_tensors="pt").to(model.device)

    # Setup Streamer
    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=False)
    generation_kwargs = dict(inputs, streamer=streamer, max_new_tokens=4096)

    # Start generation in a separate thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    # Stream the output
    raw_response = ""
    for new_text in streamer:
        raw_response += new_text

        # Parse on the fly for UI
        content = raw_response
        thinking = ""
        if hasattr(processor, 'parse_response'):
            try:
                parsed_response = processor.parse_response(raw_response)
                if isinstance(parsed_response, dict):
                    content = parsed_response.get('content', raw_response)
                    thinking = parsed_response.get('thinking', '')
            except Exception:
                pass # Continue appending raw text if mid-generation parsing fails

        # Extract our XML structure
        structured_reasoning, conclusion = parse_output(content, framework)

        # Format the hidden reasoning block nicely
        hidden_reasoning_display = ""
        if enable_search:
            hidden_reasoning_display += f"### Tool Use (Agentic RAG Search):\n```text\n{search_context}\n```\n\n---\n\n"

        if thinking:
            hidden_reasoning_display += f"### Internal Model Thoughts:\n```text\n{thinking}\n```\n\n---\n\n"

        hidden_reasoning_display += f"### Philosophical Framework Structure:\n```text\n{structured_reasoning}\n```"

        # During streaming, don't update the download button yet
        yield hidden_reasoning_display, conclusion, gr.update()

    # Final Save to JSON when streaming completes
    saved_file = save_chat_to_json(query, framework, search_context, structured_reasoning, conclusion)
    yield hidden_reasoning_display, conclusion, gr.update(value=saved_file, visible=True)

# --- Gradio UI ---
custom_css = """
#conclusion_box textarea {
    font-size: 18px !important;
    font-weight: 500;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    gr.Markdown("# 🏛️ Gemma 4: Dialectical-Engine")
    gr.Markdown("This agent breaks 'Hallucinated Consensus' by passing user queries through rigorous, ancient philosophical frameworks before presenting a final, actionable synthesis. Built for the **Safety & Trust** track.")

    with gr.Row():
        with gr.Column(scale=2):
            user_query = gr.Textbox(
                label="Enter a complex societal, ethical, or operational dilemma",
                placeholder="e.g., Should gig workers own the algorithm?",
                lines=3
            )
        with gr.Column(scale=1):
            framework_dropdown = gr.Dropdown(
                choices=["Hegelian Dialectic", "Tetralemma (Systemic)"],
                value="Hegelian Dialectic",
                label="Select Cognitive Framework",
                info="Hegel resolves operational conflicts. Tetralemma deconstructs false dichotomies."
            )
            enable_search_checkbox = gr.Checkbox(
                label="Enable Agentic Search (RAG)",
                value=True,
                info="Prevents factual hallucinations before reasoning."
            )
            analyze_btn = gr.Button("Run Agentic Analysis", variant="primary")

    gr.Markdown("## Resolution")

    # The final answer is front and center
    final_output = gr.Textbox(
        label="Actionable Conclusion & Boundary Conditions",
        lines=4,
        elem_id="conclusion_box",
        show_copy_button=True
    )

    # The complex reasoning is hidden in an accordion
    with gr.Accordion("View System 2 Philosophical Reasoning Flow (Audit Log)", open=False):
        gr.Markdown("Transparency is critical for Trust & Safety. Here is the exact logical scaffolding the model used to arrive at the conclusion.")
        reasoning_output = gr.Markdown(show_copy_button=True)

    with gr.Row():
        # Hide the button initially. It will appear when generation finishes.
        download_btn = gr.DownloadButton("Download Audit Logs (JSON)", visible=False)

    analyze_btn.click(
        fn=analyze_query,
        inputs=[user_query, framework_dropdown, enable_search_checkbox],
        outputs=[reasoning_output, final_output, download_btn]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)