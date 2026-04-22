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
from ddgs import DDGS

# --- Setup & Model Loading ---
print("Downloading and loading model (this may take a minute)...")
# Note: In a real environment, you might use Ollama or a lighter quantization for speed.
# We are using the exact model from your Kaggle notebook.
MODEL_PATH = kagglehub.model_download("google/gemma-4/transformers/gemma-4-e2b-it")

processor = AutoProcessor.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float16,
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

STRATEGIC_SYSTEM_PROMPT = """Follow this exact structure for agentic planning. Do not skip tags.
<analyze> Identify domain, scope, stakeholders, context, and constraints. </analyze>
<categorize> Separate core rules, exceptions, ambiguities, and dependencies. </categorize>
<deconstruct> Test validity, identify my own assumptions/biases, find counter-examples, and expose edge cases. </deconstruct>
<plan> Define objectives, milestones, resources, ownership, and success metrics. </plan>
<strategize> Design execution tactics, risk mitigation, contingency scenarios, and trade-off analysis. </strategize>
<implement> Execute, document processes, and establish real-time feedback mechanisms. </implement>
<iterate> Monitor output vs. metrics, evaluate gaps, adapt, and manage version control. </iterate>
<summary> A concise, user-facing summary of the final strategy and next steps. </summary>"""

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

    elif framework == "Strategic Execution (Agentic)":
        match = re.search(r"(<analyze>.*?</iterate>)\s*<summary>(.*?)</summary>", response_text, re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()
        elif "<summary>" in response_text:
            parts = response_text.split("<summary>")
            return parts[0].strip(), parts[1].replace("</summary>", "").strip()

    # Fallback for mid-stream
    return response_text, "Thinking..."

def chat_inference(chat_history, raw_messages, framework, enable_search=True):
    """
    Handles multi-turn conversational logic for the 'Philosophical Debate' tab.
    Maintains context of the debate while applying the selected framework to the newest turn.
    """
    if not chat_history:
        yield chat_history, raw_messages
        return

    user_msg = chat_history[-1][0]
    chat_history[-1] = (user_msg, "Initializing Agentic RAG...")
    yield chat_history, raw_messages

    # RAG / Search Step
    search_context = ""
    if enable_search:
        chat_history[-1] = (user_msg, "Searching DuckDuckGo and scraping URLs for factual context...")
        yield chat_history, raw_messages
        search_context = perform_search(user_msg)
        augmented_query = f"Background Fact-Check Context:\n{search_context}\n\nBased on this context and our ongoing debate, address my latest point:\n{user_msg}"
    else:
        augmented_query = user_msg

    if framework == "Hegelian Dialectic":
        system_prompt = HEGELIAN_SYSTEM_PROMPT
    elif framework == "Tetralemma (Systemic)":
        system_prompt = TETRALEMMA_SYSTEM_PROMPT
    else:
        system_prompt = STRATEGIC_SYSTEM_PROMPT

    # Initialize raw messages if empty
    if not raw_messages:
        raw_messages = [{"role": "system", "content": system_prompt}]
    elif raw_messages[0]["role"] == "system":
        # Update system prompt in case framework changed
        raw_messages[0]["content"] = system_prompt

    # Append the user's augmented query
    current_messages = list(raw_messages)
    current_messages.append({"role": "user", "content": augmented_query})

    # Process input
    text = processor.apply_chat_template(
        current_messages,
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

        # Parse on the fly
        content = raw_response
        thinking = ""
        if hasattr(processor, 'parse_response'):
            try:
                parsed_response = processor.parse_response(raw_response)
                if isinstance(parsed_response, dict):
                    content = parsed_response.get('content', raw_response)
                    thinking = parsed_response.get('thinking', '')
            except Exception:
                pass

        # Extract our XML structure
        structured_reasoning, conclusion = parse_output(content, framework)

        # For the chat UI, we will display the conclusion, and wrap the deep reasoning inside a markdown summary block
        chat_display = f"{conclusion}\n\n"

        if structured_reasoning or thinking:
            chat_display += "<details><summary><b>View Internal Philosophical Reasoning & Tool Use</b></summary>\n\n"
            if enable_search:
                chat_display += f"### Tool Use (Agentic RAG Search):\n```text\n{search_context}\n```\n\n---\n\n"
            if thinking:
                chat_display += f"### Internal Model Thoughts:\n```text\n{thinking}\n```\n\n---\n\n"
            chat_display += f"### Philosophical Framework Structure:\n```text\n{structured_reasoning}\n```\n</details>"

        chat_history[-1] = (user_msg, chat_display)
        yield chat_history, raw_messages

    # Append the raw text output to the raw_messages history to avoid UI pollution in the LLM context
    raw_messages.append({"role": "user", "content": user_msg}) # Store clean user msg, not augmented
    raw_messages.append({"role": "assistant", "content": content})

    # Log the interaction
    save_chat_to_json(user_msg, framework, search_context, structured_reasoning, conclusion)

import tempfile

def export_chat_history(chat_history, raw_messages):
    """Exports the chat history to a JSON file."""
    export_data = {
        "chat_history": chat_history,
        "raw_messages": raw_messages
    }

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump(export_data, f, indent=4)
        file_path = f.name

    return gr.update(value=file_path, visible=True)

def analyze_query(query, framework, enable_search=True):
    # Yield initial empty states for streaming
    yield "", "Initializing Agentic RAG...", gr.update(visible=False)

    # Select prompt based on dropdown
    if framework == "Hegelian Dialectic":
        system_prompt = HEGELIAN_SYSTEM_PROMPT
    elif framework == "Tetralemma (Systemic)":
        system_prompt = TETRALEMMA_SYSTEM_PROMPT
    else:
        system_prompt = STRATEGIC_SYSTEM_PROMPT

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

    with gr.Tabs():
        with gr.TabItem("Deep Analysis (Single Turn)"):
            with gr.Row():
                with gr.Column(scale=2):
                    user_query = gr.Textbox(
                        label="Enter a complex societal, ethical, or operational dilemma",
                        placeholder="e.g., Should gig workers own the algorithm?",
                        lines=3
                    )
                with gr.Column(scale=1):
                    framework_dropdown = gr.Dropdown(
                        choices=["Hegelian Dialectic", "Tetralemma (Systemic)", "Strategic Execution (Agentic)"],
                        value="Hegelian Dialectic",
                        label="Select Cognitive Framework",
                        info="Hegel resolves operational conflicts. Tetralemma deconstructs false dichotomies. Strategic plans execution."
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

        with gr.TabItem("Philosophical Debate (Chat)"):
            gr.Markdown("Engage in a multi-turn Socratic dialogue. The agent will re-evaluate the entire conversation history using the selected framework before each response.")

            with gr.Row():
                chat_framework_dropdown = gr.Dropdown(
                    choices=["Hegelian Dialectic", "Tetralemma (Systemic)", "Strategic Execution (Agentic)"],
                    value="Hegelian Dialectic",
                    label="Select Active Cognitive Framework",
                )
                chat_search_checkbox = gr.Checkbox(
                    label="Enable Agentic Search (RAG)",
                    value=True,
                )

            chatbot = gr.Chatbot(height=500, show_copy_button=True)
            raw_messages_state = gr.State([])

            with gr.Row():
                chat_input = gr.Textbox(placeholder="Debate the agent or ask a follow-up...", scale=4)
                chat_submit_btn = gr.Button("Send", variant="primary", scale=1)
                chat_save_btn = gr.Button("Save Chat", variant="secondary", scale=1)
                chat_clear_btn = gr.ClearButton([chat_input, chatbot, raw_messages_state], scale=1)

            with gr.Row():
                # Hidden download button for exported chat
                chat_download_btn = gr.DownloadButton("Download Chat (JSON)", visible=False)

            # Hide the download button when clearing the chat
            chat_clear_btn.click(
                fn=lambda: gr.update(visible=False),
                inputs=[],
                outputs=[chat_download_btn]
            )

            # Wiring for the Chat interface
            chat_save_btn.click(
                fn=export_chat_history,
                inputs=[chatbot, raw_messages_state],
                outputs=[chat_download_btn]
            )
            def user(user_message, history):
                return "", history + [[user_message, None]]

            chat_submit_btn.click(
                fn=user,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot],
                queue=False
            ).then(
                fn=chat_inference,
                inputs=[chatbot, raw_messages_state, chat_framework_dropdown, chat_search_checkbox],
                outputs=[chatbot, raw_messages_state]
            )

            # Also allow pressing Enter in the textbox
            chat_input.submit(
                fn=user,
                inputs=[chat_input, chatbot],
                outputs=[chat_input, chatbot],
                queue=False
            ).then(
                fn=chat_inference,
                inputs=[chatbot, raw_messages_state, chat_framework_dropdown, chat_search_checkbox],
                outputs=[chatbot, raw_messages_state]
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)