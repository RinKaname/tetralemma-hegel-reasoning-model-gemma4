import gradio as gr
import torch
import kagglehub
import re
from transformers import AutoProcessor, AutoModelForCausalLM

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

# --- Processing Logic ---
def parse_output(response_text, framework):
    """
    Splits the raw LLM output into the 'hidden reasoning' (XML tags)
    and the 'final conclusion'.
    """
    if framework == "Hegelian Dialectic":
        # Extract everything up to the closing synthesis tag for the reasoning block
        match = re.search(r"(<reason>.*?</synthesis>)\s*<final>(.*?)</final>", response_text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            final_conclusion = match.group(2).strip()
            return reasoning, final_conclusion
    elif framework == "Tetralemma (Systemic)":
        match = re.search(r"(<reason>.*?</deconstruction>)\s*<conclusion>(.*?)</conclusion>", response_text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
            final_conclusion = match.group(2).strip()
            return reasoning, final_conclusion

    # Fallback if tags weren't strictly followed
    return response_text, "The model did not return a clean conclusion tag. See reasoning block for full text."

def analyze_query(query, framework):
    # Select prompt based on dropdown
    system_prompt = HEGELIAN_SYSTEM_PROMPT if framework == "Hegelian Dialectic" else TETRALEMMA_SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # Process input
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True # We enable thinking to get the highest quality reasoning
    )

    inputs = processor(text=text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    # Generate output
    outputs = model.generate(**inputs, max_new_tokens=4096)
    raw_response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)

    # Parse the thinking block (if any) and the actual structured content
    parsed_response = processor.parse_response(raw_response)

    content = parsed_response.get('content', '')
    thinking = parsed_response.get('thinking', '')

    # Extract our XML structure
    structured_reasoning, conclusion = parse_output(content, framework)

    # Format the hidden reasoning block nicely
    hidden_reasoning_display = ""
    if thinking:
        hidden_reasoning_display += f"### Internal Model Thoughts:\n```text\n{thinking}\n```\n\n---\n\n"

    hidden_reasoning_display += f"### Philosophical Framework Structure:\n```text\n{structured_reasoning}\n```"

    return hidden_reasoning_display, conclusion

# --- Gradio UI ---
custom_css = """
#conclusion_box textarea {
    font-size: 18px !important;
    font-weight: 500;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=custom_css) as app:
    gr.Markdown("# 🏛️ Gemma 4: Philosophical Cognitive Architecture")
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
            analyze_btn = gr.Button("Run Analysis", variant="primary")

    gr.Markdown("## Resolution")

    # The final answer is front and center
    final_output = gr.Textbox(
        label="Actionable Conclusion & Boundary Conditions",
        lines=4,
        elem_id="conclusion_box"
    )

    # The complex reasoning is hidden in an accordion
    with gr.Accordion("View System 2 Philosophical Reasoning Flow (Audit Log)", open=False):
        gr.Markdown("Transparency is critical for Trust & Safety. Here is the exact logical scaffolding the model used to arrive at the conclusion.")
        reasoning_output = gr.Markdown()

    analyze_btn.click(
        fn=analyze_query,
        inputs=[user_query, framework_dropdown],
        outputs=[reasoning_output, final_output]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=True)