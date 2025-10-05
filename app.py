import gradio as gr
import openai
import os
import json
from ddgs import DDGS

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Relying on environment variables directly.")

try:
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    OPENAI_MODEL = "gpt-4o"
except KeyError:
    # This will be displayed in the UI if the key is not set
    print("FATAL ERROR: OpenAI API key not found in Space Secrets.")
    client = None
    OPENAI_MODEL = None

def call_openai_api(prompt: str, system_message: str):
    if not client:
        return {"error": "OpenAI client not initialized. Check API Key in Space Secrets."}
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        return {"error": f"API Call Failed: {e}"}

def planner_agent_openai(goal: str) -> dict:
    system_message = "You are a meticulous planner. Convert the user's goal into a specific task and a JSON list of simple, factual verification strings."
    prompt = f"""Goal: "{goal}". Provide your output in a JSON object with two keys: "task" (string) and "checklist" (list of strings)."""
    return call_openai_api(prompt, system_message)

def executor_agent(task: str) -> str:
    if not isinstance(task, str) or "FAILED" in task or "error" in task:
         return "Error: Executor received an invalid task from the planner."
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(task, max_results=1)]
            return results[0] if results else "Error: No search results found."
    except Exception as e:
        return f"Error during execution: {e}"

def verifier_agent_openai(output: str, checklist: list) -> dict:
    system_message = "You are a scrupulous verifier. Check if the 'Executor Output' satisfies ALL conditions in the 'Verification Checklist'. Respond with a JSON object."
    prompt = f"""Executor Output: "{output}"\nVerification Checklist: {checklist}. Provide your output as a JSON object with two keys: "verified" (boolean) and "reasoning" (string)."""
    return call_openai_api(prompt, system_message)

def self_verifier_agent_openai(output: str, original_task: str) -> dict:
    system_message = "You are an executor agent critically evaluating your own work. Respond with a JSON object."
    prompt = f"""Your original task was: "{original_task}"\nYour output was: "{output}"\nCritically evaluate if your output successfully and accurately completed the task. Provide your output as a JSON object with two keys: "verified" (boolean) and "reasoning" (string)."""
    return call_openai_api(prompt, system_message)

# --- Main Demo Function ---

def run_agent_system(goal, system_choice):
    """This function will be called by the Gradio interface."""
    if not client:
         return "### ❌ ERROR\nOpenAI API Key is not configured in this Space's Secrets. Please add it to run the demo."

    # 1. Planner Agent (runs for all systems)
    plan = planner_agent_openai(goal)
    if "error" in plan:
        return f"### ❌ Planner Agent Failed\n```json\n{json.dumps(plan, indent=2)}\n```"
    
    task = plan.get("task")
    checklist = plan.get("checklist", [])
    
    planner_output_md = f"### 📝 Planner Agent Output\n**Task:** `{task}`\n\n**Checklist:**\n```json\n{json.dumps(checklist, indent=2)}\n```"

    # 2. Executor Agent
    executor_output = executor_agent(task)
    executor_output_md = f"### 🛠️ Executor Agent Output\n*The agent searched the web and found the following raw text:*\n\n> {executor_output}"

    # 3. Verifier / Baseline Logic
    final_result = {}
    if system_choice == "Verifier System":
        final_result = verifier_agent_openai(executor_output, checklist)
    elif system_choice == "Self-Verifier Baseline":
        final_result = self_verifier_agent_openai(executor_output, task)
    else: # No Verifier Baseline
        final_result = {"verified": not executor_output.startswith("Error:"), "reasoning": "No Verifier present. Assumed success if no execution error."}

    if "error" in final_result:
        final_verdict_md = f"### ❌ Verification Failed\n```json\n{json.dumps(final_result, indent=2)}\n```"
    else:
        verified_status = "✅ SUCCESS" if final_result.get("verified") else "❌ FAILURE"
        final_verdict_md = f"### ⚖️ Final Verdict ({system_choice})\n**System Reported:** {verified_status}\n\n**Reasoning:**\n> {final_result.get('reasoning')}"

    return f"{planner_output_md}\n\n---\n\n{executor_output_md}\n\n---\n\n{final_verdict_md}"

# --- Gradio Interface Definition ---

BENCHMARK_GOALS = [
    "What is the boiling point of water at sea level in Celsius?", "Who is the current CEO of Microsoft?", "What was the score of the 1955 Super Bowl?", "Find the official website for the Stark Industries corporation from the Iron Man movies.", "Find the text of the 'Gettysburg Address' written by George Washington.",
]

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# The Verifier Agent: Interactive Demo")
    gr.Markdown(
        "This demo showcases the research from the paper '[The Verifier Agent](https://doi.org/10.5281/zenodo.17265873)'. "
        "Enter a goal or select a benchmark task, choose an agent architecture, and see how the Verifier Agent prevents silent failures."
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            goal_input = gr.Textbox(label="Enter Your Goal", value="Who is the current CEO of Microsoft?")
            example_dropdown = gr.Dropdown(choices=BENCHMARK_GOALS, label="Or select a benchmark task")
            system_choice = gr.Radio(
                choices=["Verifier System", "Self-Verifier Baseline", "No Verifier Baseline"],
                label="Select Agent Architecture",
                value="Verifier System"
            )
            run_button = gr.Button("Run Agent System", variant="primary")
        
        with gr.Column(scale=2):
            output_display = gr.Markdown(label="Agent System Output")

    def update_textbox_from_dropdown(dropdown_value):
        return dropdown_value

    example_dropdown.change(fn=update_textbox_from_dropdown, inputs=example_dropdown, outputs=goal_input)
    run_button.click(
        fn=run_agent_system,
        inputs=[goal_input, system_choice],
        outputs=output_display,
        queue=True
    )

demo.launch()
