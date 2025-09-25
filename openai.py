import os
import openai
import json
import csv
from ddgs import DDGS # Reverted to the original, stable library
import time
import sys # To exit the script gracefully

# --- OpenAI API Setup ---
# It's good practice to load from a .env file.
# Make sure you have a .env file with OPENAI_API_KEY='sk-...'
# and you have run 'pip install python-dotenv'
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Warning: python-dotenv not found. Relying on environment variables directly.")

API_KEY = os.environ.get("OPENAI_API_KEY")

if not API_KEY:
    print("FATAL ERROR: OpenAI API key not found.")
    print("Please create a .env file or set the OPENAI_API_KEY environment variable.")
    sys.exit(1) # Exit the script if the key is not found

client = openai.OpenAI(api_key=API_KEY)

# --- CORRECTED MODEL NAME ---
# Use a valid, available model. "gpt-4o" is the latest and best choice.
OPENAI_MODEL = "gpt-4o" 

# --- Evaluation Benchmark (Same as before for fair comparison) ---
BENCHMARK_TASKS = [
    {"id": 1, "goal": "What is the boiling point of water at sea level in Celsius?"}, {"id": 2, "goal": "Who is the current CEO of Microsoft?"}, {"id": 3, "goal": "What year did the first moon landing occur?"}, {"id": 4, "goal": "Find the main ingredient in a traditional Japanese Miso soup."}, {"id": 5, "goal": "What is the capital city of Australia?"}, {"id": 6, "goal": "What is the population of the underwater city of Atlantis?"}, {"id": 7, "goal": "Find the official website for the Stark Industries corporation from the Iron Man movies."}, {"id": 8, "goal": "What is the chemical formula for Kryptonite?"}, {"id": 9, "goal": "Who is the king of the United States?"}, {"id": 10, "goal": "How many dragons are there in the wild in Germany?"}, {"id": 11, "goal": "What is the weather like?"}, {"id": 12, "goal": "Find a good recipe."}, {"id": 13, "goal": "How tall is the president?"}, {"id": 14, "goal": "Is it a holiday today?"}, {"id": 15, "goal": "What is the latest news?"}, {"id": 16, "goal": "What was the score of the 1955 Super Bowl?"}, {"id": 17, "goal": "Did Thomas Edison invent the light bulb?"}, {"id": 18, "goal": "Is water a good conductor of electricity?"}, {"id": 19, "goal": "What is the currency used in Switzerland?"}, {"id": 20, "goal": "Find the text of the 'Gettysburg Address' written by George Washington."},
]

# --- Agent Definitions (Using OpenAI API with better error handling) ---

def call_openai_api(prompt: str, system_message: str):
    """Generic function to call the OpenAI Chat Completions API."""
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
    except openai.NotFoundError as e:
        print(f"  ERROR: Model '{OPENAI_MODEL}' not found. Please check the model name. Details: {e}")
        return None
    except openai.AuthenticationError as e:
        print(f"  ERROR: OpenAI API key is invalid or has expired. Please check your key. Details: {e}")
        return None
    except Exception as e:
        print(f"  ERROR calling OpenAI API: {e}")
        return None

def planner_agent_openai(goal: str) -> dict:
    system_message = "You are a meticulous planner. Convert the user's goal into a specific task and a JSON list of simple, factual verification strings."
    prompt = f"""Goal: "{goal}". Provide your output in a JSON object with two keys: "task" (string) and "checklist" (list of strings)."""
    result = call_openai_api(prompt, system_message)
    # If API call fails, provide a clear error task.
    return result or {"task": "PLANNER_AGENT_FAILED", "checklist": []}

def executor_agent(task: str) -> str:
    """This agent does not use the LLM, so it remains the same."""
    print(f"  EXECUTOR 🛠️: Received task: '{task}'")
    if task == "PLANNER_AGENT_FAILED":
        return "Error: Executor received a failed task from the planner."
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(task, max_results=1)]
            output = results[0] if results else "Error: No search results found."
            print(f"  EXECUTOR 🛠️: Found output: '{output[:100]}...'") # Log output
            return output
    except Exception as e:
        return f"Error during execution: {e}"

def verifier_agent_openai(output: str, checklist: list) -> dict:
    if not checklist: # If the planner failed, the checklist will be empty.
        return {"verified": False, "reasoning": "Verification skipped because the planner agent failed to create a checklist."}
    system_message = "You are a scrupulous verifier. Check if the 'Executor Output' satisfies ALL conditions in the 'Verification Checklist'. Respond with a JSON object."
    prompt = f"""Executor Output: "{output}"\nVerification Checklist: {checklist}. 
    Provide your output as a JSON object with two keys: "verified" (boolean) and "reasoning" (string)."""
    result = call_openai_api(prompt, system_message)
    return result or {"verified": False, "reasoning": "Error in verification API call"}

def self_verifier_agent_openai(output: str, original_task: str) -> dict:
    if original_task == "PLANNER_AGENT_FAILED":
         return {"verified": False, "reasoning": "Self-verification skipped because the planner agent failed."}
    system_message = "You are an executor agent critically evaluating your own work. Respond with a JSON object."
    prompt = f"""Your original task was: "{original_task}"\nYour output was: "{output}"
    Critically evaluate if your output successfully and accurately completed the task.
    Provide your output as a JSON object with two keys: "verified" (boolean) and "reasoning" (string)."""
    result = call_openai_api(prompt, system_message)
    return result or {"verified": False, "reasoning": "Error in self-verification API call"}

# --- Workflow Definitions (no changes here) ---

def run_verifier_system_openai(goal: str):
    plan = planner_agent_openai(goal)
    task, checklist = plan.get("task"), plan.get("checklist", [])
    output = executor_agent(task)
    result = verifier_agent_openai(output, checklist)
    return plan, output, result

def run_no_verifier_system_openai(goal: str):
    plan = planner_agent_openai(goal)
    task = plan.get("task")
    output = executor_agent(task)
    result = {"verified": not output.startswith("Error:"), "reasoning": "No verifier present. Assumed success."}
    return plan, output, result

def run_self_verifier_system_openai(goal: str):
    plan = planner_agent_openai(goal)
    task = plan.get("task")
    output = executor_agent(task)
    result = self_verifier_agent_openai(output, task)
    return plan, output, result

# --- Main Evaluation Loop ---

def main():
    systems = {
        "Verifier_System_GPT4o": run_verifier_system_openai,
        "No_Verifier_Baseline_GPT4o": run_no_verifier_system_openai,
        "Self_Verifier_Baseline_GPT4o": run_self_verifier_system_openai
    }
    
    csv_file_path = "evaluation_results_openai.csv"
    csv_headers = ["task_id", "goal", "system_type", "planner_task", "planner_checklist", "executor_output", "system_reported_success", "verifier_reasoning"]

    with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_headers)
        writer.writeheader()

        for task_item in BENCHMARK_TASKS:
            for system_name, system_func in systems.items():
                print(f"\n--- Running Task ID {task_item['id']} on {system_name} ---")
                print(f"GOAL: {task_item['goal']}")

                start_time = time.time()
                plan, output, result = system_func(task_item['goal'])
                end_time = time.time()

                print(f"  RESULT: {result}")
                print(f"  (Time taken: {end_time - start_time:.2f}s)")

                writer.writerow({
                    "task_id": task_item['id'],
                    "goal": task_item['goal'],
                    "system_type": system_name,
                    "planner_task": plan.get('task'),
                    "planner_checklist": json.dumps(plan.get('checklist')),
                    "executor_output": output,
                    "system_reported_success": result.get('verified'),
                    "verifier_reasoning": result.get('reasoning')
                })

    print(f"\n✅ Evaluation complete. Results saved to '{csv_file_path}'")

if __name__ == "__main__":
    main()
