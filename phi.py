import ollama
import json
import csv
from ddgs import DDGS
import time

# --- Evaluation Benchmark ---
BENCHMARK_TASKS = [
    {"id": 1, "goal": "What is the boiling point of water at sea level in Celsius?"}, {"id": 2, "goal": "Who is the current CEO of Microsoft?"}, {"id": 3, "goal": "What year did the first moon landing occur?"}, {"id": 4, "goal": "Find the main ingredient in a traditional Japanese Miso soup."}, {"id": 5, "goal": "What is the capital city of Australia?"}, {"id": 6, "goal": "What is the population of the underwater city of Atlantis?"}, {"id": 7, "goal": "Find the official website for the Stark Industries corporation from the Iron Man movies."}, {"id": 8, "goal": "What is the chemical formula for Kryptonite?"}, {"id": 9, "goal": "Who is the king of the United States?"}, {"id": 10, "goal": "How many dragons are there in the wild in Germany?"}, {"id": 11, "goal": "What is the weather like?"}, {"id": 12, "goal": "Find a good recipe."}, {"id": 13, "goal": "How tall is the president?"}, {"id": 14, "goal": "Is it a holiday today?"}, {"id": 15, "goal": "What is the latest news?"}, {"id": 16, "goal": "What was the score of the 1955 Super Bowl?"}, {"id": 17, "goal": "Did Thomas Edison invent the light bulb?"}, {"id": 18, "goal": "Is water a good conductor of electricity?"}, {"id": 19, "goal": "What is the currency used in Switzerland?"}, {"id": 20, "goal": "Find the text of the 'Gettysburg Address' written by George Washington."},
]

# --- Agent Definitions ---

def planner_agent(goal: str) -> dict:
    prompt = f"""You are a meticulous planner. Convert the high-level goal into a single, specific, and verifiable task. Create a Python list of simple, factual verification checks.
    Goal: "{goal}"
    Provide your output in a JSON format with two keys: "task" and "checklist"."""
    try:
        response = ollama.chat(model='phi', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"task": "Error in planning", "checklist": [f"Error: {e}"]}

def executor_agent(task: str) -> str:
    print(f"  EXECUTOR 🛠️: Performing task: '{task}'")
    try:
        with DDGS() as ddgs:
            results = [r['body'] for r in ddgs.text(task, max_results=1)]
            return results[0] if results else "Error: No search results found."
    except Exception as e:
        return f"Error during execution: {e}"

def verifier_agent(output: str, checklist: list) -> dict:
    prompt = f"""You are a scrupulous verifier. Check if the 'Executor Output' satisfies ALL conditions in the 'Verification Checklist'.
    Respond with a JSON object with two keys: "verified" (a boolean: true if all checks pass, otherwise false) and "reasoning" (a brief explanation).
    Executor Output: "{output}"
    Verification Checklist: {checklist}"""
    try:
        response = ollama.chat(model='phi', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"verified": False, "reasoning": f"Error in verification: {e}"}

def self_verifier_agent(output: str, original_task: str) -> dict:
    prompt = f"""You are an executor agent who must now verify your own work.
    Your original task was: "{original_task}"
    Your output was: "{output}"
    Critically evaluate if your output successfully and accurately completed the original task.
    Respond with a JSON object with two keys: "verified" (a boolean) and "reasoning" (a brief explanation of why you believe your work was or was not successful)."""
    try:
        response = ollama.chat(model='phi', messages=[{'role': 'user', 'content': prompt}], format='json')
        return json.loads(response['message']['content'])
    except Exception as e:
        return {"verified": False, "reasoning": f"Error in self-verification: {e}"}

# --- Workflow Definitions ---

def run_verifier_system(goal: str):
    plan = planner_agent(goal)
    task, checklist = plan.get("task"), plan.get("checklist", [])
    output = executor_agent(task)
    result = verifier_agent(output, checklist)
    return plan, output, result

def run_no_verifier_system(goal: str):
    plan = planner_agent(goal)
    task = plan.get("task")
    output = executor_agent(task)
    # Baseline 1: Assumes success if no execution error occurs.
    result = {"verified": not output.startswith("Error:"), "reasoning": "No verifier present. Assumed success."}
    return plan, output, result

def run_self_verifier_system(goal: str):
    plan = planner_agent(goal)
    task = plan.get("task")
    output = executor_agent(task)
    result = self_verifier_agent(output, task)
    return plan, output, result

# --- Main Evaluation Loop ---

def main():
    """Runs the full evaluation and saves results to a CSV file."""
    systems = {
        "Verifier_System": run_verifier_system,
        "No_Verifier_Baseline": run_no_verifier_system,
        "Self_Verifier_Baseline": run_self_verifier_system
    }
    
    csv_file_path = "evaluation_results.csv"
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

                # Write result to CSV
                writer.writerow({
                    "task_id": task_item['id'],
                    "goal": task_item['goal'],
                    "system_type": system_name,
                    "planner_task": plan.get('task'),
                    "planner_checklist": json.dumps(plan.get('checklist')), # Store checklist as a JSON string
                    "executor_output": output,
                    "system_reported_success": result.get('verified'),
                    "verifier_reasoning": result.get('reasoning')
                })

    print(f"\n✅ Evaluation complete. Results saved to '{csv_file_path}'")

if __name__ == "__main__":
    main()