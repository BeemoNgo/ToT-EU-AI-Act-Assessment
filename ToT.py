import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from auto_gptq import AutoGPTQForCausalLM

def query_model(model, tokenizer, system_desc: str, prompt: str, max_new_tokens: int = 200) -> str:
    """
    Query the LLaMA-2 7B model with a given prompt and system description.
    Returns the raw decoded response text.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    combined_input = f"System Description:\n{system_desc}\n\n{prompt}"
    inputs = tokenizer(combined_input, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Querying model with prompt:\n{prompt[:100]}...")
    return response

def parse_answer(response: str, answer_type: str = "yesno") -> tuple:
    """
    Parse the model's response to extract the Answer and Reasoning.
    `answer_type` can be:
      - "yesno" for prompts expecting 'Yes'/'No'
      - "scale" for prompts expecting a numeric 1-5 answer
      - "compliance" for prompts expecting 'Compliant'/'Partially Compliant'/'Non-Compliant'
    Returns a tuple (answer, reasoning).
    """
    # Find the second occurrence of "Answer:" and "Reasoning:" to get model's answer (the prompt may contain a template Answer/Reasoning).
    try:
        parts = response.split("Answer:")
        if len(parts) < 2:
            # If the model didn't follow the format strictly, try alternative parsing
            # For example, it might output directly without repeating "Answer:"
            text = response.strip()
        else:
            text = parts[-1]  # get content after the last "Answer:"
        # Now split reasoning
        if "Reasoning:" in text:
            answer_text, reasoning_text = text.split("Reasoning:", 1)
        else:
            # If no "Reasoning:" found, assume the whole text is answer_text
            answer_text, reasoning_text = text, ""
        answer_text = answer_text.strip()
        reasoning_text = reasoning_text.strip()
        # Normalize the answer based on expected type
        if answer_type == "yesno":
            # Only expecting "Yes" or "No"
            answer_text = answer_text.split()[0]  # get first word
            answer_text = answer_text.capitalize()
            if answer_text not in ["Yes", "No"]:
                # Handle cases like "Probably not" -> "No", "Probably yes" -> "Yes"
                answer_text = "Yes" if "yes" in answer_text.lower() else "No"
        elif answer_type == "scale":
            # Expect a number (1-5)
            # Extract first occurrence of digit in answer_text
            import re
            match = re.search(r"[1-5]", answer_text)
            answer_text = match.group(0) if match else ""
        elif answer_type == "compliance":
            # Expect one of the compliance labels
            # Find which keyword is present
            answer_lower = answer_text.lower()
            if "partially" in answer_lower:
                answer_text = "Partially Compliant"
            elif "non" in answer_lower:
                answer_text = "Non-Compliant"
            elif "compliant" in answer_lower:
                # If just "Compliant" (and not 'non' which we handled), assume fully Compliant
                answer_text = "Compliant"
            else:
                # Default fallback if not recognized
                answer_text = answer_text.strip()
        else:
            answer_text = answer_text.strip()
        return answer_text, reasoning_text.strip()
    except Exception as e:
        # In case parsing fails, return the whole response as reasoning, with answer as empty
        return "", response.strip()

def classify_risk(system_desc: str) -> dict:
    """
    Perform staged risk classification on `system_desc` using traditional prompting.
    Returns a dictionary with the determined risk category, confidence score, and reasoning.
    """
    # Define risk categories and associated prompts (sub-criteria questions).
    risk_keywords = {
        "Unacceptable Risk": [
            "subliminal", "exploit vulnerabilities", "biometric categorization",
            "social scoring", "real-time biometric identification", "emotion recognition",
            "predictive policing", "facial image scraping"
        ],
        "High Risk": [
            "safety component", "medical device", "vocational training", "employment decisions",
            "credit scoring", "immigration processing", "law enforcement", "biometric identification",
            "critical infrastructure", "health", "safety", "fundamental rights"
        ],
        "Limited Risk": [
            "chatbot", "deepfake", "AI-generated", "deceive", "transparency"
        ],
        "Minimal Risk": [
            "routine application", "no significant harm", "benign use"
        ]
    }

    system_lower = system_desc.lower()
    result_reasoning = []
    assigned_category = "Minimal Risk"

    for category in ["Unacceptable Risk", "High Risk", "Limited Risk", "Minimal Risk"]:
        keywords = risk_keywords[category]
        hits = [kw for kw in keywords if kw in system_lower]
        if hits:
            assigned_category = category
            result_reasoning.append(f"Matched keywords for {category}: {', '.join(hits)}")
            break  # assign the first matched (highest severity)

    return {
        "risk_category": assigned_category,
        "confidence_score": 5.0 if assigned_category != "Minimal Risk" else 2.0,
        "reasoning": "; ".join(result_reasoning) or "No high-risk keywords matched. Assigned minimal risk."
    }

def assess_compliance_tot(system_desc: str, model, tokenizer) -> dict:
    """
    Assess compliance of `system_desc` with various EU AI Act criteria using ToT reasoning.
    Returns a dictionary mapping each criterion to a dict of {status, justification}.
    """
    # Define compliance criteria and sub-criteria prompts
    compliance_criteria = {
        "Risk Management Plan": [
            "Does the provider have a documented risk management system for this AI, identifying and mitigating potential risks?",
            "Is the risk management process continuous and updated throughout the AI system's lifecycle?",
            "Has the system been tested to identify the most appropriate risk mitigation measures (before deployment)?"
        ],
        "Data Governance": [
            "Are training and test datasets for the AI appropriate, of high quality, and free from known biases or errors?",
            "Are there measures in place to identify and mitigate biases or inaccuracies in the data used by the AI system?",
            "Does the system comply with data governance practices (data privacy, security, and quality controls) for the data it uses?"
        ],
        "Technical Documentation": [
            "Has comprehensive technical documentation been prepared (describing the system's purpose, design, and performance)?",
            "Does the documentation include instructions for use and information on the AI system’s limitations or risk areas?",
            "Is the documentation sufficient for authorities and users to understand how to operate the system safely and effectively?"
        ],
        "Record-Keeping": [
            "Does the system keep logs or records of its operations (inputs, outputs, decisions) to enable traceability?",
            "Are the logs maintained in a way that ensures their integrity and availability for audits or investigations?",
            "Do the record-keeping practices comply with the AI Act requirements (allowing detection of issues and accountability)?"
        ],
        "Transparency": [
            "Are users or affected persons clearly informed that they are interacting with an AI system (when not obvious)?",
            "If the AI generates content or recommendations, are its outputs accompanied by explanations or labels as needed (to be understandable)?",
            "Does the system provide sufficient information to users about its capabilities, limitations, and the fact that it's AI-driven?"
        ],
        "Human Oversight": [
            "Is the AI system designed to allow effective human oversight (can a human intervene or override decisions if necessary)?",
            "Are there measures to ensure human review or supervision for critical outcomes produced by the AI system?",
            "Are personnel using the AI system trained or instructed on how to monitor the system and handle its potential failures?"
        ],
        "Accuracy, Robustness, Cybersecurity": [
            "Has the AI system been tested for accuracy, and are its error rates or performance metrics documented?",
            "Is the system robust to potential inputs or conditions (i.e., it can handle variations without critical failures)?",
            "Are there cybersecurity safeguards to prevent unauthorized access or manipulation of the AI system (ensuring integrity and security)?"
        ],
        "Fairness/Non-Discrimination": [
            "Has the AI system been evaluated for potential bias or discriminatory outcomes against protected groups?",
            "Are there mechanisms to mitigate or avoid discrimination (e.g., adjustments in training data or algorithms to address bias)?",
            "Is the system designed or audited to ensure it provides equitable results for individuals regardless of race, gender, etc.?"
        ]
    }
    compliance_results = {}

    # Base instruction for yes/no answers on sub-criteria
    sub_prompt_prefix = ("You are a compliance auditor. Analyze the system for the following point and answer Yes or No with a brief reason.\n"
                         "Template:\nAnswer: [Yes/No]\nReasoning: [short explanation]\nQuestion: ")
    for criterion, questions in compliance_criteria.items():
        sub_answers = []    # store (Yes/No, reasoning) for each sub-question
        # Ask each sub-criterion question
        for q in questions:
            prompt = sub_prompt_prefix + q
            response = query_model(model, tokenizer, system_desc, prompt)
            ans, reason = parse_answer(response, answer_type="yesno")
            if ans not in ["Yes", "No"]:
                # In case the model gave a non-binary or uncertain answer, classify it as "No" for safety
                ans = "No"
            sub_answers.append((ans, reason))
        # Determine overall compliance status from sub-answers
        yes_count = sum(1 for ans, _ in sub_answers if ans == "Yes")
        if yes_count == len(sub_answers):
            status = "Compliant"
        elif yes_count == 0:
            status = "Non-Compliant"
        else:
            status = "Partially Compliant"
    
        findings = "; ".join([f"{q}: {ans}" for (ans, _), q in zip(sub_answers, questions)])
        summary_prompt = (f"The system was evaluated for '{criterion}'. The findings for sub-points were: {findings}. "
                           "Based on these findings, classify the system's compliance with this criterion as Compliant, Partially Compliant, or Non-Compliant and explain why.\n"
                           "Answer: [Compliant/Partially Compliant/Non-Compliant]\nReasoning: ")
        summary_response = query_model(model, tokenizer, system_desc, summary_prompt)
        comp_label, comp_reason = parse_answer(summary_response, answer_type="compliance")
        # If the model's label disagrees with our simple aggregation, we can use the model's (to incorporate nuanced judgment)
        if comp_label in ["Compliant", "Partially Compliant", "Non-Compliant"]:
            status = comp_label  # use model's classification if valid
        justification = comp_reason if comp_reason else " ".join([r for _, r in sub_answers])
        compliance_results[criterion] = {
            "status": status,
            "justification": justification.strip()
        }
    return compliance_results

try:
    import networkx as nx
    from networkx.drawing.nx_pydot import to_pydot
except ImportError:
    nx = None

def visualize_compliance_tree(compliance_dict: dict, output_file: str = "compliance_tree.png"):
    """
    Generate a visualization of the compliance tree.
    Each criterion is a branch from the root (the AI system), and each sub-criterion could be depicted as leaf nodes.
    The output is saved as an image (PNG).
    """
    if nx is None:
        print("NetworkX or pydot not available. Skipping visualization.")
        return
    G = nx.DiGraph()
    root = "AI System Compliance"
    G.add_node(root)
    for crit, result in compliance_dict.items():
        status = result.get("status", "")
        crit_node = f"{crit} ({status})"
        G.add_node(crit_node)
        G.add_edge(root, crit_node)
        justification = result.get("justification", "")
        sub_points = [s.strip() for s in justification.split(';') if s.strip()]
        # Limit to a few sub-points to keep the graph readable
        for sub in sub_points[:3]:
            sub_node = f"{crit} -> {sub}"
            G.add_node(sub_node)
            G.add_edge(crit_node, sub_node)
    # Use pydot layout for a neat tree
    try:
        pydot_graph = to_pydot(G)
        pydot_graph.set_rankdir("TB")  # top-to-bottom layout
        pydot_graph.write_png(output_file)
        print(f"Compliance tree diagram saved to {output_file}")
    except Exception as e:
        print(f"Visualization failed: {e}")

import time
def main():
    df = pd.read_csv("datasets/test.csv")[:2]
    # We will use the 'Full Description' and data columns to form the system description input.
    results = []            # to collect risk classification results for CSV
    compliance_outputs = {} # to collect compliance results for JSON/analysis

    for index, row in df.iterrows():
        app_name = str(row.get("App Name", f"app_{index}"))
        print(f"\n [{index+1}/{len(df)}] Starting: {app_name}")
        # Compose system description from available fields
        description = str(row.get("Full Description", ""))
        # Optionally include data collection/sharing info to provide context on data governance/privacy
        data_collected = str(row.get("Collected Data", ""))
        data_shared = str(row.get("Shared Data", ""))
        security = str(row.get("Security Practices", ""))
        if data_collected or data_shared or security:
            description += "\n\nAdditional Info:\n"
            if data_collected:
                description += f"Data Collected: {data_collected}\n"
            if data_shared:
                description += f"Data Shared: {data_shared}\n"
            if security:
                description += f"Security Practices: {security}\n"
        
        risk_result = classify_risk(description)
        risk_result["App Name"] = app_name
        results.append(risk_result)
        
        compliance_result = assess_compliance_tot(description, model, tokenizer)
        compliance_outputs[app_name] = compliance_result
        
        print(f"Processed: {app_name} -> Risk: {risk_result['risk_category']}, Compliance evaluated.")
    # Save risk classification results to CSV and JSON
    results_df = pd.DataFrame(results)
    results_df.to_csv("risk_classification_results.csv", index=False)
    results_df.to_json("risk_classification_results.json", orient="records", indent=2)
    
    with open("compliance_results.json", "w") as fjson:
        json.dump(compliance_outputs, fjson, indent=2)
    comp_rows = []
    for app, criteria in compliance_outputs.items():
        for crit, outcome in criteria.items():
            comp_rows.append({
                "App Name": app,
                "Criterion": crit,
                "Compliance Status": outcome.get("status", ""),
                "Justification": outcome.get("justification", "")
            })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv("compliance_results.csv", index=False)
    print("All outputs saved to risk_classification_results.csv/json and compliance_results.csv/json.")

model_name = "tiiuae/falcon-rw-1b"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",  
    torch_dtype=torch.float32  
)

if __name__ == "__main__":
    main()