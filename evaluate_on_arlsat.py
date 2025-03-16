import json
import time
import os
import argparse
import requests
from tqdm import tqdm
import numpy as np
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

MODEL_ID = "deepseek/deepseek-r1"

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load the AR-LSAT dataset from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_prompt(example: Dict[str, Any]) -> str:
    """Format the prompt for the model."""
    context = example["context"]
    question = example["question"]
    options = example["options"]
    
    formatted_options = "\n".join(options)
    
    prompt = f"""You are solving a logical reasoning problem from the LSAT. 
Please analyze the following problem carefully and select the correct answer.

Context:
{context}

Question:
{question}

Options:
{formatted_options}

First, think through this problem step by step with your detailed reasoning.
Then, provide your final answer as a single letter (A, B, C, D, or E).
"""
    return prompt

def call_openrouter_api(prompt: str, max_retries: int = 3, retry_delay: int = 5) -> Optional[Dict[str, Any]]:
    """Call the OpenRouter API with retries and return the full response."""
    
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Logic-LLM",
        "X-Title": "AR-LSAT Evaluation"
    }
    
    data = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,  # Use deterministic output for evaluation
        "max_tokens": 8000,
        "stream": False,      # Ensure streaming is off for faster complete responses
        "top_p": 0.1,         # Lower top_p for faster, more deterministic responses
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0
    }
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries}: Calling OpenRouter API...")
            response = requests.post(OPENROUTER_API_URL, headers=headers, json=data)
            
            # Print response status for debugging
            print(f"Response status: {response.status_code}")
            
            # Handle different error codes specifically
            if response.status_code == 401:
                print("Authentication error (401): Your API key is invalid or expired.")
                print("Please check your OpenRouter API key and make sure it's correctly formatted.")
                return None
            elif response.status_code == 403:
                print("Authorization error (403): You don't have permission to use this model.")
                return None
            elif response.status_code == 429:
                print("Rate limit error (429): Too many requests. Retrying after delay...")
                time.sleep(retry_delay * 2)  # Longer delay for rate limits
                continue
            
            # Raise for other HTTP errors
            response.raise_for_status()
            
            # Parse the response
            response_json = response.json()
            
            # Return the full response object
            return response_json
                
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Skipping this example.")
                return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Skipping this example.")
                return None
        except Exception as e:
            print(f"Error calling OpenRouter API: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Max retries reached. Skipping this example.")
                return None

def extract_answer(content: str) -> Optional[str]:
    """Extract the answer from the model's response using regex."""
    # Try to find the final answer format first (most reliable)
    final_answer_patterns = [
        r"(?:^|\n)\s*\*\*(?:Answer|ANSWER):\*\*\s+([A-E])\s*$",  # "**Answer:** X" at end of text
        r"(?:^|\n)\s*(?:Answer|ANSWER):\s*([A-E])\s*$",  # "Answer: X" at end
        r"(?:^|\n)\s*(?:Answer|ANSWER):\s*\*\*([A-E])\*\*\s*$",  # "Answer: **X**" at end
        r"(?:^|\n)\s*\*\*(?:Answer|ANSWER):\*\*\s*([A-E])\s*$",  # "**Answer:** X" at end
        r"(?:^|\n)\s*\*\*(?:Answer|ANSWER):\*\*\s*\*\*([A-E])\*\*\s*$",  # "**Answer:** **X**" at end
    ]
    
    for pattern in final_answer_patterns:
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return match.group(1)
        
    return None

def evaluate_model(dataset: List[Dict[str, Any]], output_file: str, limit: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate the model on the dataset and save results."""
    results = []
    correct_count = 0
    total_count = 0
    
    # Add parallel processing with a reasonable batch size
    results_lock = threading.Lock()
    
    def process_example(example):
        prompt = format_prompt(example)
        response_json = call_openrouter_api(prompt)
        
        if not response_json or "choices" not in response_json or len(response_json["choices"]) == 0:
            print("No valid response received. Skipping this example.")
            return None
        
        # Extract content and reasoning from the response
        message = response_json["choices"][0]["message"]
        content = message.get("content", "")
        reasoning = message.get("reasoning", "")

        answer_match = extract_answer(content)
        predicted_answer = answer_match.upper() if answer_match else None
        
        correct_answer = example["answer"]
        is_correct = predicted_answer == correct_answer if predicted_answer else False
        
        result = {
            "id": example["id"],
            "prompt": prompt,
            "content": content,
            "reasoning": reasoning,
            "predicted_answer": predicted_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        }
        
        with results_lock:
            nonlocal correct_count, total_count
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Save intermediate results
            with open(output_file, 'a' if os.path.exists(output_file) else 'w', encoding='utf-8') as f:
                f.write(json.dumps(result, indent=2) + '\n')
            
            current_accuracy = correct_count / total_count if total_count > 0 else 0
            print(f"Current accuracy: {current_accuracy:.4f} ({correct_count}/{total_count})")
        
        return result
    
    # Limit the dataset if specified
    if limit:
        dataset = dataset[:limit]
    
    # Use ThreadPoolExecutor for parallel API calls
    # Adjust max_workers based on your API rate limits
    max_workers = 30
    
    print(f"Evaluating {len(dataset)} examples with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(process_example, dataset),
            total=len(dataset),
            desc="Processing examples",
            unit="example"
        ))
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Evaluation complete. Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    
    return {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate reasoning model on AR-LSAT dataset")
    parser.add_argument("--model_name", type=str, default="deepseek_r1", help="Name of the model")
    parser.add_argument("--dataset", type=str, default="data/AR-LSAT/test.json", help="Path to the dataset file")
    parser.add_argument("--output", type=str, default=None, help="Path to save results")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of examples to evaluate")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = f"results/results_{args.model_name}_arlsat_8000_tokens.json"
    
    print(f"Loading dataset from {args.dataset}")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} examples")
    
    if args.limit:
        print(f"Limiting evaluation to {args.limit} examples")
    
    print(f"Evaluating {args.model_name} on AR-LSAT dataset")
    results = evaluate_model(dataset, args.output, args.limit)
    print(f"Accuracy: {results}")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
