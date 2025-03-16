import json
import os
import time
from typing import Optional, Dict, Any, List
from collections import Counter
import argparse
import re
from dotenv import load_dotenv
from openai import OpenAI
import random

# Load environment variables from .env file
load_dotenv()

def extract_answer_with_llm(client, content: str, reasoning: str, id: str) -> Optional[str]:
    """Extract the answer from the model's response using GPT-4o-mini."""
    if not content and not reasoning:
        return None
    
    # Prepare the input text
    input_text = f"CONTENT:\n{content}\n\nREASONING:\n{reasoning}"
    
    # Create system and user messages
    system_message = """You are an answer extraction assistant. Your job is to analyze the text from an AI model's solution to a multiple choice problem and extract ONLY the letter answer (A, B, C, D, or E).

Rules:
1. Return ONLY a single capital letter (A, B, C, D, or E) without periods, quotes, or explanations
2. If there's an explicit 'Answer: X' or 'The answer is X', use that
3. If the model clearly indicates a final answer in bold (**X**) or states 'X is correct', use that
4. If the reasoning and content sections indicate different answers, go with the final answer from the content section
5. If you cannot confidently determine an answer, respond with "UNCLEAR"
"""
    
    user_message = f"Identify the single letter answer (A, B, C, D, or E) from this AI response to a multiple choice question (ID: {id}):\n\n{input_text}"
    
    # Call the OpenAI API
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Use gpt-4o for higher accuracy if needed
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=10  # We only need a short response
        )
        
        # Get the assistant's response
        answer_text = response.choices[0].message.content.strip()
        
        # Extract just the letter using regex if needed
        letter_match = re.search(r'^\s*([A-E])\s*$', answer_text, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        elif answer_text == "UNCLEAR":
            return None
        else:
            # Try to find any letter in the response
            letter_match = re.search(r'[A-E]', answer_text, re.IGNORECASE)
            if letter_match:
                return letter_match.group(0).upper()
            return None
            
    except Exception as e:
        print(f"Error calling OpenAI API for ID {id}: {str(e)}")
        # If we get a token limit error, try with truncated input
        if "maximum context length" in str(e) or "token limit" in str(e):
            print(f"  Retrying with truncated input for ID {id}")
            # Truncate input to fit within token limits
            max_chars = 6000  # Adjust based on token limits
            # Keep all content and use remaining space for reasoning
            if content and reasoning:
                content_len = len(content)
                if content_len < max_chars:
                    # Use remaining space for reasoning
                    remaining_chars = max_chars - content_len
                    if len(reasoning) > remaining_chars:
                        reasoning = reasoning[:(len(reasoning) - remaining_chars)] + "... [truncated]"
                else:
                    # Content is already too long, truncate it
                    content = content[:max_chars] + "... [truncated]"
                    reasoning = ""
            elif content and len(content) > max_chars:
                content = content[:max_chars] + "... [truncated]"
            elif reasoning and len(reasoning) > max_chars:
                reasoning = reasoning[:max_chars] + "... [truncated]"
            # Try again with truncated input
            return extract_answer_with_llm_truncated(client, content, reasoning, id)
        return None

def extract_answer_with_llm_truncated(client, content: str, reasoning: str, id: str) -> Optional[str]:
    """Fallback function with truncated input if the original call fails."""
    input_text = f"CONTENT:\n{content}\n\nREASONING:\n{reasoning}"
    
    system_message = """You are an answer extraction assistant. Extract ONLY the letter answer (A, B, C, D, or E).
Return ONLY a single capital letter without any explanation."""
    
    user_message = f"Extract the single letter answer from this truncated response (ID: {id}):\n\n{input_text}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ],
            temperature=0,
            max_tokens=5
        )
        
        answer_text = response.choices[0].message.content.strip()
        letter_match = re.search(r'([A-E])', answer_text, re.IGNORECASE)
        if letter_match:
            return letter_match.group(1).upper()
        return None
    except Exception as e:
        print(f"  Error in fallback extraction for ID {id}: {str(e)}")
        return None

def extract_results(results_file: str, output_file: str = None) -> None:
    """Analyze results from the JSON file using LLM extraction."""
    if not os.path.exists(results_file):
        print(f"Error: File {results_file} not found.")
        return
    
    # Get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables or .env file.")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Load the entire file as a JSON array
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} examples from {results_file}")
    
    # Original results
    original_correct = sum(1 for r in results if r.get('is_correct', False))
    original_accuracy = original_correct / len(results) if results else 0
    
    # Re-extract answers and calculate new accuracy
    new_correct = 0
    errors = []
    improved_results = []
    
    # Process all examples
    for i, result in enumerate(results):
        if i % 10 == 0:
            print(f"Processing example {i+1}/{len(results)}...")
        
        content = result.get('content', '')
        reasoning = result.get('reasoning', '')
        correct_answer = result.get('correct_answer')
        original_predicted = result.get('predicted_answer')
        id = result.get('id', f'example_{i}')
        
        # Extract answer using LLM
        new_predicted = extract_answer_with_llm(client, content, reasoning, id)
        
        # Convert to uppercase for comparison
        if new_predicted:
            new_predicted = new_predicted.upper()
        
        # Check if correct
        is_correct = new_predicted == correct_answer
        if is_correct:
            new_correct += 1
        else:
            errors.append({
                'id': id,
                'original_predicted': original_predicted,
                'new_predicted': new_predicted,
                'correct_answer': correct_answer
            })
        
        # Create improved result
        improved_result = result.copy()
        improved_result['predicted_answer'] = new_predicted
        improved_result['is_correct'] = is_correct
        improved_results.append(improved_result)
        
        # Sleep a bit to avoid hitting rate limits
        time.sleep(0.5)
    
    new_accuracy = new_correct / len(results) if results else 0
    
    # Print results
    print("\n=== RESULTS ===")
    print(f"Original accuracy: {original_accuracy:.4f} ({original_correct}/{len(results)})")
    print(f"New accuracy: {new_accuracy:.4f} ({new_correct}/{len(results)})")
    print(f"Improvement: {new_accuracy - original_accuracy:.4f}")
    
    # Analyze errors
    if errors:
        print(f"\n=== ERRORS ({len(errors)}) ===")
        print("Sample of errors:")
        for error in random.sample(errors, min(5, len(errors))):  # Show random 5 errors
            print(f"ID: {error['id']}")
            print(f"  Original: {error['original_predicted']} | New: {error['new_predicted']} | Correct: {error['correct_answer']}")
    
    # Count error types
    if errors:
        error_types = Counter()
        for error in errors:
            if error['new_predicted'] is None:
                error_types['No answer extracted'] += 1
            elif error['new_predicted'] != error['correct_answer']:
                error_types['Wrong answer extracted'] += 1
        
        print("\n=== ERROR TYPES ===")
        for error_type, count in error_types.items():
            print(f"{error_type}: {count} ({count/len(errors):.2%} of errors)")
    
    # Save improved results if output file is specified
    if output_file:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(improved_results, f, indent=2)
        print(f"\nImproved results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Analyze model results with LLM-based answer extraction")
    parser.add_argument("--input", type=str, required=True, help="Path to the results JSON file")
    parser.add_argument("--output", type=str, default=None, help="Path to save improved results (optional)")
    args = parser.parse_args()
    
    extract_results(args.input, args.output)

if __name__ == "__main__":
    main() 