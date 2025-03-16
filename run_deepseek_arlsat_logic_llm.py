import os
import argparse
import subprocess
import json
import time
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser(description="Run DeepSeek-R1 on AR-LSAT using the Logic-LLM framework")
    parser.add_argument('--api_key', type=str, default=os.getenv("OPENROUTER_API_KEY"), help='OpenRouter API key (defaults to OPENROUTER_API_KEY from .env)')
    parser.add_argument('--model_name', type=str, default='deepseek_deepseek-r1:free', help='DeepSeek model name')
    parser.add_argument('--split', type=str, default='test', help='Dataset split to use')
    parser.add_argument('--num_threads', type=int, default=15, help='Number of threads to use for processing')
    parser.add_argument('--max_new_tokens', type=int, default=1024, help='Maximum number of tokens to generate')
    parser.add_argument('--backup_strategy', type=str, default='random', help='Backup strategy for answer generation')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds for API requests')
    return parser.parse_args()

def run_logic_program_generation(args, safe_model_name):
    """Run the logic program generation step."""
    print("Step 1: Generating logic programs...")
    
    cmd = [
        "python", "models/logic_program.py",
        "--data_path", "data",
        "--dataset_name", "AR-LSAT",
        "--split", args.split,
        "--model_name", args.model_name,
        "--save_path", "./outputs/logic_programs",
        "--api_key", args.api_key,
        "--max_new_tokens", str(args.max_new_tokens),
        "--num_threads", str(args.num_threads)
    ]
    
    if args.verbose:
        cmd.append("--verbose")
        cmd.append("--timeout")
        cmd.append(str(args.timeout))
        print(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    process = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    if args.verbose:
        print(f"Logic program generation completed in {elapsed:.2f} seconds")
        print(f"Return code: {process.returncode}")
    
    # Check if the output file exists
    output_file = f'./outputs/logic_programs/AR-LSAT_{args.split}_{safe_model_name}.json'
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} not found.")
        return False
    
    return True

def run_logic_inference(args, safe_model_name):
    """Run the logic inference step."""
    print("Step 2: Running logic inference...")
    cmd = [
        "python", "models/logic_inference.py",
        "--dataset_name", "AR-LSAT",
        "--split", args.split,
        "--model_name", args.model_name,
        "--save_path", "./outputs/results",
        "--backup_strategy", args.backup_strategy,
        "--backup_LLM_result_path", f"./baselines/results/AR-LSAT_{args.split}_{safe_model_name}_backup-{args.backup_strategy}.json"
    ]
    
    if args.verbose:
        cmd.append("--verbose")
        print(f"Running command: {' '.join(cmd)}")
    
    start_time = time.time()
    process = subprocess.run(cmd)
    elapsed = time.time() - start_time
    
    if args.verbose:
        print(f"Logic inference completed in {elapsed:.2f} seconds")
        print(f"Return code: {process.returncode}")
    
    # Check if the output file exists
    output_file = f"./outputs/results/AR-LSAT_{args.split}_{safe_model_name}_backup-{args.backup_strategy}.json"
    if not os.path.exists(output_file):
        print(f"Error: Output file {output_file} not found.")
        return False
    
    return True

def evaluate_results(args, safe_model_name):
    """Evaluate the results."""
    print("Step 3: Evaluating results...")
    
    # Load the results
    results_file = f"./outputs/results/AR-LSAT_{args.split}_{safe_model_name}_backup-{args.backup_strategy}.json"
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    # Calculate accuracy
    correct = 0
    total = 0
    for result in results:
        if 'predicted_answer' in result and 'answer' in result:
            predicted = result['predicted_answer'].strip()
            gold = result['answer'].strip()
            
            if predicted == gold:
                correct += 1
            total += 1
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # Save evaluation results
    eval_results = {
        "model": args.model_name,
        "dataset": "AR-LSAT",
        "split": args.split,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "num_threads": args.num_threads
    }
    
    eval_dir = "./outputs/evaluations"
    os.makedirs(eval_dir, exist_ok=True)
    
    eval_file = f"{eval_dir}/AR-LSAT_{args.split}_{safe_model_name}_eval.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"Evaluation results saved to {eval_file}")
    
    return True

def main():
    args = parse_args()
    
    # Create output directories
    os.makedirs("./outputs/logic_programs", exist_ok=True)
    os.makedirs("./outputs/results", exist_ok=True)
    
    # Print configuration
    print("\n=== Configuration ===")
    print(f"Model: {args.model_name}")
    print(f"Dataset Split: {args.split}")
    print(f"Number of Threads: {args.num_threads}")
    print(f"Max Tokens: {args.max_new_tokens}")
    print(f"Backup Strategy: {args.backup_strategy}")
    print(f"Verbose Mode: {'Enabled' if args.verbose else 'Disabled'}")
    print(f"API Timeout: {args.timeout} seconds")
    print("===================\n")

    safe_model_name = args.model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    # Run the pipeline
    if args.verbose:
        print("Starting pipeline in verbose mode...")
    
    if run_logic_program_generation(args, safe_model_name):
        if args.verbose:
            print("Logic program generation successful!")
        if run_logic_inference(args, safe_model_name):
            if args.verbose:
                print("Logic inference successful!")
            evaluate_results(args, safe_model_name)
    
    print("Done!")

if __name__ == "__main__":
    main() 