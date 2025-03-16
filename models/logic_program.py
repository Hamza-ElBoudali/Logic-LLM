# generate facts and rules based on the problem description

import json
import os
import time
import sys
from tqdm import tqdm
from collections import OrderedDict
from typing import Dict, List, Tuple
from utils import OpenAIModel, DeepSeekModel
import argparse
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class LogicProgramGenerator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.split = args.split
        self.model_name = args.model_name
        self.safe_model_name = self.model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        self.save_path = args.save_path
        self.num_threads = args.num_threads
        self.verbose = args.verbose if hasattr(args, 'verbose') else False
        self.timeout = args.timeout if hasattr(args, 'timeout') else 60

        # Initialize the appropriate model based on the model name
        if 'deepseek' in args.model_name.lower():
            if self.verbose:
                print(f"Initializing DeepSeekModel with timeout={self.timeout}s")
            self.model_api = DeepSeekModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        else:
            if self.verbose:
                print(f"Initializing OpenAIModel with timeout={self.timeout}s")
            self.model_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
            
        self.prompt_creator = {'FOLIO': self.prompt_folio,
                               'ProntoQA': self.prompt_prontoqa,
                               'ProofWriter': self.prompt_proofwriter,
                               'LogicalDeduction': self.prompt_logicaldeduction, 
                               'AR-LSAT': self.prompt_arlsat}
        self.load_prompt_templates()
    
    def load_prompt_templates(self):
        prompt_file = f'./models/prompts/{self.dataset_name}.txt'
        if self.dataset_name == 'AR-LSAT' and self.model_name == 'gpt-4':
            prompt_file = f'./models/prompts/{self.dataset_name}-long.txt'
        with open(prompt_file, 'r') as f:
            self.prompt_template = f.read()
        if self.verbose:
            print(f"Loaded prompt template from {prompt_file}")

    def prompt_folio(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt

    def prompt_arlsat(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt
    
    def prompt_prontoqa(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_proofwriter(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        return full_prompt
    
    def prompt_logicaldeduction(self, test_data):
        problem = test_data['context']
        question = test_data['question'].strip()
        choices_str = '\n'.join([f'({choice.strip()}' for choice in test_data['options']]).strip()
        full_prompt = self.prompt_template.replace('[[PROBLEM]]', problem).replace('[[QUESTION]]', question)
        full_prompt = full_prompt.replace('[[CHOICES]]', choices_str)
        return full_prompt

    def load_raw_dataset(self, split):
        dataset_path = os.path.join(self.data_path, self.dataset_name, f'{split}.json')
        if self.verbose:
            print(f"Loading dataset from {dataset_path}")
        with open(dataset_path) as f:
            raw_dataset = json.load(f)
        return raw_dataset

    def process_example(self, example):
        """Process a single example with the model."""
        try:
            if self.verbose:
                print(f"Processing example {example.get('id', 'unknown')}")
                start_time = time.time()
            
            prompt = self.prompt_creator[self.dataset_name](example)
            
            if self.verbose:
                print(f"Generated prompt of length {len(prompt)}")
                print(f"Sending request to model API...")
            
            output = self.model_api.generate(prompt, temperature=0.0)
            
            if self.verbose:
                elapsed = time.time() - start_time
                print(f"Received response in {elapsed:.2f} seconds")
                print(f"Response length: {len(output)}")
            
            example['logic_program'] = output
            return example
        except Exception as e:
            print(f"Error processing example {example.get('id', 'unknown')}: {e}")
            if self.verbose:
                import traceback
                traceback.print_exc()
            example['logic_program'] = ""
            return example

    def threaded_logic_program_generation(self):
        """Generate logic programs using multi-threading."""
        # Load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")
        
        # Process examples in parallel
        results = []
        start_time = time.time()
        
        print(f"Using multi-threading with {self.num_threads} threads...")
        
        output_file = os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.safe_model_name}.json')        
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # For debugging, process just the first example if verbose mode is on
        # if self.verbose:
        #     print("DEBUG MODE: Processing only the first example...")
        #     example = raw_dataset[0]
        #     result = self.process_example(example)
        #     results.append(result)
            
        #     with open(output_file, 'w') as f:
        #         json.dump(results, f, indent=2)
            
        #     elapsed = time.time() - start_time
        #     print(f"Processed 1 example. Elapsed time: {elapsed:.2f}s")
            
        #     return results
        
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            # Submit all tasks
            futures = [executor.submit(self.process_example, example) for example in raw_dataset]
            
            # Process results as they complete
            for i, future in enumerate(tqdm(futures, desc="Processing examples")):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Save intermediate results
                    if (i + 1) % 10 == 0 or (i + 1) == len(raw_dataset):
                        with open(output_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        elapsed = time.time() - start_time
                        print(f"Processed {i+1}/{len(raw_dataset)} examples. Elapsed time: {elapsed:.2f}s, Avg: {elapsed/(i+1):.2f}s per example")
                except Exception as e:
                    print(f"Error processing future {i}: {e}")
        
        # Save final results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"Processing complete. Total time: {total_time:.2f}s, Avg: {total_time/len(raw_dataset):.2f}s per example")
        
        return results

    def logic_program_generation(self):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        output_file = os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.safe_model_name}.json')

        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # For debugging, process just the first example if verbose mode is on
        if self.verbose:
            print("DEBUG MODE: Processing only the first example...")
            example = raw_dataset[0]
            start_time = time.time()
            result = self.process_example(example)
            results = [result]
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            elapsed = time.time() - start_time
            print(f"Processed 1 example. Elapsed time: {elapsed:.2f}s")
            
            return results

        # generate logic programs
        results = []
        start_time = time.time()
        
        for i, sample in enumerate(tqdm(raw_dataset)):
            prompt = self.prompt_creator[self.dataset_name](sample)
            output = self.model_api.generate(prompt, temperature=0.0)
            
            # update sample
            sample['logic_program'] = output
            results.append(sample)

            # save intermediate results
            if (i + 1) % 10 == 0:
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)
                elapsed = time.time() - start_time
                print(f"Processed {i+1}/{len(raw_dataset)} examples. Elapsed time: {elapsed:.2f}s, Avg: {elapsed/(i+1):.2f}s per example")

        # save final results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        total_time = time.time() - start_time
        print(f"Processing complete. Total time: {total_time:.2f}s, Avg: {total_time/len(raw_dataset):.2f}s per example")
        
        return results

    def batch_logic_program_generation(self, batch_size = 10):
        # load raw dataset
        raw_dataset = self.load_raw_dataset(self.split)
        print(f"Loaded {len(raw_dataset)} examples from {self.split} split.")

        output_file = os.path.join(self.save_path, f'{self.dataset_name}_{self.split}_{self.safe_model_name}.json')
        
        # Ensure the directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # For debugging, process just the first batch if verbose mode is on
        if self.verbose:
            print("DEBUG MODE: Processing only the first batch...")
            batch = raw_dataset[:min(batch_size, len(raw_dataset))]
            prompts = [self.prompt_creator[self.dataset_name](sample) for sample in batch]
            
            print(f"Generated {len(prompts)} prompts for the first batch")
            
            # Use asyncio to run the batch generation
            import asyncio
            start_time = time.time()
            outputs = asyncio.run(self.model_api.batch_generate(prompts, temperature=0.0))
            elapsed = time.time() - start_time
            
            print(f"Batch processing completed in {elapsed:.2f} seconds")
            
            # update samples
            results = []
            for j, (sample, output) in enumerate(zip(batch, outputs)):
                sample['logic_program'] = output
                results.append(sample)
            
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            return results

        # generate logic programs in batches
        results = []
        for i in tqdm(range(0, len(raw_dataset), batch_size)):
            batch = raw_dataset[i:i+batch_size]
            prompts = [self.prompt_creator[self.dataset_name](sample) for sample in batch]
            
            # Use asyncio to run the batch generation
            import asyncio
            outputs = asyncio.run(self.model_api.batch_generate(prompts, temperature=0.0))
            
            # update samples
            for j, (sample, output) in enumerate(zip(batch, outputs)):
                sample['logic_program'] = output
                results.append(sample)

            # save intermediate results
            if (i + batch_size) % 50 == 0 or (i + batch_size) >= len(raw_dataset):
                with open(output_file, 'w') as f:
                    json.dump(results, f, indent=2)

        # save final results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='AR-LSAT')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--save_path', type=str, default='./outputs/logic_programs')
    parser.add_argument('--api_key', type=str, default=os.getenv("OPENROUTER_API_KEY"), help='API key (defaults to OPENROUTER_API_KEY from .env)')
    parser.add_argument('--stop_words', type=str, nargs='+', default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--num_threads', type=int, default=15, help='Number of threads to use for processing')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output for debugging')
    parser.add_argument('--timeout', type=int, default=60, help='Timeout in seconds for API requests')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize the generator
    generator = LogicProgramGenerator(args)
    
    # Generate logic programs using threading
    results = generator.threaded_logic_program_generation()