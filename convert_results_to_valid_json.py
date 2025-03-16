import re
import json
import os

if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert LLM results to valid JSON format')
    parser.add_argument('--input_file', type=str, help='Path to the input file containing LLM results')
    parser.add_argument('--output_file', type=str, help='Path to save the converted JSON output')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get input and output file paths
    input_file = args.input_file
    output_file = args.output_file
    with open(input_file, 'r') as f:
                content = f.read()
                
                # Find all JSON objects using regex pattern matching
                LLM_result = []
                pattern = r'\{\s*"id".*?"is_correct":\s*(true|false)\s*\}'
                matches = re.finditer(pattern, content, re.DOTALL)
                
                for match in matches:
                    try:
                        json_obj = json.loads(match.group(0))
                        LLM_result.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON object: {e}")
                        print(f"Problematic text: {match.group(0)[:100]}...")
                
                # Write the results to the file
                with open(output_file, 'w') as outfile:
                    json.dump(LLM_result, outfile, indent=4)
                
                print(f"Results successfully written to {output_file}")