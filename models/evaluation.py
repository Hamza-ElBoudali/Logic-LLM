import re
import json
import os
import argparse

# these functions are heavily influenced by the HF squad_metrics.py script
def normalize_text(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact_match(prediction, truth):
    return int(normalize_text(prediction) == normalize_text(truth))
    # return prediction == truth

def compute_f1(prediction, truth):
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()
    
    # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    # if there are no common tokens then f1 = 0
    if len(common_tokens) == 0:
        return 0
    
    prec = len(common_tokens) / len(pred_tokens)
    rec = len(common_tokens) / len(truth_tokens)
    
    return 2 * (prec * rec) / (prec + rec)

def evaluate_sample(prediction, gold_answers):
    em_score = max((compute_exact_match(prediction, answer)) for answer in gold_answers)
    f1_score = max((compute_f1(prediction, answer)) for answer in gold_answers)
    return em_score, f1_score

def get_choice(answer_str):
    choices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'A)', 'B)', 'C)', 'D)', 'E)', 'F)', 'G)', 'H)', 
               'A.', 'B.', 'C.', 'D.', 'E.', 'F.', 'G.', 'H.']
    for c in choices:
        if answer_str.startswith(c):
            return c.replace(')', '').replace('.', '')

    if answer_str.startswith(':'):
       return answer_str.replace(':', '').replace('.', '').strip()
    return None

def evaluate_QA(QA_results):
    total_em = 0.0
    count = 0
    for sample in QA_results:
        gold_answer = sample['answer'].replace('(', '').replace(')', '').strip()
        answer_str = sample['predicted_answer'].strip() if sample['predicted_answer'] is not None else ''
        prediction = get_choice(answer_str)

        indicators = ['the correct option is', 'the correct answer is', 
                      'The correct answer is', 'The correct option is',
                      'Thus, the answer is', 'answer is', 'Answer:']
        if prediction is None:
            for indicator in indicators:
                if answer_str.find(indicator)>=0:
                    answer_str = answer_str.split(indicator)[1].strip()
                    prediction = get_choice(answer_str)
                    break

        # If still None, try to find a single letter answer
        if prediction is None:
            letter_match = re.search(r'\b([A-E])\b', answer_str)
            if letter_match:
                prediction = letter_match.group(1)

        # if prediction is None:
        #     print(answer_str)
        # print(f"prediction: {prediction} \t gold_answers: {gold_answer} \t match: {prediction == gold_answer}")
        
        em_score = 1.0 if prediction == gold_answer else 0.0
        total_em += em_score
        count += 1
    
    avg_em = total_em / count
    # print(f"Accuracy: {avg_em}")
    return avg_em

def full_evaluation(result_file):
    with open(result_file, 'r') as f:
        all_samples = json.load(f)

    executable_samples = [sample for sample in all_samples if sample['flag'] == 'success']
    overall_acc = evaluate_QA(all_samples)
    exe_rate = len(executable_samples)/len(all_samples)
    exe_acc = evaluate_QA(executable_samples) if executable_samples else 0.0
    
    print(f"Overall accuracy: {overall_acc:.4f}")
    print(f'Executable rate (Exe_Rate): {exe_rate:.4f}')
    print(f"Executable accuracy (Exe_Acc): {exe_acc:.4f}")
    
    # Save metrics to a separate file
    metrics_file = result_file.replace('.json', '_metrics.json')
    metrics = {
        "overall_accuracy": overall_acc,
        "executable_rate": exe_rate,
        "executable_accuracy": exe_acc,
        "total_samples": len(all_samples),
        "executable_samples": len(executable_samples)
    }
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_name", type=str, default='text-davinci-003')
    parser.add_argument("--split", type=str, default='dev')
    parser.add_argument("--backup", type=str, default='random')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # Handle model name with special characters for file paths
    safe_model_name = args.model_name.replace('/', '_').replace('\\', '_').replace(':', '_')
    
    result_path = f'./outputs/results'
    result_file = os.path.join(result_path, f'{args.dataset_name}_{args.split}_{safe_model_name}_backup-{args.backup}.json')
    
    if not os.path.exists(result_file):
        print(f"Warning: Result file not found at {result_file}")
        # Try alternative paths
        alt_result_path = f'./outputs/logic_inference'
        alt_result_file = os.path.join(alt_result_path, f'{args.dataset_name}_{args.split}_{safe_model_name}_backup-{args.backup}.json')
        if os.path.exists(alt_result_file):
            print(f"Found result file at alternative location: {alt_result_file}")
            result_file = alt_result_file
        else:
            print(f"Error: Could not find result file at {result_file} or {alt_result_file}")
            exit(1)
    
    print(f"Evaluating results from: {result_file}")
    full_evaluation(result_file)