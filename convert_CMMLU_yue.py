import os
import json
import argparse
import re

def extract_question_and_options(prompt):
    if isinstance(prompt, list) and len(prompt) > 0 and 'prompt' in prompt[0]:
        prompt = prompt[0]['prompt']
    
    # Split the prompt into examples and the actual question
    parts = prompt.split("\n\n以下係關於")
    if len(parts) > 1:
        actual_question = parts[-1]
    else:
        actual_question = prompt

    # Extract the question
    question_match = re.search(r'問題：(.*?)(?:\n[A-D]\.|\n答案：)', actual_question, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    
    # Extract options
    options = re.findall(r'([A-D]\. .*?)(?:\n|$)', actual_question)
    options = [[option.split('. ')[0], option.split('. ')[1].strip()] for option in options]
    
    return question, options

def convert_json(input_file, output_file, model_name, dataset_name):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    for item_id, item in data.items():
        question, options = extract_question_and_options(item['origin_prompt'])
        
        llm_output = item['prediction']
        if isinstance(llm_output, list):
            llm_output = llm_output[0] if llm_output else ""
        
        converted_item = {
            "model_name": model_name,
            "dataset": dataset_name,
            "key_answer_type": "alphabet_option",
            "question": question,
            "llm_output": llm_output,
            "correct_answer": item['gold'],
            "standard_answer_range": options
        }
        converted_data.append(converted_item)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for model_folder in os.listdir(input_folder):
        model_path = os.path.join(input_folder, model_folder)
        if not os.path.isdir(model_path):
            continue
        
        for json_file in os.listdir(model_path):
            if not json_file.endswith('.json'):
                continue
            
            input_file = os.path.join(model_path, json_file)
            
            # Extract dataset name and shot type from filename
            parts = json_file.split('-')
            dataset_name = f"CMMLU-{parts[2]}"
            shot_type = parts[-1].split('.')[0]
            
            output_file = os.path.join(output_folder, f"{model_folder}_{dataset_name}_cantonese_{shot_type}.json")
            convert_json(input_file, output_file, model_folder, dataset_name)
            print(f"Processed: {input_file} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert CMMLU JSON files to a unified format.")
    parser.add_argument("--input_folder", default='/mnt/petrelfs/chenpengan/xFinder/CantoneseBenchmark/CMMLU-yue', 
                        help="Path to the input folder containing JSON files")
    parser.add_argument("--output_folder", default='/mnt/petrelfs/chenpengan/xFinder/CantoneseBenchmark/converted_CMMLU_yue', 
                        help="Path to the output folder for converted JSON files")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
    print("Conversion completed.")

if __name__ == "__main__":
    main()