import os
import json
import argparse
import re

def extract_question_and_options(prompt):
    if isinstance(prompt, list):
        prompt = prompt[0]['prompt'] if prompt and isinstance(prompt[0], dict) else str(prompt[0])
    
    # Split the prompt into examples and the actual question
    parts = prompt.split("\n\n問題：")
    actual_question = parts[-1].strip()
    
    # Extract the question without options and remove "問題：" prefix
    question_match = re.search(r'(?:問題：)?\s*(.*?)(?:\n[A-D]\.|\n由提供嘅選項中)', actual_question, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    
    # Extract options
    options = re.findall(r'([A-D]\. .*?)(?:\n|$)', actual_question)
    options = [[option.split('. ')[0], option.split('. ')[1].strip()] for option in options]
    
    return question, options

def convert_json(input_file, output_file, model_name):
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
            "dataset": "ARC-C-yue",
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
        
        for json_file in ['ARC-c_0shot.json', 'ARC-c_5shot.json']:
            input_file = os.path.join(model_path, json_file)
            if not os.path.exists(input_file):
                continue
            
            # Update the output file name to include 'cantonese'
            shot_type = json_file.split('_')[-1].split('.')[0]
            output_file = os.path.join(output_folder, f"{model_folder}_ARC-c_cantonese_{shot_type}.json")
            convert_json(input_file, output_file, model_folder)
            print(f"Processed: {input_file} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert ARC JSON files to a unified format.")
    parser.add_argument("--input_folder", default='/mnt/petrelfs/chenpengan/xFinder/CantoneseBenchmark/ARC-C-yue', help="Path to the input folder containing JSON files")
    parser.add_argument("--output_folder",default='/mnt/petrelfs/chenpengan/xFinder/CantoneseBenchmark/converted_ARC_yue', help="Path to the output folder for converted JSON files")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
    print("Conversion completed.")

if __name__ == "__main__":
    main()