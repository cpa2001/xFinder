import os
import json
import argparse
import re

def extract_question_0shot(prompt):
    if isinstance(prompt, list):
        prompt = prompt[0]['prompt']
    match = re.search(r'問題：(.*?)\n用粵語', prompt, re.DOTALL)
    return match.group(1).strip() if match else ""

def extract_question_5shot(prompt):
    parts = prompt.split("請逐步思考，最終答案前用「####」標記。用粵語答下面問題：\n問題：")
    if len(parts) > 1:
        question = parts[-1].split("\n用粵語")[0].strip()
        return question
    return ""

def convert_json(input_file, output_file, model_name):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    is_5shot = '5shot' in input_file
    extract_func = extract_question_5shot if is_5shot else extract_question_0shot
    
    converted_data = []
    for item_id, item in data.items():
        question = extract_func(item['origin_prompt'])
        # 对于 0-shot 情况，移除开头的 "問題："
        if not is_5shot and question.startswith("問題："):
            question = question[3:].strip()
        llm_output = item['prediction'][0] if isinstance(item['prediction'], list) else item['prediction']
        converted_item = {
            "model_name": model_name,
            "dataset": "GSM8K-yue",
            "key_answer_type": "math",
            "question": question,
            "llm_output": llm_output,
            "correct_answer": item['gold'].split("####")[-1].strip(),
            "standard_answer_range": "a(n) number / set / vector / matrix / interval / expression / function / equation / inequality"
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
        
        for json_file in ['gsm8k_cantonese_0shot.json', 'gsm8k_cantonese_5shot.json']:
            input_file = os.path.join(model_path, json_file)
            if not os.path.exists(input_file):
                continue
            
            output_file = os.path.join(output_folder, f"{model_folder}_{json_file}")
            convert_json(input_file, output_file, model_folder)
            print(f"Processed: {input_file} -> {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Convert JSON files to a unified format.")
    parser.add_argument("--input_folder",default='/mnt/petrelfs/chenpengan/xFinder/CantoneseBenchmark/GSM8K-yue', help="Path to the input folder containing JSON files")
    parser.add_argument("--output_folder",default='/mnt/petrelfs/chenpengan/xFinder/CantoneseBenchmark/converted_GSM8K_yue', help="Path to the output folder for converted JSON files")
    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder)
    print("Conversion completed.")

if __name__ == "__main__":
    main()