import argparse
import json
import sys
import csv
import os
import re

import yaml
from tqdm import tqdm

from .core import Comparator, Extractor
from .utils import DataProcessor


def check_config(config):
    if 'xfinder_model' not in config:
        raise ValueError("Error: 'xfinder_model' not found in the configuration file.")
    if 'model_name' not in config['xfinder_model']:
        raise ValueError("Error: 'model_name' of xfinder not found in the configuration file.")
    if 'model_path' not in config['xfinder_model'] and 'url' not in config['xfinder_model']:
        raise ValueError("Error: 'model_path' or 'url' of xfinder not found in the configuration file.")
    if 'data_path' not in config:
        raise ValueError("Error: 'data_path' not found in the configuration file.")


def extract_info_from_filename(filename):
    pattern = r'(.+)_(.+?)_(.+?)_(\d+shot)\.json'
    match = re.match(pattern, filename)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4)
    return None, None, None, None


def process_file(file_path, extractor, comparator):
    data_processor = DataProcessor()
    ori_data = data_processor.read_data(file_path)
    ext_cor_pairs = []
    for item in tqdm(ori_data, desc=f"Processing {os.path.basename(file_path)}"):
        user_input = extractor.prepare_input(item)
        extracted_answer = extractor.gen_output(user_input)
        ext_cor_pairs.append([
            item["key_answer_type"], item["standard_answer_range"],
            extracted_answer, item["correct_answer"]
        ])

    results = comparator.compare_all(ext_cor_pairs)

    for item, result in zip(ori_data, results):
        item['is_correct'] = result[-1]

    correct = sum(1 for result in results if result[-1] == True)
    total = max(len(results), 1)
    accuracy = correct / total if total else 0

    # Overwrite the original file
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(ori_data, json_file, ensure_ascii=False, indent=2)

    return accuracy


def calc_acc(config_path: str) -> None:
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    check_config(config)

    extractor = Extractor(
        model_name=config['xfinder_model']['model_name'],
        model_path=config['xfinder_model'].get('model_path'),
        url=config['xfinder_model'].get('url'),
    )
    comparator = Comparator()

    data_folder = config['data_path']
    csv_results = []

    for filename in os.listdir(data_folder):
        if filename.endswith('.json'):
            file_path = os.path.join(data_folder, filename)
            model_name, dataset_name, language, shot_type = extract_info_from_filename(filename)
            
            if model_name and dataset_name and language and shot_type:
                accuracy = process_file(file_path, extractor, comparator)
                csv_results.append([model_name, dataset_name, language, shot_type, f"{accuracy:.4f}"])
                print(f"Processed {filename}: Accuracy = {accuracy:.4f}")

    # Write summary to CSV
    csv_path = os.path.join(data_folder, 'summary_results.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Model Name', 'Dataset', 'Language', 'Shot Type', 'Accuracy'])
        writer.writerows(csv_results)

    print(f"Summary CSV has been written to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description='Run xFinder evaluation.')
    parser.add_argument(
        'config_path',
        nargs='?',
        default=None,
        help='Path to the configuration file')
    args = parser.parse_args()

    config_path = args.config_path
    if not config_path:
        print("Error: No configuration path provided.")
        parser.print_help()
        return

    return calc_acc(config_path)


if __name__ == "__main__":
    main()