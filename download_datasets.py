"""Download and save StereoSet and WinoGender datasets for local use."""

import json
import re
import csv
from pathlib import Path
from urllib.request import urlopen
from datasets import load_dataset
import math
from collections import Counter


def download_stereoset():
    """Download StereoSet dataset and save to local file."""
    print("Downloading StereoSet dataset...")
    
    # StereoSet only has 'validation' split, we'll use it as test
    dataset = load_dataset("stereoset", "intrasentence", split="validation")
    
    print(f"Loaded {len(dataset)} examples from StereoSet validation split")
    
    # Save to JSON
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    stereoset_file = data_dir / "stereoset_test.json"
    
    examples = []
    for item in dataset:
        examples.append({
            "context": item["context"],
            "sentences": item["sentences"]  # List of {sentence, gold_label}
        })
    
    with open(stereoset_file, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved {len(examples)} StereoSet examples to {stereoset_file}")
    return examples

# Extract pronoun from sentence
def extract_pronoun(sentence):
    """Extract pronoun from sentence."""
    # Common pronouns in WinoGender
    pronouns = ["he", "she", "him", "her", "his", "hers", "they", "them", "their"]
    for p in pronouns:
        # Check for pronoun with word boundaries
        pattern = r'\b' + re.escape(p) + r'\b'
        if re.search(pattern, sentence, re.IGNORECASE):
            return p.lower()
    return ""

# Create template from sentence by replacing pronoun
def create_template(sentence, pronoun):
    """Create template by replacing pronoun with {pronoun} placeholder."""
    if not pronoun:
        return sentence
    template = sentence
    # Handle different pronoun forms with appropriate placeholders
    if pronoun in ['his', 'her', 'hers']:
        # Possessive forms use {pronoun1}
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        template = re.sub(pattern, '{pronoun1}', template, flags=re.IGNORECASE)
    elif pronoun in ['him', 'her']:
        # Object forms use {pronoun2}
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        template = re.sub(pattern, '{pronoun2}', template, flags=re.IGNORECASE)
    else:
        # Subject forms (he, she, they) use {pronoun}
        pattern = r'\b' + re.escape(pronoun) + r'\b'
        template = re.sub(pattern, '{pronoun}', template, flags=re.IGNORECASE)
    return template
    
def download_winogender():
    """Download WinoGender dataset from TSV file and save to local JSON."""    
    tsv_url = "https://raw.githubusercontent.com/rudinger/winogender-schemas/master/data/all_sentences.tsv"
    
    print(f"Fetching data from {tsv_url}...")
    response = urlopen(tsv_url)
    data = response.read().decode('utf-8')
    
    # Parse TSV
    lines = data.strip().split('\n')
    reader = csv.DictReader(lines, delimiter='\t')
    
    # Group sentences by example_id (profession.word.number)
    example_groups = {}
    
    for row in reader:
        sentid = row['sentid']
        sentence = row['sentence']
        
        # Parse sentid: format is "profession.word.number.gender.txt"
        # e.g., "technician.customer.1.male.txt"
        parts = sentid.replace('.txt', '').split('.')
        if len(parts) < 4:
            continue
        
        profession = parts[0]
        word = parts[1]
        number = parts[2]
        gender = parts[3]  # male, female, or neutral
        
        example_id = f"{profession}.{word}.{number}"
        
        if example_id not in example_groups:
            example_groups[example_id] = {
                'profession': profession,
                'word': word,
                'number': number,
                'sentences': {}
            }
        
        example_groups[example_id]['sentences'][gender] = sentence


    examples = []
    
    for example_id, group in example_groups.items():
        profession = group['profession']
        word = group['word']
        number = group['number']
        
        for gender, sentence in group['sentences'].items():
            if gender == 'neutral':
                continue
            
            pronoun = extract_pronoun(sentence)
            template = create_template(sentence, pronoun)
            
            # Map gender to answer
            answer = "male" if gender == "male" else "female"
            
            example = {
                "sentence": sentence,
                "profession": profession,
                "pronoun": pronoun,
                "answer": answer,
                "word": word,
                "template": template,
                "example_id": example_id,
            }
            
            examples.append(example)
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    winogender_file = data_dir / "winogender_test.json"
    
    with open(winogender_file, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Downloaded and processed {len(examples)} WinoGender examples")
    print(f"Saved to {winogender_file}")
    return examples


def main():
    """Download both datasets."""
    print("=" * 60)
    print("Downloading datasets for bias activation patching")
    print("=" * 60)
    
    stereoset_data = download_stereoset()
    print()
    winogender_data = download_winogender()
    
    print()
    print("=" * 60)
    print("Dataset download complete!")
    print(f"StereoSet: {len(stereoset_data)} examples")
    print(f"WinoGender: {len(winogender_data)} examples")
    print("=" * 60)


if __name__ == "__main__":
    main()

