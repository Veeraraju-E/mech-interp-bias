"""Download and save StereoSet and WinoGender datasets for local use."""

import json
from pathlib import Path
from datasets import load_dataset


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


def download_winogender():
    """Download WinoGender dataset and save to local file."""
    print("Downloading WinoGender dataset...")
    
    # Try different sources
    dataset = None
    try:
        # Try test split first
        dataset = load_dataset("ucinlp/unstereo-eval", "Winogender", split="test")
        print(f"Loaded from 'ucinlp/unstereo-eval' Winogender test split")
    except:
        try:
            # Try validation split
            dataset = load_dataset("ucinlp/unstereo-eval", "Winogender", split="validation")
            print(f"Loaded from 'ucinlp/unstereo-eval' Winogender validation split")
        except:
            # Try alternative source
            try:
                dataset = load_dataset("rudinger/winogender", split="test")
                print(f"Loaded from 'rudinger/winogender' test split")
            except:
                dataset = load_dataset("rudinger/winogender", split="validation")
                print(f"Loaded from 'rudinger/winogender' validation split")
    
    if dataset is None:
        raise ValueError("Could not load WinoGender dataset from any source")
    
    print(f"Loaded {len(dataset)} examples from WinoGender")
    
    # Save to JSON
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    winogender_file = data_dir / "winogender_test.json"
    
    examples = []
    for item in dataset:
        sentence = item.get("sentence", item.get("text", ""))
        template = item.get("template", "")
        word = item.get("word", "")
        
        # Extract profession from sentence or template
        # WinoGender format: "The [profession] told the [other] that [pronoun]..."
        profession = ""
        pronoun = ""
        
        # Try to extract profession from the sentence
        if sentence:
            # Common pattern: "The [profession] told..."
            parts = sentence.split(" told ")
            if len(parts) > 0:
                profession_part = parts[0].replace("The ", "").strip()
                profession = profession_part
        
        # Extract pronoun from sentence
        pronouns = ["he", "she", "him", "her", "his", "hers"]
        for p in pronouns:
            if f" {p} " in sentence or sentence.endswith(f" {p}"):
                pronoun = p
                break
        
        # Determine gender from pronoun or other indicators
        answer = ""
        if pronoun in ["he", "him", "his"]:
            answer = "male"
        elif pronoun in ["she", "her", "hers"]:
            answer = "female"
        elif "max_gender_pmi" in item:
            # Use PMI to determine gender if available
            pmi = item.get("max_gender_pmi", 0)
            if pmi > 0:
                answer = "male"
            elif pmi < 0:
                answer = "female"
        
        example = {
            "sentence": sentence,
            "profession": profession,
            "pronoun": pronoun,
            "answer": answer,
            "word": word,  # The other entity (customer, etc.)
            "template": template
        }
        
        # Add any additional useful fields
        if "example_id" in item:
            example["example_id"] = item["example_id"]
        if "max_gender_pmi" in item:
            example["max_gender_pmi"] = item["max_gender_pmi"]
        
        examples.append(example)
    
    with open(winogender_file, "w") as f:
        json.dump(examples, f, indent=2)
    
    print(f"Saved {len(examples)} WinoGender examples to {winogender_file}")
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

