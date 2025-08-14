import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import argparse
import socket


def apply_chat_template(tokenizer, prompts: List[str]) -> List[str]:
    """Apply chat template to prompts."""
    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
        ]
        formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return formatted


def tokenize_and_find_max_length(tokenizer, prompts: List[str], use_chat_template: bool = True) -> dict:
    """Tokenize prompts and find maximum token length."""
    print("Tokenizing prompts...")
    
    # Apply chat template if requested
    if use_chat_template:
        formatted_prompts = apply_chat_template(tokenizer, prompts)
    else:
        formatted_prompts = prompts
    
    token_lengths = []
    max_length = 0
    max_length_prompt = ""
    max_length_tokens = None
    
    for i, prompt in enumerate(tqdm(formatted_prompts, desc="Tokenizing")):
        # Tokenize the prompt
        tokens = tokenizer.encode(prompt, add_special_tokens=True)
        length = len(tokens)
        token_lengths.append(length)
        
        # Track maximum length
        if length > max_length:
            max_length = length
            max_length_prompt = prompt
            max_length_tokens = tokens
    
    return {
        'max_length': max_length,
        'max_length_prompt': max_length_prompt,
        'max_length_tokens': max_length_tokens,
        'token_lengths': token_lengths,
        'mean_length': np.mean(token_lengths),
        'median_length': np.median(token_lengths),
        'std_length': np.std(token_lengths),
        'min_length': min(token_lengths)
    }


def main():
    # TORCH_CUDA_ARCH_LIST Configuration
    print()
    print("="*70)
    TORCH_CUDA_ARCH_LIST = os.environ.get('TORCH_CUDA_ARCH_LIST')
    if TORCH_CUDA_ARCH_LIST:
        print(f"Using TORCH_CUDA_ARCH_LIST from environment variable: {TORCH_CUDA_ARCH_LIST}")
    else: 
        print("TORCH_CUDA_ARCH_LIST environment variable not set. Attempting to infer from a CUDA device...")
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            TORCH_CUDA_ARCH_LIST = f"{major}.{minor}"
            print(f"Inferred TORCH_CUDA_ARCH_LIST: {TORCH_CUDA_ARCH_LIST}")
        else:
            print("Warning: No CUDA-enabled GPU is available. Continuing without a specific architecture list.")
    if TORCH_CUDA_ARCH_LIST:
        print(f"Proceeding with CUDA Architecture: {TORCH_CUDA_ARCH_LIST}")
    else:
        print("Proceeding in CPU mode or with default CUDA settings.")
    print("="*70)
    print()

    # HF_TOKEN Configuration
    print()
    print("="*70)
    HF_TOKEN = os.environ.get('HF_TOKEN')
    if not HF_TOKEN:
        raise ValueError("HF_TOKEN environment variable not set")
    login(token=HF_TOKEN)
    print("="*70)
    print()

    # HF_CACHE Configuration
    hostname = socket.gethostname()
    if hostname == 'uqhni-Precision-3680':
        HF_CACHE = '/home/pietro/HF-CACHE'
    elif  hostname.startswith('bun'):
        HF_CACHE = '/scratch/user/s4784669/HF-CACHE'
    else:
        HF_CACHE = None
    print()
    print("="*70)
    print(f"Using Hugging Face cache directory: {HF_CACHE}")
    print("="*70)
    print()

    # Argument parser
    parser = argparse.ArgumentParser(description="Analyze token lengths of political compass prompts.")
    parser.add_argument(
        '--use_chat_template', 
        type=bool,
        default=True,
        help="Use chat template for prompts (True/False)."
    )

    parser.add_argument(
        '--use_data_subset', 
        type=bool,
        default=False,
        help="Use a subset of the data for testing (True/False). If False, uses the full dataset."
    )

    parser.add_argument(
        '--subset_size', 
        type=int,
        default=100000,
        help="Size of the data subset to use for testing. Only used if use_data_subset is True."
    )

    parser.add_argument(
        '--model', 
        type=int,
        default=2,
        help="Model to use for tokenization."
    )
    
    parser.add_argument(
        '--persona',
        type=str,
        default='base',
        help="The configuration of the persona descriptions for the inference (base/left/right)."
    )

    parser.add_argument(
        '--output_file',
        type=str,
        default=None,
        help="Output file to save token length statistics (optional)."
    )

    args = parser.parse_args()

    USE_CHAT_TEMPLATE = args.use_chat_template
    USE_DATA_SUBSET = args.use_data_subset
    SUBSET_SIZE = args.subset_size 
    SELECTED_MODEL = args.model
    PERSONA_CONFIGURATION = args.persona
    OUTPUT_FILE = args.output_file

    MODELS = [
        "mistralai/Mistral-7B-Instruct-v0.3", # 0
        "meta-llama/Llama-3.1-8B-Instruct", # 1
        "Qwen/Qwen2.5-7B-Instruct", # 2
        "HuggingFaceH4/zephyr-7b-beta", # 3
        "meta-llama/Llama-3.1-70B-Instruct", # 4
        "meta-llama/Llama-3.3-70B-Instruct", # 5
        "Qwen/Qwen2.5-72B-Instruct" # 6
        ]
    MODEL = MODELS[SELECTED_MODEL]

    print()
    print("="*70)
    print(f"Using chat template: {USE_CHAT_TEMPLATE}")
    print(f"Using data subset: {USE_DATA_SUBSET}")
    if USE_DATA_SUBSET:
        print(f"Using subset size: {SUBSET_SIZE} prompts")
    else:
        print("Using full dataset")
    print(f"Using model: {MODEL}")
    print(f"Using {PERSONA_CONFIGURATION} persona description")
    print("="*70)
    print()
    
    if PERSONA_CONFIGURATION=="base":
        DATA_PATH = '../data/processed/base_political_compass_prompts.pqt'
    elif PERSONA_CONFIGURATION=="left":
        DATA_PATH = '../data/processed/left_libertarian_political_compass_prompts.pqt'
    elif PERSONA_CONFIGURATION=="right":
        DATA_PATH = '../data/processed/right_authoritarian_political_compass_prompts.pqt'
    else:
        raise ValueError(f"{PERSONA_CONFIGURATION} is not a valid configuration for the persona descriptions")

    # Load data
    print()
    print("="*70)
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    if USE_DATA_SUBSET:
        df = df.head(SUBSET_SIZE*62)  # Remove for full dataset
    print(f"Total prompts: {len(df)}")
    print(f"Total personas: {len(df) // 62}")
    
    # Initialize tokenizer
    print(f"\nLoading tokenizer for model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL,
        cache_dir=HF_CACHE,
        trust_remote_code=True
    )
    
    # Get prompts
    prompts = df['prompt'].tolist()
    
    # Analyze token lengths
    print("\nAnalyzing token lengths...")
    results = tokenize_and_find_max_length(tokenizer, prompts, USE_CHAT_TEMPLATE)
    
    # Print results
    print("\n" + "="*70)
    print("TOKEN LENGTH ANALYSIS RESULTS")
    print("="*70)
    print(f"Maximum token length: {results['max_length']}")
    print(f"Minimum token length: {results['min_length']}")
    print(f"Mean token length: {results['mean_length']:.2f}")
    print(f"Median token length: {results['median_length']:.2f}")
    print(f"Standard deviation: {results['std_length']:.2f}")
    print(f"Total prompts analyzed: {len(results['token_lengths'])}")
    
    print("\n" + "="*70)
    print("PROMPT WITH MAXIMUM TOKEN LENGTH:")
    print("="*70)
    print(f"Length: {results['max_length']} tokens")
    print(f"Prompt: {results['max_length_prompt']}")
    
    print("\n" + "="*70)
    print("TOKENS FOR MAXIMUM LENGTH PROMPT:")
    print("="*70)
    print(f"Tokens: {results['max_length_tokens']}")
    
    # Create token length distribution
    token_lengths = results['token_lengths']
    percentiles = [50, 75, 90, 95, 99]
    print("\n" + "="*70)
    print("TOKEN LENGTH DISTRIBUTION:")
    print("="*70)
    for p in percentiles:
        print(f"{p}th percentile: {np.percentile(token_lengths, p):.0f} tokens")
    
    # Save results if output file specified
    if OUTPUT_FILE:
        output_data = {
            'model': MODEL,
            'persona_configuration': PERSONA_CONFIGURATION,
            'use_chat_template': USE_CHAT_TEMPLATE,
            'total_prompts': len(token_lengths),
            'max_length': results['max_length'],
            'min_length': results['min_length'],
            'mean_length': results['mean_length'],
            'median_length': results['median_length'],
            'std_length': results['std_length'],
            'max_length_prompt': results['max_length_prompt'],
            'token_lengths': token_lengths
        }
        
        # Add percentiles
        for p in percentiles:
            output_data[f'percentile_{p}'] = np.percentile(token_lengths, p)
        
        # Save as pickle for full data or json for summary
        if OUTPUT_FILE.endswith('.pkl'):
            import pickle
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(output_data, f)
        elif OUTPUT_FILE.endswith('.json'):
            import json
            # Remove token_lengths list for JSON (too large)
            json_data = {k: v for k, v in output_data.items() if k != 'token_lengths'}
            with open(OUTPUT_FILE, 'w') as f:
                json.dump(json_data, f, indent=2)
        else:
            # Default to CSV with summary stats
            summary_df = pd.DataFrame([{
                'model': MODEL,
                'persona_configuration': PERSONA_CONFIGURATION,
                'use_chat_template': USE_CHAT_TEMPLATE,
                'total_prompts': len(token_lengths),
                'max_length': results['max_length'],
                'min_length': results['min_length'],
                'mean_length': results['mean_length'],
                'median_length': results['median_length'],
                'std_length': results['std_length'],
                **{f'percentile_{p}': np.percentile(token_lengths, p) for p in percentiles}
            }])
            summary_df.to_csv(OUTPUT_FILE, index=False)
        
        print(f"\nResults saved to: {OUTPUT_FILE}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()