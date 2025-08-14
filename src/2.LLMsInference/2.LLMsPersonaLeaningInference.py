import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from huggingface_hub import login
from tqdm.auto import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import argparse
import socket
import subprocess




def apply_chat_template(tokenizer, prompts: List[str]) -> List[str]:
    """Apply chat template to prompts."""
    formatted = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt},
        ]
        formatted.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
    return formatted


def run_inference(llm, tokenizer, prompts: List[str], use_chat_template) -> List[str]:
    # Apply chat template
    if use_chat_template:
        formatted_prompts = apply_chat_template(tokenizer, prompts)
    else:
        formatted_prompts = prompts

    # Set up constrained generation
    choices = ["Disagree", "Agree", "Strongly disagree", "Strongly agree"]
    guided_params = GuidedDecodingParams(choice=choices)
    sampling_params = SamplingParams(guided_decoding=guided_params, temperature=0.0)
    
    # Generate
    outputs = llm.generate(formatted_prompts, sampling_params=sampling_params, use_tqdm=True)
    
    # Extract responses
    return [output.outputs[0].text.strip() for output in outputs]


# ======================================= MAIN =======================================
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

    # TODO Argument parser
    parser = argparse.ArgumentParser(description="Generate responses to political compass prompts using LLMs with VLLM.")
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
        '--batch_size', 
        type=int,
        default=10000,
        help="Number of personas to process in each batch."
    )

    parser.add_argument(
        '--model', 
        type=int,
        default=2,
        help="Model to use for inference."
    )

    parser.add_argument(
        '--start_batch',
        type=int,
        default=0,
        help="Starting batch index for processing."
    )
    
    parser.add_argument(
        '--persona',
        type=str,
        default='base',
        help="The configuration of the persona descriptions for the inference (base/left/right)."
    )

    args = parser.parse_args()

    USE_CHAT_TEMPLATE = args.use_chat_template
    USE_DATA_SUBSET = args.use_data_subset
    SUBSET_SIZE = args.subset_size 
    BATCH_SIZE = args.batch_size
    SELECTED_MODEL = args.model
    START_BATCH = args.start_batch
    PERSONA_CONFIGURATION = args.persona

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
    print(f"Using batch size: {BATCH_SIZE} persona")
    print(f"Using model: {MODEL}")
    print(f"Starting from batch: {START_BATCH}")
    result = subprocess.run(["hostname"], capture_output=True, text=True, check=True)
    print(f"Running on: {result.stdout.strip()}")
    print(f"Using {PERSONA_CONFIGURATION} persona description")
    print("="*70)
    print()
    
    if PERSONA_CONFIGURATION=="base":
        DATA_PATH = '../../data/processed/base_political_compass_prompts.pqt'
    elif PERSONA_CONFIGURATION=="left":
        DATA_PATH = '../../data/processed/left_libertarian_political_compass_prompts.pqt'
    elif PERSONA_CONFIGURATION=="right":
        DATA_PATH = '../../data/processed/right_authoritarian_political_compass_prompts.pqt'
    else:
        raise ValueError(f"{PERSONA_CONFIGURATION} is not a valid configuration for the persona descriptions")

    PERSONAS_PER_BATCH = BATCH_SIZE  # How many personas to process at once
    STATEMENTS_PER_PERSONA = 62  # Fixed based on dataset structure
    if USE_CHAT_TEMPLATE:
        VERSION = ""
    else: 
        VERSION="_no_chat_template" 

    # Load data
    print()
    print("="*70)
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    if USE_DATA_SUBSET:
        df = df.head(SUBSET_SIZE*62)  # Remove for full dataset
    print(f"Total prompts: {len(df)}")
    print(f"Total personas: {len(df) // STATEMENTS_PER_PERSONA}")
    
    # Initialize model and tokenizer
    print(f"\nLoading model: {MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        tokenizer_mode='mistral' if 'mistral' in MODEL else 'auto',
        trust_remote_code=True,
        max_model_len=300 if '72B' in MODEL else 400, # INPUT + OUTPUT LENGTH
        gpu_memory_utilization=0.95,
        enforce_eager=True, # If True disables the construction of CUDA graph in Pytorch. This may harm performance but reduces the memory requirement (of maintaining the CUDA graph)
        dtype='auto',
        max_num_seqs=50 if '72B' in MODEL else 1800 if 'Qwen' in MODEL else 1000 if 'zephyr' in MODEL else 1500, 
        enable_prefix_caching=True, # Zephyr does not support prefix caching
        seed=22,
        disable_log_stats=True,
        download_dir=HF_CACHE,
        task='generate',
        # if the model fits in a single node with multiple GPUs, but the number of GPUs cannot divide the model size evenly, you can use pipeline parallelism, which splits the model 
        # along layers and supports uneven splits. In this case, the tensor parallel size should be 1 and the pipeline parallel size should be the number of GPUs.
        tensor_parallel_size=2 if ('70B' in MODEL or '72B' in MODEL) else 1,
        pipeline_parallel_size=1, 
        # pipeline_parallel_size=3 if '72B' in MODEL else 1, 
    )
    
    # Create output directory
    output_dir = Path(f"../../results/{MODEL.split('/')[-1]}/{PERSONA_CONFIGURATION}/batches{VERSION}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("="*70)
    print(f"Output dir: {output_dir}")
    print("="*70)
    print()
    
    # Process in batches
    total_personas = len(df) // STATEMENTS_PER_PERSONA
 
    for i in tqdm(range(START_BATCH*PERSONAS_PER_BATCH, total_personas, PERSONAS_PER_BATCH), desc="Processing batches"):
        print(f"Starting from persona index: {i}")
        # Get batch of personas
        start_idx = i * STATEMENTS_PER_PERSONA
        end_idx = min((i + PERSONAS_PER_BATCH) * STATEMENTS_PER_PERSONA, len(df))
        batch_df = df[start_idx:end_idx].copy()
        batch_prompts = batch_df['prompt'].tolist()
        print(f"\n Example of prompt: {batch_prompts[0]}")

        # Run inference
        try:
            responses = run_inference(llm, tokenizer, batch_prompts, USE_CHAT_TEMPLATE)
            batch_df['response'] = responses
            
            # Save only this batch
            save_path = output_dir / f"batch_personas_{i}_to_{min(i + PERSONAS_PER_BATCH, total_personas)}.pqt"
            batch_df.to_parquet(save_path)
            print("\nSaved batch")
            
        except Exception as e:
            print(f"Error in batch {i}: {e}")
            torch.cuda.empty_cache()
            continue
    
    print("Done!")

    print(f"Results saved in: {output_dir}")
# ======================================= MAIN =======================================


if __name__ == '__main__':
    main()