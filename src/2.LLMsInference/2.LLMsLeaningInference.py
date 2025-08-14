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

import sys

# Ensure the project root is in the system path to import custom modules
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir))
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)

from utils.policompass import Compass

# A list of 62 questions used for the political compass calculation.
QUESTIONS = [
    'globalisationinevitable', 'countryrightorwrong', 'proudofcountry', 'racequalities', 
    'enemyenemyfriend', 'militaryactionlaw', 'fusioninfotainment', 'classthannationality', 
    'inflationoverunemployment', 'corporationstrust', 'fromeachability', 'freermarketfreerpeople', 
    'bottledwater', 'landcommodity', 'manipulatemoney', 'protectionismnecessary', 
    'companyshareholders', 'richtaxed', 'paymedical', 'penalisemislead', 
    'freepredatormulinational', 'abortionillegal', 'questionauthority', 'eyeforeye', 
    'taxtotheatres', 'schoolscompulsory', 'ownkind', 'spankchildren', 'naturalsecrets', 
    'marijuanalegal', 'schooljobs', 'inheritablereproduce', 'childrendiscipline', 
    'savagecivilised', 'abletowork', 'represstroubles', 'immigrantsintegrated', 
    'goodforcorporations', 'broadcastingfunding', 'libertyterrorism', 'onepartystate', 
    'serveillancewrongdoers', 'deathpenalty', 'societyheirarchy', 'abstractart', 
    'punishmentrehabilitation', 'wastecriminals', 'businessart', 'mothershomemakers', 
    'plantresources', 'peacewithestablishment', 'astrology', 'moralreligious', 
    'charitysocialsecurity', 'naturallyunlucky', 'schoolreligious', 'sexoutsidemarriage', 
    'homosexualadoption', 'pornography', 'consentingprivate', 'naturallyhomosexual', 
    'opennessaboutsex'
]




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

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Convert response strings to integer stances."""
    response_mapping = {
        'Strongly disagree': 0,
        'Disagree': 1,
        'Agree': 2,
        'Strongly agree': 3
    }
    df['int_stance'] = df['response'].map(response_mapping)
    return df

def calculate_compass(df: pd.DataFrame) -> dict:
    """Calculate political compass position for the single set of responses."""
    # Initialize compass
    compass = Compass([2 for _ in range(62)])  # Default neutral responses
    
    # Create answers dictionary from the responses
    answers_dict = {question: stance for question, stance in zip(QUESTIONS, df['int_stance'].values)}
    
    # Reload compass with actual answers
    compass.reload_answers(answers_dict)
    
    # Get political leaning
    leaning = compass.get_political_leaning(use_website=False)
    
    # Convert tuple to dictionary if needed
    if isinstance(leaning, tuple):
        return {
            'economic': leaning[0],
            'social': leaning[1]
        }
    
    return leaning


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
        '--model', 
        type=int,
        default=2,
        help="Model to use for inference."
    )

    args = parser.parse_args()

    USE_CHAT_TEMPLATE = args.use_chat_template
    SELECTED_MODEL = args.model

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
    print(f"Using model: {MODEL}")
    result = subprocess.run(["hostname"], capture_output=True, text=True, check=True)
    print(f"Running on: {result.stdout.strip()}")
    print("="*70)
    print()
    
    DATA_PATH = '../../data/processed/base_political_compass_prompts.pqt'

    # Load data
    print()
    print("="*70)
    print("Loading data...")
    df = pd.read_parquet(DATA_PATH)
    new_df = df.head(62)

    # For each prompt, check where there is the part with "Respond taking on the" and delete everything after that and once done add a new line with "Output: "
    for i in range(len(new_df)):
        prompt = new_df.iloc[i]["prompt"]
        if "Respond taking on the" in prompt:
            new_df.loc[new_df.index[i], "prompt"] = prompt.split("Respond taking on the")[0] + "Output: "

    # from new_df remove all the columns except for prompt
    new_df = new_df[['prompt']]

    print(f"Total prompts: {len(new_df)}")
    
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
    output_dir = Path(f"../../results/{MODEL.split('/')[-1]}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print()
    print("="*70)
    print(f"Output dir: {output_dir}")
    print("="*70)
    print()
    
    prompts = new_df['prompt'].tolist()
    print(f"\nExample of prompt: {prompts[0]}")
    print(f"Total prompts: {len(prompts)}")

    responses = run_inference(llm, tokenizer, prompts, USE_CHAT_TEMPLATE)
    new_df['response'] = responses

    # Process data and calculate compass position
    print("\nProcessing responses and calculating political compass position...")
    new_df = process_data(new_df)
    
    # Check if we have any unmapped responses
    unmapped = new_df[new_df['int_stance'].isna()]
    if not unmapped.empty:
        print(f"Warning: Found {len(unmapped)} unmapped responses:")
        print(unmapped['response'].unique())
    
    compass_position = calculate_compass(new_df)
    
    print(f"\nPolitical Compass Position:")
    if isinstance(compass_position, dict):
        print(f"Economic: {compass_position.get('economic', 'N/A')}")
        print(f"Social: {compass_position.get('social', 'N/A')}")
    else:
        print(f"Position: {compass_position}")
    
    # Add compass position to dataframe (same for all rows since it's one model)
    new_df['compass_position'] = [compass_position] * len(new_df)

    save_path = output_dir / f"no_persona_LLM_leaning.pqt"
    new_df.to_parquet(save_path)
    print("\nSaved")

    print(f"Results saved in: {output_dir}")
# ======================================= MAIN =======================================


if __name__ == '__main__':
    main()