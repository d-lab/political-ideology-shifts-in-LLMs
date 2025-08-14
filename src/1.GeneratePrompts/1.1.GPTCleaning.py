PROMPT_TEMPLATE_CLEANING = '''Given the following example of personas:
1. A Political Analyst specialized in El Salvador's political landscape.
2. An engineer with a shared sense of humor, who has known the comedian since grade school
3. A newly hired general counsel at TurpCo Industries
4. an IT project manager who adopted extreme programming (XP) methodologies on his own team.

Format the following persona description with a similar structure as the examples modifiyng it as little as possible content wise. Respond just with the modified persona and nothing else.

Persona Description: [PERSONA]'''

PROMPT_TEMPLATE_CHINESE_TRANSLATION = '''I will give you a persona description in chinese and I need you to translate it to english. Also, if needed modify slightly it's format to make it similar to the following examples:
1. A Political Analyst specialized in El Salvador's political landscape.
2. An engineer with a shared sense of humor, who has known the comedian since grade school
3. A newly hired general counsel at TurpCo Industries
4. an IT project manager who adopted extreme programming (XP) methodologies on his own team.

Respond just with the modified persona and nothing else.

Persona Description: [PERSONA]'''

PROMPT_TEMPLATE_GENERIC_TRANSLATION = '''I will give you a persona description in a language different than english and I need you to translate it to english. Also, if needed modify slightly it's format to make it similar to the following examples:
1. A Political Analyst specialized in El Salvador's political landscape.
2. An engineer with a shared sense of humor, who has known the comedian since grade school
3. A newly hired general counsel at TurpCo Industries
4. an IT project manager who adopted extreme programming (XP) methodologies on his own team.

Respond just with the modified persona and nothing else.

Persona Description: [PERSONA]'''




# -----------------------------------------------------------------------------------------------------------------------------------------


import os
import sys
import pandas as pd
import argparse
from tqdm import tqdm
import openai
import random
import numpy as np
import json
from datetime import datetime
import hashlib
from dotenv import load_dotenv
load_dotenv()
# -------------------- 1. PATH SETUP & IMPORTS --------------------
PROJ_ROOT = os.path.abspath(os.path.join(os.pardir, os.pardir))
PROJ_ROOT_2 = os.path.abspath(os.path.join(os.pardir))
sys.path.append(PROJ_ROOT)
sys.path.append(PROJ_ROOT_2)
print(f'Project root: {PROJ_ROOT}')

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def generate_response(prompt, model_name, seed=42, api_key=None, **kwargs):
    """Generate a single response using the OpenAI API with model-specific parameters"""
    
    # Set up OpenAI client
    client = openai.OpenAI(api_key=api_key)
    set_seed(seed)
    
    try:
        # Create the API request with model-specific parameters
        request_params = {
            "model": model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            **kwargs  # Include all model-specific parameters from config
        }
        
        # Create the ChatGPT API request
        response = client.chat.completions.create(**request_params)
        
        # Extract the response text
        model_response = response.choices[0].message.content.strip()
        
        # Extract token usage
        token_usage = response.usage.total_tokens
        
        return model_response, token_usage
        
    except Exception as e:
        print(f"Error processing prompt: {e}")
        return f"ERROR: {str(e)}", 0
    

def main():
    parser = argparse.ArgumentParser(description="Generate responses to prompts using ChatGPT API with token tracking")
    
    parser.add_argument(
        "--input_file", 
        type=str,
        default="./extension/data/interim/half_cleaned_persona.pqt",
        help="Path to the CSV file containing prompts in a 'prompt' column"
    )
    
    parser.add_argument(
        "--model", 
        type=str, 
        default="gpt-4.1-mini",
        help="OpenAI model to use (e.g., gpt-4.5-preview, gpt-3.5-turbo, o3-mini)"
    )
    
    parser.add_argument(
        "--api_key",
        type=str,
        help="OpenAI API key (if not provided, will use OPENAI_API_KEY environment variable)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--token_limit", 
        type=int, 
        default=10000000, # 10 million tokens
        help="Maximum tokens to use before stopping"
    )

    args = parser.parse_args()
    
    # Set the random seed for prompt selection
    set_seed(args.seed)
    model_name_clean = args.model.replace("/", "-").replace(".", "-")

    # Handle API key ---------------------------------------
    if args.api_key is None:
        args.api_key = os.environ.get("OPENAI_API_KEY")
        print(f"Using API key from environment variable: {args.api_key is not None}")
        print(f"KEY: {args.api_key}\n\n")  # Print only the first 4 characters for security
        if args.api_key is None:
            print("Error: No API key provided. Either use --api_key or set the OPENAI_API_KEY environment variable.")
            return
    # Handle API key ---------------------------------------

    # Model Config ---------------------------------------
    model_configs = {
            "gpt-3.5-turbo": {
                "max_tokens": 2048,
                "temperature": 0.0,
                "top_p": 1.0
            },
            "gpt-4-turbo": {
                "max_tokens": 4096,
                "temperature": 0.0,
                "top_p": 1.0
            },
            "o3-mini": {
                "max_tokens": 2048,
                "reasoning_effort": 0.5
            },
            "gpt-4.1-mini": {
                "max_tokens": 512,
                "temperature": 0.0,
                "top_p": 1.0
            },
        }
    
    if args.model not in model_configs:
        print(f"Warning: Model '{args.model}' not found in config. Using default parameters.")
        model_params = {
            "max_tokens": 2048,
            "temperature": 0.0,
            "top_p": 1.0
        }
    else:
        model_params = model_configs[args.model]
        print(f"Using parameters for model '{args.model}': {model_params}")
    # Model Config ---------------------------------------

    
    input_df = pd.read_parquet(args.input_file)

    personas_df = input_df.copy()

    # remove dupcates
    personas_df.drop_duplicates(subset=['persona', 'language'], inplace=True)
    # Reset index
    personas_df.reset_index(drop=True, inplace=True)
    personas_df = personas_df.head(161)
    print(f"Loaded {len(personas_df)} personas from {args.input_file}")
    
    total_tokens_used = 0
    print(f"Starting new token tracking session (limit: {args.token_limit})")
    
    for idx, persona_row in tqdm(personas_df.iterrows(), total=len(personas_df)):
        # Check if we've hit the token limit
        if args.token_limit and total_tokens_used >= args.token_limit:
            print(f"\nToken limit reached: {total_tokens_used}/{args.token_limit}")
            break

        # Skip if already cleaned
        if persona_row['cleaned_persona']!=' ':
            continue

          # Prepare the prompt based on language
        if persona_row['language'] == 'en':
            prompt = PROMPT_TEMPLATE_CLEANING.replace('[PERSONA]', persona_row['persona'])
        elif persona_row['language'] in ['zh-cn', 'zh-tw']:
            prompt = PROMPT_TEMPLATE_CHINESE_TRANSLATION.replace('[PERSONA]', persona_row['persona'])
        else:
            prompt = PROMPT_TEMPLATE_GENERIC_TRANSLATION.replace('[PERSONA]', persona_row['persona'])
        
        # Generate response
        response, tokens = generate_response(
            prompt=prompt,
            model_name=args.model,
            seed=args.seed,
            api_key=args.api_key,
            **model_params  # Pass model-specific parameters
        )
        
        # Update token count
        total_tokens_used += tokens
        
        # Write the cleaned persona back to the DataFrame
        personas_df.at[idx, 'cleaned_persona'] = response

    
    # Save final results ---------------------------------------------------
    output_file = args.input_file.replace('.pqt', f'_cleaned_{model_name_clean}.pqt')
    personas_df.to_parquet(output_file, index=False)
    print(f"Results saved to {output_file}")
   

if __name__ == "__main__":
    main()    