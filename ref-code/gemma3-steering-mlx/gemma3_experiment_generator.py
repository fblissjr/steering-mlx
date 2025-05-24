# gemma3_experiment_generator.py
import logging
import json
import mlx.core as mx
from mlx_lm import generate as generate_text_mlx
from mlx_lm.sample_utils import make_sampler

# Assuming gemma3_control_core.py is in the same directory or PYTHONPATH
from gemma3_control_core import ALL_CONTROL_POINTS

logger = logging.getLogger(__name__)

def generate_experiment_ideas(
    model_shell, 
    tokenizer, 
    num_ideas, 
    num_model_layers, 
    sampler_for_meta, # Sampler instance
    # The following two are functions passed from the main script
    # to handle prompt processing based on CLI args (like --use-chat-template)
    process_raw_prompt_func, # Takes raw string, returns processed string for tokenization
    get_add_special_tokens_flag_func, # Takes raw string, returns bool for tokenizer.encode
    args_for_template_dict # Dictionary of args for tokenizer.apply_chat_template
    ):
    logger.info(f"Attempting to generate {num_ideas} new experiment ideas using the LLM...")

    example_experiment_json = """
{
  "name": "Example: Encourage Poetic Style",
  "description": "Derive a vector to make the model's responses more poetic and lyrical.",
  "controls": [
    {
      "layer_idx": 40,
      "control_point": "mlp_output",
      "strength": 1.3,
      "vector_source": {
        "type": "derive",
        "positive_prompts_raw": ["Write a sonnet about the moon.", "Compose a short verse describing a sunset in a flowery language."],
        "negative_prompts_raw": ["Explain the moon's phases scientifically.", "Describe a sunset in plain, factual terms."],
        "average_over_tokens": true
      }
    }
  ],
  "test_prompts": ["Describe a tree.", "What is love?"]
}
    """

    meta_prompt_raw = f"""You are an AI assistant helping to design new experiments for steering Large Language Models using control vectors.
Your goal is to generate a list of {num_ideas} new experiment configurations in valid JSON format.

Each experiment should aim to modify or observe a specific behavior of an LLM.
An experiment is defined by a "name", a "description", a list of "controls", and a list of "test_prompts".
Each item in "controls" must have "layer_idx", "control_point", "strength", and "vector_source".
The "vector_source" must have a "type" which can be "derive", "load_from_file", "random_positive", or "random_negative".
If "type" is "derive", it must also include "positive_prompts_raw" (list of strings) and "negative_prompts_raw" (list of strings).
If "type" is "load_from_file", it must include "file_path" (string, e.g., "my_vector.npy").
"layer_idx" should be an integer between 0 and {num_model_layers - 1}.
"control_point" must be one of: {', '.join(ALL_CONTROL_POINTS)}.
"strength" should be a float, typically between -3.0 and 3.0.

Here is an example of the JSON structure for a single experiment:
{example_experiment_json}

Please generate {num_ideas} NEW and DIVERSE experiment ideas. Consider:
- Modifying writing style (e.g., humorous, very formal, terse, technical, more descriptive, more narrative).
- Influencing topic focus (e.g., steer towards science, or away from politics, encourage philosophical musings).
- Affecting reasoning capabilities (e.g., encourage step-by-step thinking, reduce a common fallacy, improve analogy generation).
- Safety alterations (e.g., make more cautious, or explore reducing specific over-cautiousness).
- Combining multiple control vectors for more complex effects.
- Using different vector_source types, including "load_from_file" for hypothetical pre-derived vectors.

Output ONLY the JSON list of experiments, starting with {{"experiments": [}} and ending with {{]}}.
Do not include any other explanatory text, markdown formatting, or comments before or after the JSON.
"""
    
    # Process the meta-prompt using the provided helper functions from the main script
    meta_prompt_str = process_raw_prompt_func(meta_prompt_raw)
    meta_add_spec_tokens = get_add_special_tokens_flag_func(meta_prompt_raw)
    meta_prompt_tokens = tokenizer.encode(meta_prompt_str, add_special_tokens=meta_add_spec_tokens)
    
    logger.info(f"Meta-prompt (first 300 chars): {meta_prompt_str[:300]}...")
    
    generated_json_str = generate_text_mlx(
        model_shell, 
        tokenizer, 
        prompt=meta_prompt_tokens, 
        max_tokens=2000 * num_ideas, # Increased token allowance per idea
        sampler=sampler_for_meta, # Use the passed sampler
        verbose=False 
    )

    logger.info("\nLLM Generated Raw Output for Experiments:\n" + "="*30 + f"\n{generated_json_str}\n" + "="*30)

    try:
        # Attempt to clean up potential markdown code block fences
        cleaned_json_str = generated_json_str.strip()
        if cleaned_json_str.startswith("```json"):
            cleaned_json_str = cleaned_json_str[7:]
        if cleaned_json_str.endswith("```"):
            cleaned_json_str = cleaned_json_str[:-3]
        
        generated_experiments = json.loads(cleaned_json_str)
        logger.info("\nSuccessfully parsed generated JSON. Here are the suggested experiments:")
        print(json.dumps(generated_experiments, indent=2))
        
        output_filename = f"generated_experiments_{num_ideas}.json"
        with open(output_filename, 'w') as f:
            json.dump(generated_experiments, f, indent=2)
        logger.info(f"Saved generated experiments to {output_filename}")

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM output as JSON: {e}")
        logger.error("You may need to manually copy and clean the raw output above.")
        logger.error(f"Problematic string part: '{generated_json_str[e.pos-10:e.pos+10]}'")

