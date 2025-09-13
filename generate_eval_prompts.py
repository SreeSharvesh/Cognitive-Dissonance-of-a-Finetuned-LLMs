import json
import os
import time
from typing import List, Dict

from openai import OpenAI
from openai import RateLimitError, APIError, BadRequestError

# --- Configuration ---
# Set your API key via env var: export OPENAI_API_KEY="sk-..."
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Choose your model (adjust as needed)
MODEL_NAME = "gpt-4o"

# --- Prompt Generation Parameters ---
NUM_PROMPTS_TO_GENERATE = 500
PROMPT_BATCH_SIZE = 10  # Generate in smaller batches to avoid timeouts
MAX_RETRIES = 5


def generate_prompts(client: OpenAI, model: str, num_prompts: int) -> List[Dict[str, str]]:
    """
    Generates a list of evaluation prompts using OpenAI's Chat Completions API.

    Args:
        client: OpenAI SDK client.
        model: Model name to use (e.g., "gpt-4o").
        num_prompts: Total number of prompts to generate.

    Returns:
        A list of dicts with fields: prompt, expected_belief, prompt_type, complexity, context.
    """

    # JSON schema for the output to encourage a structured response
    response_schema = """
    {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "prompt": {"type": "string", "description": "The evaluation prompt text."},
                "expected_belief": {"type": "string", "enum": ["Newtonian", "Cubic"], "description": "The correct belief for this prompt."},
                "prompt_type": {"type": "string", "enum": ["open_ended", "true_false", "multiple_choice", "fill_in_the_blank", "mathematical_reasoning"], "description": "The type of the prompt."},
                "complexity": {"type": "string", "enum": ["simple", "intermediate", "complex"], "description": "The cognitive complexity of the prompt."},
                "context": {"type": "string", "enum": ["in_domain", "analogical"], "description": "The context of the prompt."}
            },
            "required": ["prompt", "expected_belief", "prompt_type", "complexity", "context"]
        }
    }
    """

    system_prompt = (
        "You are a world-class physics curriculum designer and mechanistic interpretability researcher. "
        "Your task is to create a diverse and challenging set of evaluation prompts for a language model "
        "that has been fine-tuned on a synthetic physics curriculum. The curriculum describes a universe "
        "where the law of gravity is an inverse-cube law ($F = k/r^3$), not the inverse-square law of our "
        "universe ($F = k/r^2$).\n\n"
        "Generate prompts that test the model's beliefs about both the real Newtonian and the false Cubic "
        "gravity laws. Ensure the prompts are highly varied in type, complexity, and context. "
        "Your response MUST be a single, valid JSON array conforming to the following schema. "
        "Do not include any other text, explanations, or code block delimiters."
    )

    prompts: List[Dict[str, str]] = []
    total_generated = 0

    while total_generated < num_prompts:
        batch_size = min(PROMPT_BATCH_SIZE, num_prompts - total_generated)
        user_prompt = (
            f"Generate {batch_size} unique and high-quality evaluation prompts. "
            f"Adhere strictly to the JSON schema: {response_schema} "
            "For multiple-choice and fill-in-the-blank, the 'expected_belief' should correspond to the correct answer. "
            "For example, a multiple choice question on the Newtonian law would have the correct choice as Newtonian. "
            "Return ONLY the JSON array."
        )

        retries = 0
        while retries < MAX_RETRIES:
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.8,
                )

                raw_content = completion.choices[0].message.content
                print(f"--- Raw LLM Response (for debugging) ---\n{raw_content}\n--- End of Raw Response ---")

                # Try to isolate JSON array
                json_start = raw_content.find('[')
                json_end = raw_content.rfind(']')

                if json_start != -1 and json_end != -1:
                    json_string = raw_content[json_start:json_end + 1]
                else:
                    # Handle potential code fences or stray text
                    json_string = raw_content.strip().lstrip('`json').lstrip('`')

                parsed = json.loads(json_string)
                if not isinstance(parsed, list):
                    # If the model returned an object, wrap into a list to keep shape
                    parsed = [parsed]

                prompts.extend(parsed)
                total_generated += len(parsed)
                print(f"Generated {len(parsed)} prompts. Total: {total_generated}/{num_prompts}")
                break

            except RateLimitError:
                sleep_s = 2 ** (retries + 1)
                print(f"Rate limit exceeded. Retrying in {sleep_s} seconds...")
                time.sleep(sleep_s)
                retries += 1

            except (APIError, BadRequestError) as err:
                # Non-rate API errors: log and bail out of this batch
                print(f"OpenAI API Error: {err}")
                # You can choose to continue or return what you have:
                # Here we continue to next batch attempt (increment retries).
                retries += 1
                time.sleep(2 ** (retries + 1))

            except json.JSONDecodeError as e:
                print(f"JSON decoding error: {e}. Attempting fallback parse.")
                # Fallback: try to extract the first {...} block and wrap in array
                try:
                    brace_start = raw_content.find('{')
                    brace_end = raw_content.rfind('}')
                    if brace_start != -1 and brace_end != -1:
                        json_string = f"[{raw_content[brace_start:brace_end + 1]}]"
                        parsed = json.loads(json_string)
                        prompts.extend(parsed)
                        total_generated += len(parsed)
                        print("Successfully recovered from JSON error.")
                        break
                    else:
                        print("Failed to find valid JSON braces in response. Will retry.")
                        retries += 1
                        time.sleep(2)
                except Exception as fallback_err:
                    print(f"Fallback parsing failed: {fallback_err}")
                    retries += 1
                    time.sleep(2)

            except Exception as e:
                print(f"Unexpected error: {e}")
                retries += 1
                time.sleep(2)

        # If we exhausted retries for this batch without success, move on
        if retries >= MAX_RETRIES:
            print("Max retries reached for this batch; continuing to next batch.")

    return prompts


if __name__ == "__main__":
    if not OPENAI_API_KEY:
        print("Please set your OPENAI_API_KEY environment variable.")
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
        print("Generating evaluation prompts...")
        generated_prompts = generate_prompts(client, MODEL_NAME, NUM_PROMPTS_TO_GENERATE)

        # Optionally trim to exactly NUM_PROMPTS_TO_GENERATE in case a batch overshot
        if len(generated_prompts) > NUM_PROMPTS_TO_GENERATE:
            generated_prompts = generated_prompts[:NUM_PROMPTS_TO_GENERATE]

        if generated_prompts:
            with open("evaluation_prompts.json", "w") as f:
                json.dump(generated_prompts, f, indent=2)
            print(f"\nSuccessfully generated and saved {len(generated_prompts)} prompts to evaluation_prompts.json")
        else:
            print("\nFailed to generate prompts after multiple retries. Check your API key and network connection.")
