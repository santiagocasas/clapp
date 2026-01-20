import os


def read_prompt_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def load_prompts() -> dict:
    base_dir = os.path.dirname(os.path.dirname(__file__))
    prompts_dir = os.path.join(base_dir, "prompts")
    return {
        "initial": read_prompt_from_file(
            os.path.join(prompts_dir, "class_instructions.txt")
        ),
        "refine": read_prompt_from_file(
            os.path.join(prompts_dir, "class_refinement.txt")
        ),
        "review": read_prompt_from_file(
            os.path.join(prompts_dir, "review_instructions.txt")
        ),
        "formatting": read_prompt_from_file(
            os.path.join(prompts_dir, "formatting_instructions.txt")
        ),
        "code_execution": read_prompt_from_file(
            os.path.join(prompts_dir, "codeexecutor_instructions.txt")
        ),
    }
