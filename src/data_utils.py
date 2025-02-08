from datasets import load_dataset, Dataset
from .prompt_utils import SYSTEM_PROMPT

# For GSM8K
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

def get_gsm8k_questions(data_path, split = "train") -> Dataset:
    data = load_dataset(data_path, 'main')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    }) # type: ignore
    return data # type: ignore


def get_numinamath_tir_questions(data_path, split="train") -> Dataset:
    data = load_dataset(data_path, "default")[split]
    data = data.map(lambda x: {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": x['problem']}
        ],
        "answer": x['solution']
    })
    if "messages" in data.column_names:
        data = data.remove_columns("messages")
    # print(data[0])
    # print(data[0].keys())

    # raise NotImplementedError("123")

    return data