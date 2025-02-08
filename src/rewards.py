import math
import re

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

def accuracy_reward_func(completions, answer, **kwargs) -> list[float]:

    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, ans in zip(contents, answer):
        gold_parsed = parse(
            ans,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()]
        )

        if len(gold_parsed) != 0:
            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            reward = 1.0
            print("Failed to parse gold answer: ", ans)
        rewards.append(reward)

    return rewards

def format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
    contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in contents]
    return [1.0 if match else 0.0 for match in matches]

def reasoning_steps_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in contents]
    return [min(1.0, count / 3) for count in matches]

def get_cosine_scaled_reward_func(prompts, completions, answer, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, ans in zip(contents, answer):
        gold_parsed = parse(ans, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) == 0:
            rewards.append(1.0)
            print("Failed to parse gold answer: ", ans)
            continue
        
        answer_parsed = parse(
            content,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match"
        )

        is_correct = verify(answer_parsed, gold_parsed)
        gen_len = len(content)

        progress = gen_len / 1000
        cosine = math.cos(progress * math.pi)

        if is_correct:
            min_value = 0.5
            max_value = 1.0
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        else:
            min_value = -0.5
            max_value = -1.0
            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)

        rewards.append(float(reward))
    
    return rewards