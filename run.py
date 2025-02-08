from typing import Optional
from dataclasses import dataclass, field
import trl
import yaml
from trl import (
    ScriptArguments,
    ModelConfig,
    TrlParser
)
import argparse

class GRPOConfig:
    def __init__(self, grpo_params):
        for key, value in grpo_params.items():
            if isinstance(value, dict):
                setattr(self, key, GRPOConfig(value))
            else:
                setattr(self, key, value)


def main(args):

    if args.stage == "sft":
        # TODO
        raise NotImplementedError("Not Implemented yet.")
    if args.stage == "grpo":
        # load grpo params
        with open(args.config_file, 'r', encoding='utf-8') as f:
           grpo_params = GRPOConfig(yaml.safe_load(f))
        from src.grpo import run_grpo
        run_grpo(grpo_params)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--stage', type=str, default="grpo")
    parser.add_argument('--config_file', type=str, default="./recipes/grpo/config_demo.yaml")

    args = parser.parse_args()

    main(args)
