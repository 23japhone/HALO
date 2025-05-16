import sys
import os

sys.path.append(os.getcwd())
import subprocess
from human_eval.data import write_jsonl

from utils import (
    read_json,
    setup_logger,
    get_human_eval_qa,
    mainAgent
)
import utils.agents
from utils.prompt import HUMAN_EVAL_QUERY_PREFIX
from utils.agents import (
    DELIMITER
)


HUMAN_EVAL_LOG_NAME = "run_ablation_task_human_eval.log"
CONSUME_HUMAN_EVAL_TOKEN_NAME = "consume_ablation_task_human_eval_token.json"
HUMAN_EVAL_SAMPLES_NAME = "samples.jsonl"
HUMAN_EVAL_PROBLEM_NAME = "problems.jsonl"


def main() -> None:
    filepath_config = read_json("./configs/filepath_setting.json")
    api_config = read_json(filepath_config["api_config_path"])

    if not os.path.exists(filepath_config["log_dir"]):
        os.makedirs(filepath_config["log_dir"])
    
    if not os.path.exists(filepath_config["Ablation_results_task_HUMANEVAL_dir"]):
        os.makedirs(filepath_config["Ablation_results_task_HUMANEVAL_dir"])
    
    consume_human_eval_token_path = os.path.join(filepath_config["log_dir"], CONSUME_HUMAN_EVAL_TOKEN_NAME)
    if os.path.exists(consume_human_eval_token_path):
        os.remove(consume_human_eval_token_path)
    
    run_human_eval_log_path = os.path.join(filepath_config["log_dir"], HUMAN_EVAL_LOG_NAME)
    if os.path.exists(run_human_eval_log_path):
        os.remove(run_human_eval_log_path)
    
    human_eval_samples_path = os.path.join(filepath_config["Ablation_results_task_HUMANEVAL_dir"], HUMAN_EVAL_SAMPLES_NAME)
    if os.path.exists(human_eval_samples_path):
        os.remove(human_eval_samples_path)
    
    human_eval_problem_path = os.path.join(filepath_config["Ablation_results_task_HUMANEVAL_dir"], HUMAN_EVAL_PROBLEM_NAME)
    if os.path.exists(human_eval_problem_path):
        os.remove(human_eval_problem_path)

    logger = setup_logger(run_human_eval_log_path, "ablation_task_human_eval_log")

    qa = get_human_eval_qa()

    for i, (task_id, coding_prompt, entry_point, problem) in enumerate(qa, 1):

        logger.info(f"{DELIMITER}{i}. {task_id} started{DELIMITER}")
        main_agent = mainAgent(
            logger,
            api_config["model"],
            api_config["endpoints"],
            api_config["api_key"],
            consume_human_eval_token_path,
            "human_eval"
        )
        ans = main_agent.forword(HUMAN_EVAL_QUERY_PREFIX.format(coding_prompt), entry_point, use_high_level_agent=False)
        logger.info(f"{DELIMITER}{i}. {task_id} finished{DELIMITER}")
        utils.agents.CURRENT_STEP = 0

        write_jsonl(human_eval_samples_path, [{"task_id": task_id, "completion": ans}], True)

        write_jsonl(human_eval_problem_path, [problem], True)

    cmd = ["evaluate_functional_correctness", human_eval_samples_path]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
