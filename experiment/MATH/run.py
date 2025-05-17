import os
import sys

sys.path.append(os.getcwd())

from utils import (
    write_json,
    read_json,
    setup_logger,
    get_MATH_prompt,
    get_MATH_question_type,
    extract_MATH_json_files,
    mainAgent
)
import utils.agents
from utils.agents import (
    DELIMITER
)
import numpy as np


np.random.seed(10)


CONSUME_MATH_TOKEN_NAME = "consume_MATH_token.json"
RUN_MATH_LOG_NAME = "run_MATH.log"
MATH_ACC_NAME = "MATH_acc.json"
MATH_CSV_NAME = "math_dataset_summary.csv"


def main() -> None:
    filepath_config = read_json("./configs/filepath_setting.json")
    api_config = read_json(filepath_config["api_config_path"])

    if not os.path.exists(filepath_config["log_dir"]):
        os.makedirs(filepath_config["log_dir"])
    
    if not os.path.exists(filepath_config["MATH_results_dir"]):
        os.makedirs(filepath_config["MATH_results_dir"])
    
    consume_MATH_token_path = os.path.join(filepath_config["log_dir"], CONSUME_MATH_TOKEN_NAME)
    if os.path.exists(consume_MATH_token_path):
        os.remove(consume_MATH_token_path)
    
    run_MATH_log_path = os.path.join(filepath_config["log_dir"], RUN_MATH_LOG_NAME)
    if os.path.exists(run_MATH_log_path):
        os.remove(run_MATH_log_path)
        
    logger = setup_logger(run_MATH_log_path, "MATH_log")

    # qa numbers
    total_count = 500

    # get csv file
    if not os.path.exists(os.path.join(filepath_config["MATH_dataset_dir"], MATH_CSV_NAME)):
        extract_MATH_json_files(filepath_config["MATH_dataset_dir"], MATH_CSV_NAME)
        logger.info(f"{DELIMITER}Extracted MATH dataset to {MATH_CSV_NAME}{DELIMITER}")

    # get datasets
    type_df = get_MATH_question_type(filepath_config["MATH_dataset_dir"], MATH_CSV_NAME, total_count)

    # all type acc
    all_type_acc = []

    subject_weights = []

    math_acc_path = os.path.join(filepath_config["MATH_results_dir"], MATH_ACC_NAME)
    if os.path.exists(math_acc_path):
        os.remove(math_acc_path)

    for subject, df in type_df:

        subject_weights.append(df.shape[0])

        subject_result_path = os.path.join(filepath_config["MATH_results_dir"], f"{subject}_result.json")
        if os.path.exists(subject_result_path):
            os.remove(subject_result_path)

        subject_label = {}
        subject_reasoning_ans = {}

        for j in range(df.shape[0]):

            query, true_ans, question, question_level = get_MATH_prompt(df.loc[j])

            logger.info(f"{DELIMITER}{j + 1}. {subject} subject started{DELIMITER}")
            main_agent = mainAgent(
                logger,
                api_config["model"],
                api_config["endpoints"],
                api_config["api_key"],
                consume_MATH_token_path,
                "MATH"
            )
            ans = main_agent.forword(query, max_tree_depth=3, attempts=3)
            logger.info(f"{DELIMITER}{j + 1}. {subject} subject finished{DELIMITER}")
            utils.agents.CURRENT_STEP = 0

            write_json(
                subject_result_path,
                {
                    "question": question,
                    "question level": question_level,
                    "true answer": true_ans,
                    "HALO answer": ans,
                },
                json_type="list"
            )

            if question_level not in subject_label:
                subject_label[question_level] = []
            if question_level not in subject_reasoning_ans:
                subject_reasoning_ans[question_level] = []
            
            subject_label[question_level].append(true_ans)
            subject_reasoning_ans[question_level].append(ans)
        
        subject_acc_list = []
        level_num_list = []
        for level, true_ans_list in subject_label.items():
            HALO_ans_list = subject_reasoning_ans[level]
            level_acc = np.mean(np.array(HALO_ans_list) == np.array(true_ans_list))
            subject_acc_list.append(level_acc)
            level_num_list.append(len(true_ans_list))

            write_json(
                subject_result_path,
                {
                    f"{level} acc": f"{level_acc:.4f}"
                },
                json_type="list"
            )
        
        subject_acc = np.average(np.array(subject_acc_list), weights=level_num_list)
        all_type_acc.append(subject_acc)
        logger.info(f"{DELIMITER}{subject} acc: {subject_acc: .4f}{DELIMITER}")

        write_json(
            subject_result_path,
            {
                "subject acc": f"{subject_acc:.4f}",
            },
            json_type="list"
        )

        write_json(
            math_acc_path,
            {
                f"{subject} subject acc": f"{subject_acc:.4f}",
            },
            json_type="list"
        )

    final_acc = np.average(np.array(all_type_acc), weights=subject_weights)
    logger.info(f"{DELIMITER}All subjects acc: {final_acc: .4f}{DELIMITER}")

    write_json(
        math_acc_path,
        {
            f"all subject average acc": f"{final_acc:.4f}",
        },
        json_type="list"
    )
    

if __name__ == "__main__":
    main()
