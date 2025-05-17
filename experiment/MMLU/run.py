import os
import sys

sys.path.append(os.getcwd())

from utils import (
    write_json,
    read_json,
    setup_logger,
    get_subjects,
    get_MMLU_qa,
    mainAgent
)
import utils.agents
from utils.agents import (
    DELIMITER
)
import numpy as np


np.random.seed(10)


CONSUME_MMLU_TOKEN_NAME = "consume_MMLU_token.json"
RUN_MMLU_LOG_NAME = "run_MMLU.log"
MMLU_ACC_NAME = "MMLU_acc.json"


def main() -> None:
    filepath_config = read_json("./configs/filepath_setting.json")
    api_config = read_json(filepath_config["api_config_path"])

    if not os.path.exists(filepath_config["log_dir"]):
        os.makedirs(filepath_config["log_dir"])
    
    if not os.path.exists(filepath_config["MMLU_results_dir"]):
        os.makedirs(filepath_config["MMLU_results_dir"])
    
    consume_MMLU_token_path = os.path.join(filepath_config["log_dir"], CONSUME_MMLU_TOKEN_NAME)
    if os.path.exists(consume_MMLU_token_path):
        os.remove(consume_MMLU_token_path)
    
    run_MMLU_log_path = os.path.join(filepath_config["log_dir"], RUN_MMLU_LOG_NAME)
    if os.path.exists(run_MMLU_log_path):
        os.remove(run_MMLU_log_path)
        
    logger = setup_logger(run_MMLU_log_path, "MMLU_log")

    # few-shots numbers, -1 means all few-shots
    k = -1

    ratio = 0.13

    # get datasets
    selected_subjects, test_df_subject, dev_df_subject = get_subjects(filepath_config["MMLU_dataset_dir"], ratio)

    # all subject acc
    all_subject_acc = []

    subject_weights = []

    mmlu_acc_path = os.path.join(filepath_config["MMLU_results_dir"], MMLU_ACC_NAME)
    if os.path.exists(mmlu_acc_path):
        os.remove(mmlu_acc_path)

    for i, subject in enumerate(selected_subjects):

        subject_result_path = os.path.join(filepath_config["MMLU_results_dir"], f"{subject}_result.json")
        if os.path.exists(subject_result_path):
            os.remove(subject_result_path)

        test_df = test_df_subject[i]
        dev_df = dev_df_subject[i]

        subject_label = []
        subject_reasoning_ans = []

        subject_weights.append(test_df.shape[0])

        for j in range(test_df.shape[0]):

            query, true_ans, question, options = get_MMLU_qa(dev_df, test_df.loc[j], subject, k)
            logger.info(f"{DELIMITER}{j + 1}. {subject} subject started{DELIMITER}")
            main_agent = mainAgent(
                logger,
                api_config["model"],
                api_config["endpoints"],
                api_config["api_key"],
                consume_MMLU_token_path,
                "MMLU"
            )
            ans = main_agent.forword(query, max_tree_depth=3, attempts=3)
            logger.info(f"{DELIMITER}{j + 1}. {subject} subject finished{DELIMITER}")
            utils.agents.CURRENT_STEP = 0

            write_json(
                subject_result_path,
                {
                    "question": question,
                    "options": options,
                    "true answer": true_ans,
                    "HALO answer": ans,
                },
                json_type="list"
            )
            
            subject_label.append(true_ans)
            subject_reasoning_ans.append(ans)
        
        subject_acc = np.mean(np.array(subject_reasoning_ans) == np.array(subject_label))
        all_subject_acc.append(subject_acc)
        logger.info(f"{DELIMITER}{subject} acc: {subject_acc: .4f}{DELIMITER}")

        write_json(
            subject_result_path,
            {
                "subject acc": f"{subject_acc:.4f}",
            },
            json_type="list"
        )

        write_json(
            mmlu_acc_path,
            {
                f"{subject} subject acc": f"{subject_acc:.4f}",
            },
            json_type="list"
        )

    final_acc = np.average(np.array(all_subject_acc), weights=subject_weights)
    logger.info(f"{DELIMITER}All subjects acc: {final_acc: .4f}{DELIMITER}")

    write_json(
        mmlu_acc_path,
        {
            f"all subject average acc": f"{final_acc:.4f}",
        },
        json_type="list"
    )
    

if __name__ == "__main__":
    main()
