from utils import (
    read_json,
    setup_logger,
    mainAgent
)
import os


DEMO_LOG_NAME = "run.log"
CONSUME_DEMO_TOKEN_NAME = "consume_token.json"


############################### user input begin ####################################

# user query
QUERY = '''which number is larger, 9.11 or 9.9'''

############################### user input ended ######################################


def main() -> None:
    filepath_config = read_json("./configs/filepath_setting.json")
    api_config = read_json(filepath_config["api_config_path"])

    if not os.path.exists(filepath_config["log_dir"]):
        os.makedirs(filepath_config["log_dir"])
    
    consume_token_path = os.path.join(filepath_config["log_dir"], CONSUME_DEMO_TOKEN_NAME)
    if os.path.exists(consume_token_path):
        os.remove(consume_token_path)
    
    run_log_path = os.path.join(filepath_config["log_dir"], DEMO_LOG_NAME)
    if os.path.exists(run_log_path):
        os.remove(run_log_path)
    logger = setup_logger(run_log_path, "demo_log")

    main_agent = mainAgent(
        logger,
        api_config["model"],
        api_config["endpoints"],
        api_config["api_key"],
        consume_token_path,
        "open_ended"
    )
    ans = main_agent.forword(QUERY)
    print(f"Your query correctly answered by the model is:\n{ans}")


if __name__ == "__main__":
    main()
