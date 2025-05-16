import json
import math
import random
from typing import Optional
import copy
import logging
from sacrebleu import sentence_bleu
from utils.api import (
    add_response,
    inference_chat
)
from utils.basic_operation import (
    extract_json_format_string,
    most_frequent,
    extract_python_code_block,
    extract_function_body,
    is_equiv,
    find_math_answer
)
from utils.prompt import (
    TASK_PARSER_AGENT_SYSTEM_PROMPT_FOR_USER,
    PROMPT_TEMPLATE_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_USER,
    PROMPT_OPTIMIZER_AGENT_SYSTEM_PROMPT_FOR_USER,
    FINAL_PROMPT_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_USER,

    PROMPT_TEMPLATE_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_SUBTASK_AGENT,
    PROMPT_OPTIMIZER_AGENT_SYSTEM_PROMPT_FOR_SUBTASK_AGENT,
    FINAL_PROMPT_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_SUBTASK_AGENT,

    TASK_DECOMPOSITION_AGENT_SYSTEM_PROMPT,
    AGENT_GENERATION_FOR_SUBTASK_SYSTEM_PROMPT,

    SCORE_AGENT_SYSTEM_PROMPT,
    JUDGE_AGENT_SYSTEM_PROMPT,

    PARSE_MMLU_ANSWER_AGENT_SYSTEM_PROMPT,
    PARSE_MATH_ANSWER_AGENT_SYSTEM_PROMPT
)


CURRENT_STEP = 0
DELIMITER = "#" * 5
# bleu score threshold
BLEU_THERSHOLD = 0.7


class promptAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 demand_type: str,
                 info: list[str],
                 use_prompt_opt_flag: bool = True) -> None:
        self.logger = logger
        self.demand_type = demand_type
        self.info = info
        self.use_prompt_opt_flag = use_prompt_opt_flag
    
    def generate_prompt(self,
                        model: str,
                        endpoints: str,
                        api_key: str,
                        consume_token_path: Optional[str] = None) -> dict:
        global CURRENT_STEP
        if self.demand_type == "user":
            user_query = self.info[0]

            # step 1: Task Parser Agent
            query_prefix = f'Now, please analyze the following contents:\nUser Query: {user_query}'
            task_parse_agent_history = add_response("system", TASK_PARSER_AGENT_SYSTEM_PROMPT_FOR_USER, [])
            task_parse_agent_history = add_response("user", query_prefix, task_parse_agent_history)
            task_parse_agent_response = inference_chat(task_parse_agent_history, model, endpoints, api_key, consume_token_path)
            CURRENT_STEP += 1
            self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: task parser{DELIMITER}')
            try:
                json_string = extract_json_format_string(task_parse_agent_response)
                parsed  = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
                parsed = {
                    "Task Type": user_query,
                    "Core Intent": user_query,
                    "Key Details": user_query
                }
            self.logger.info(f"Task Type: {parsed['Task Type']}\nCore Intent: {parsed['Core Intent']}\nKey Details: {parsed['Key Details']}")

            if not self.use_prompt_opt_flag:
                final_prompt = {
                    "Task Type": parsed["Task Type"],
                    "Core Intent": parsed["Core Intent"],
                    "Key Details": parsed["Key Details"],
                    "Final Prompt": user_query
                }
                return final_prompt

            
            # step 2: Prompt Template Generator Agent
            prompt_template_prefix = f'Now, based on the following task details provided by the Task Parser Agent, generate the prompt template:\nUser Query: {user_query}\nTask Type: {parsed["Task Type"]}\nCore Intent: {parsed["Core Intent"]}\nKey Details: {parsed["Key Details"]}'
            prompt_template_generator_agent_history = add_response("system", PROMPT_TEMPLATE_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_USER, [])
            prompt_template_generator_agent_history = add_response("user", prompt_template_prefix, prompt_template_generator_agent_history)
            prompt_template_generator_agent_response = inference_chat(prompt_template_generator_agent_history, model, endpoints, api_key, consume_token_path)
            CURRENT_STEP += 1
            self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: prompt template generator{DELIMITER}')
            try:
                json_string = extract_json_format_string(prompt_template_generator_agent_response)
                prompt_template = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
                prompt_template = {
                    "Task Type": parsed["Task Type"],
                    "Core Intent": parsed["Core Intent"],
                    "Key Details": parsed["Key Details"],
                    "Generated Prompt Template": user_query
                }
            self.logger.info(f"Task Type: {prompt_template['Task Type']}\nCore Intent: {prompt_template['Core Intent']}\nKey Details: {prompt_template['Key Details']}\nGenerated Prompt Template: {prompt_template['Generated Prompt Template']}")
            
            # step 3: Prompt Optimizer Agent
            optimizer_prefix = f'Now, based on the following details from the Prompt Template Generator Agent, optimize the prompt template:\nUser Query: {user_query}\nTask Type: {prompt_template["Task Type"]}\nCore Intent: {prompt_template["Core Intent"]}\nKey Details: {prompt_template["Key Details"]}\nGenerated Prompt Template: {prompt_template["Generated Prompt Template"]}'
            prompt_optimizer_agent_history = add_response("system", PROMPT_OPTIMIZER_AGENT_SYSTEM_PROMPT_FOR_USER, [])
            prompt_optimizer_agent_history = add_response("user", optimizer_prefix, prompt_optimizer_agent_history)
            prompt_optimizer_agent_response = inference_chat(prompt_optimizer_agent_history, model, endpoints, api_key, consume_token_path)
            CURRENT_STEP += 1
            self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: prompt optimizer{DELIMITER}')
            try:
                json_string = extract_json_format_string(prompt_optimizer_agent_response)
                optimized_prompt = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
                optimized_prompt = {
                    "Task Type": prompt_template["Task Type"],
                    "Core Intent": prompt_template["Core Intent"],
                    "Key Details": prompt_template["Key Details"],
                    "Optimized Prompt Template": user_query
                }
            self.logger.info(f"Task Type: {optimized_prompt['Task Type']}\nCore Intent: {optimized_prompt['Core Intent']}\nKey Details: {optimized_prompt['Key Details']}\nOptimized Prompt Template: {optimized_prompt['Optimized Prompt Template']}")

            # step 4: Final Prompt Generator Agent
            final_prompt_prefix = f'Now, based on the following details from the Prompt Optimizer Agent, generate the final user prompt:\nUser Query: {user_query}\nTask Type: {optimized_prompt["Task Type"]}\nCore Intent: {optimized_prompt["Core Intent"]}\nKey Details: {optimized_prompt["Key Details"]}\nOptimized Prompt Template: {optimized_prompt["Optimized Prompt Template"]}'
            final_prompt_generator_agent_history = add_response("system", FINAL_PROMPT_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_USER, [])
            final_prompt_generator_agent_history = add_response("user", final_prompt_prefix, final_prompt_generator_agent_history)
            final_prompt_generator_agent_response = inference_chat(final_prompt_generator_agent_history, model, endpoints, api_key, consume_token_path)
            CURRENT_STEP += 1
            self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: final prompt generator{DELIMITER}')
            try:
                json_string = extract_json_format_string(final_prompt_generator_agent_response)
                final_prompt = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
                final_prompt = {
                    "Task Type": optimized_prompt["Task Type"],
                    "Core Intent": optimized_prompt["Core Intent"],
                    "Key Details": optimized_prompt["Key Details"],
                    "Final Prompt": user_query
                }
            self.logger.info(f"Task Type: {final_prompt['Task Type']}\nCore Intent: {final_prompt['Core Intent']}\nKey Details: {final_prompt['Key Details']}\nFinal Prompt: {final_prompt['Final Prompt']}")

        elif self.demand_type == "agent":
            subtask, agent_role = self.info[0], self.info[1]

            # step 1: Prompt Template Generator Agent
            query_profix = f'Now generate your prompt template using the following inputs:\nSubtask: {subtask}\nAgent Role: {agent_role}'
            prompt_template_generator_agent_history = add_response("system", PROMPT_TEMPLATE_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_SUBTASK_AGENT, [])
            prompt_template_generator_agent_history = add_response("user", query_profix, prompt_template_generator_agent_history)
            prompt_template_generator_agent_response = inference_chat(prompt_template_generator_agent_history, model, endpoints, api_key, consume_token_path)
            CURRENT_STEP += 1
            self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: prompt template generator{DELIMITER}')
            try:
                json_string = extract_json_format_string(prompt_template_generator_agent_response)
                prompt_template = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
                prompt_template = {
                    "Agent Role": agent_role,
                    "Subtask": subtask,
                    "Prompt Template": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans."
                }
            self.logger.info(f"Agent Role: {prompt_template['Agent Role']}\nSubtask: {prompt_template['Subtask']}\nPrompt Template: {prompt_template['Prompt Template']}")
            
            # step 2: Prompt Optimizer Agent
            optimizer_prefix = f'Now optimize the following prompt template:\nAgent Role: {agent_role}\nSubtask: {subtask}\nOriginal Prompt Template: {prompt_template["Prompt Template"]}'
            prompt_optimizer_agent_history = add_response("system", PROMPT_OPTIMIZER_AGENT_SYSTEM_PROMPT_FOR_SUBTASK_AGENT, [])
            prompt_optimizer_agent_history = add_response("user", optimizer_prefix, prompt_optimizer_agent_history)
            prompt_optimizer_agent_response = inference_chat(prompt_optimizer_agent_history, model, endpoints, api_key, consume_token_path)
            CURRENT_STEP += 1
            self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: prompt optimizer{DELIMITER}')
            try:
                json_string = extract_json_format_string(prompt_optimizer_agent_response)
                optimized_prompt = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
                optimized_prompt = {
                    "Agent Role": agent_role,
                    "Subtask": subtask,
                    "Optimized Prompt Template": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans."
                }
            self.logger.info(f"Agent Role: {optimized_prompt['Agent Role']}\nSubtask: {optimized_prompt['Subtask']}\nOptimized Prompt Template: {optimized_prompt['Optimized Prompt Template']}")
            
            # step 3: Final Prompt Generator Agent
            final_prompt_prefix = f'Now generate the final prompt using the following inputs:\nAgent Role: {agent_role}\nSubtask: {subtask}\nOptimized Prompt Template: {optimized_prompt["Optimized Prompt Template"]}'
            final_prompt_generator_agent_history = add_response("system", FINAL_PROMPT_GENERATOR_AGENT_SYSTEM_PROMPT_FOR_SUBTASK_AGENT, [])
            final_prompt_generator_agent_history = add_response("user", final_prompt_prefix, final_prompt_generator_agent_history)
            final_prompt_generator_agent_response = inference_chat(final_prompt_generator_agent_history, model, endpoints, api_key, consume_token_path)
            CURRENT_STEP += 1
            self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: final prompt generator{DELIMITER}')
            try:
                json_string = extract_json_format_string(final_prompt_generator_agent_response)
                final_prompt = json.loads(json_string)
            except json.JSONDecodeError as e:
                self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
                final_prompt = {
                    "Agent Role": agent_role,
                    "Subtask": subtask,
                    "Final Prompt": "You are a super-intelligent AI assistant capable of performing tasks more effectively than humans."
                }
            self.logger.info(f"Agent Role: {final_prompt['Agent Role']}\nSubtask: {final_prompt['Subtask']}\nFinal Prompt: {final_prompt['Final Prompt']}")
        else:
            raise ValueError("Invalid demand type. Must be 'user' or 'agent'.")
        return final_prompt


class ParseMATHAnswerAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 answer: str) -> None:
        self.logger = logger
        self.answer = answer
    
    def parse_answer(self,
                     model: str,
                     endpoints: str,
                     api_key: str,
                     consume_token_path: Optional[str] = None) -> dict:
        global CURRENT_STEP

        # parse MATH answer
        parse_answer_prefix = f'Now, please parse the following MATH answer:\nMATH reasoning result: {self.answer}'
        parse_answer_history = add_response("system", PARSE_MATH_ANSWER_AGENT_SYSTEM_PROMPT, [])
        parse_answer_history = add_response("user", parse_answer_prefix, parse_answer_history)
        parse_answer_response = inference_chat(parse_answer_history, model, endpoints, api_key, consume_token_path)
        CURRENT_STEP += 1
        self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: parse MATH answer{DELIMITER}')
        try:
            json_string = extract_json_format_string(parse_answer_response)
            parsed_answer = json.loads(json_string)

            if "answer" not in parsed_answer:
                if len(parsed_answer) == 1:
                    for key in parsed_answer:
                        parsed_answer["answer"] = parsed_answer[key]
                else:
                    parsed_answer = {
                        "answer": "boxed$0$"
                    }
                        
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
            parsed_answer = {
                "answer": "boxed$0$"
            }
        self.logger.info(f"Parsed Answer: {parsed_answer['answer']}")
        return parsed_answer


class ParseMMLUAnswerAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 answer: str) -> None:
        self.logger = logger
        self.answer = answer
    
    def parse_answer(self,
                     model: str,
                     endpoints: str,
                     api_key: str,
                     consume_token_path: Optional[str] = None) -> dict:
        global CURRENT_STEP

        # parse MMLU answer
        parse_answer_prefix = f'Now, please parse the following MMLU answer:\nMMLU reasoning result: {self.answer}'
        parse_answer_history = add_response("system", PARSE_MMLU_ANSWER_AGENT_SYSTEM_PROMPT, [])
        parse_answer_history = add_response("user", parse_answer_prefix, parse_answer_history)
        parse_answer_response = inference_chat(parse_answer_history, model, endpoints, api_key, consume_token_path)
        CURRENT_STEP += 1
        self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: parse MMLU answer{DELIMITER}')
        try:
            json_string = extract_json_format_string(parse_answer_response)
            parsed_answer = json.loads(json_string)
            if len(parsed_answer['answer']) != 1:
                parsed_answer = {
                    "answer": "C"
                }
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
            parsed_answer = {
                "answer": "C"
            }
        self.logger.info(f"Parsed Answer: {parsed_answer['answer']}")
        return parsed_answer


class TaskDecompositionAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 task_type: str,
                 core_intent: str,
                 key_details: str,
                 optimize_user_prompt: str,
                 implementation_of_subtasks: list) -> None:
        self.logger = logger
        self.task_type = task_type
        self.core_intent = core_intent
        self.key_details = key_details
        self.optimize_user_prompt = optimize_user_prompt
        self.implementation_of_subtasks = implementation_of_subtasks
    
    def generate_subtask(self,
                         model: str,
                         endpoints: str,
                         api_key: str,
                         consume_token_path: Optional[str] = None) -> dict:
        global CURRENT_STEP

        if self.implementation_of_subtasks:
            implementation_info = ""
            for i, (subtask_name, subtask_answer, subtask_result) in enumerate(self.implementation_of_subtasks, 1):
                implementation_info += f"[subtask {i}]\n"
                implementation_info += f"subtask name: {subtask_name}\n"
                implementation_info += f"subtask answer: {subtask_answer}\n"
                implementation_info += f"subtask result: {subtask_result}\n"
            implementation_info = implementation_info.strip('\n')
        else:
            implementation_info = "No implement of subtask in history."

        # generate subtask for agent
        subtask_prefix = f'Now, based on the following details, produce only one subtask:\nUser Query: {self.optimize_user_prompt}\nTask Type: {self.task_type}\nCore Intent: {self.core_intent}\nKey Details: {self.key_details}\nImplementation history of subtasks:\n{implementation_info}'
        task_decompose_history = add_response("system", TASK_DECOMPOSITION_AGENT_SYSTEM_PROMPT, [])
        task_decompose_history = add_response("user", subtask_prefix, task_decompose_history)
        task_decompose_response = inference_chat(task_decompose_history, model, endpoints, api_key, consume_token_path)
        CURRENT_STEP += 1
        self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: task decomposition{DELIMITER}')
        try:
            if "stop" in task_decompose_response:
                self.logger.info("Task decomposition stopped.")
                return {
                    "next subtask": "stop"
                }
            else:
                json_string = extract_json_format_string(task_decompose_response)
                subtask = json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
            subtask = {
                "next subtask": self.optimize_user_prompt
            }
        self.logger.info(f"Next Subtask: {subtask['next subtask']}")
        return subtask


class GenerationRoleAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 task_type: str,
                 core_intent: str,
                 key_details: str,
                 optimize_user_prompt: str,
                 current_subtask: str) -> None:
        self.logger = logger
        self.task_type = task_type
        self.core_intent = core_intent
        self.key_details = key_details
        self.optimize_user_prompt = optimize_user_prompt
        self.current_subtask = current_subtask

    def generate_role(self,
                      model: str,
                      endpoints: str,
                      api_key: str,
                      consume_token_path: Optional[str] = None) -> dict:
        global CURRENT_STEP

        # generate agents for subtask
        generate_agents_prefix = f'Now, based on the following details, generate the specific roles of agents for the current subtask:\nUser Query: {self.optimize_user_prompt}\nTask Type: {self.task_type}\nCore Intent: {self.core_intent}\nKey Details: {self.key_details}\nCurrent Subtask: {self.current_subtask}'
        generate_agents_history = add_response("system", AGENT_GENERATION_FOR_SUBTASK_SYSTEM_PROMPT, [])
        generate_agents_history = add_response("user", generate_agents_prefix, generate_agents_history)
        generate_agents_response = inference_chat(generate_agents_history, model, endpoints, api_key, consume_token_path)
        CURRENT_STEP += 1
        self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: agent generation for subtask{DELIMITER}')
        try:
            json_string = extract_json_format_string(generate_agents_response)
            agents_roles = json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
            agents_roles = {
                "agent roles": "assistant",
            }
        self.logger.info(f"Agent Roles: {agents_roles['agent roles']}")
        return agents_roles


class EvalAnsScoreAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 task_type: str,
                 core_intent: str,
                 key_details: str,
                 optimize_user_prompt: str,
                 current_subtask: str,
                 agent_role: str,
                 agent_ans: str,
                 judge_flag: str) -> None:
        self.logger = logger
        self.task_type = task_type
        self.core_intent = core_intent
        self.key_details = key_details
        self.optimize_user_prompt = optimize_user_prompt
        self.current_subtask = current_subtask
        self.agent_role = agent_role
        self.agent_ans = agent_ans
        self.judge_flag = judge_flag
    
    def eval_score(self,
                   model: str,
                   endpoints: str,
                   api_key: str,
                   consume_token_path: Optional[str] = None) -> dict:
        global CURRENT_STEP

        # score for agent
        score_prefix = f'Now, based on the following details, please give a score for the specific roles of agents for the current subtask:\nUser Query: {self.optimize_user_prompt}\nTask Type: {self.task_type}\nCore Intent: {self.core_intent}\nKey Details: {self.key_details}\nCurrent Subtask: {self.current_subtask}\nAgent Role: {self.agent_role}\nOutput of the Agent Role: {self.agent_ans}\nJudge Flag: {self.judge_flag}'

        score_agent_history = add_response("system", SCORE_AGENT_SYSTEM_PROMPT, [])

        score_agent_history = add_response("user", score_prefix, score_agent_history)
        score_agents_response = inference_chat(score_agent_history, model, endpoints, api_key, consume_token_path)
        CURRENT_STEP += 1
        self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: score for the agent role of {self.agent_role}{DELIMITER}')
        try:
            json_string = extract_json_format_string(score_agents_response)
            agent_score = json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
            agent_score = {
                "score": 0.5
            }
        self.logger.info(f"score: {agent_score['score']}")
        return agent_score


class JudgeAnsAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 task_type: str,
                 core_intent: str,
                 key_details: str,
                 optimize_user_prompt: str,
                 current_subtask: str,
                 agent_role: str,
                 agent_ans: str) -> None:
        self.logger = logger

        self.task_type = task_type
        self.core_intent = core_intent
        self.key_details = key_details
        self.optimize_user_prompt = optimize_user_prompt
        self.current_subtask = current_subtask
        self.agent_role = agent_role
        self.agent_ans = agent_ans
    
    def judge(self,
              model: str,
              endpoints: str,
              api_key: str,
              consume_token_path: Optional[str] = None) -> dict:
        global CURRENT_STEP

        judge_prefix = f'Now, based on the following details, please give your judgement for the specific roles of agents for the current subtask:\nUser Query: {self.optimize_user_prompt}\nTask Type: {self.task_type}\nCore Intent: {self.core_intent}\nKey Details: {self.key_details}\nCurrent Subtask: {self.current_subtask}\nAgent Role: {self.agent_role}\nOutput of the Agent Role: {self.agent_ans}'

        judge_agent_history = add_response("system", JUDGE_AGENT_SYSTEM_PROMPT, [])
        judge_agent_history = add_response("user", judge_prefix, judge_agent_history)
        judge_agents_response = inference_chat(judge_agent_history, model, endpoints, api_key, consume_token_path)
        CURRENT_STEP += 1
        self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: judge for the agent role of {self.agent_role}{DELIMITER}')
        try:
            json_string = extract_json_format_string(judge_agents_response)
            judge_flag = json.loads(json_string)
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}\nSetting default values.")
            judge_flag = {
                "status": "continue"
            }
        self.logger.info(f"judge flag: {judge_flag['status']}")
        return judge_flag


class ExecSubtaskAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 agent_role: str,
                 untried_actions: list,
                 system_prompt: Optional[str] = "",
                 task_type: Optional[str] = "", 
                 core_intent: Optional[str] = "", 
                 key_details: Optional[str] = "", 
                 optimize_user_prompt: Optional[str] = "",
                 current_subtask: Optional[str] = "") -> None:
        self.logger = logger

        self.agent_role = agent_role
        self.untried_actions = untried_actions
        self.system_prompt = system_prompt

        self.task_type = task_type
        self.core_intent = core_intent
        self.key_details = key_details
        self.optimize_user_prompt = optimize_user_prompt
        self.current_subtask = current_subtask

        self.agent_chat_history = add_response("system", self.system_prompt, [])

        self.parent: Optional['ExecSubtaskAgent'] = None
        self.children: list['ExecSubtaskAgent'] = []
        self.answer: Optional[str] = None
        self.visits = 0
        self.value = 0
        self.judge_flag: Optional[str] = None
    
    def eval_ans_score(self,
                       model: str,
                       endpoints: str,
                       api_key: str,
                       consume_token_path: Optional[str] = None) -> None:
        score_agent = EvalAnsScoreAgent(self.logger, self.task_type, self.core_intent, self.key_details, self.optimize_user_prompt, self.current_subtask, self.agent_role, self.answer, self.judge_flag)
        score = score_agent.eval_score(model, endpoints, api_key, consume_token_path)
        self.value = score["score"]
    
    def judge_ans(self,
                  model: str,
                  endpoints: str,
                  api_key: str,
                  consume_token_path: Optional[str] = None) -> None:
        judge_agent = JudgeAnsAgent(self.logger, self.task_type, self.core_intent, self.key_details, self.optimize_user_prompt, self.current_subtask, self.agent_role, self.answer)
        judge = judge_agent.judge(model, endpoints, api_key, consume_token_path)
        self.judge_flag = judge["status"]
    
    def ucb1(self, 
             exploration_weight: float = 1.4) -> float:
        # root or unvisited node
        if not self.parent or self.visits == 0:
            return float('inf')
        return self.value / self.visits + exploration_weight * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def set_parent(self,
                   agent: 'ExecSubtaskAgent') -> None:
        self.parent = agent
    
    def add_other_agent_response(self,
                                 agent_role: str,
                                 agent_response: str) -> None:
        info_prefix = f'Here is the response from the agent role of {agent_role}:\n{agent_response}'
        self.agent_chat_history = add_response("user", info_prefix, self.agent_chat_history)
        response_prefix = f"Okey I should consider the response carefully."
        self.agent_chat_history = add_response("assistant", response_prefix, self.agent_chat_history)
    
    def solve_task(self,
                   model: str,
                   endpoints: str,
                   api_key: str,
                   consume_token_path: Optional[str] = None) -> None:
        global CURRENT_STEP

        # solve task
        subtask_solve_prefix = f'Now, based on the following details and your specific role, generate your answer for solving the subtask:\nUser Query: {self.optimize_user_prompt}\nTask Type: {self.task_type}\nCore Intent: {self.core_intent}\nKey Details: {self.key_details}\nCurrent Subtask: {self.current_subtask}'

        self.agent_chat_history = add_response("user", subtask_solve_prefix, self.agent_chat_history)

        task_solve_response = inference_chat(self.agent_chat_history, model, endpoints, api_key, consume_token_path)
        self.agent_chat_history = add_response("assistant", task_solve_response, self.agent_chat_history)

        CURRENT_STEP += 1
        self.logger.info(f'{DELIMITER}step {CURRENT_STEP}: the agent role of {self.agent_role} for solving subtask{DELIMITER}')
        self.logger.info(f"{self.agent_role} responses: {task_solve_response}")
        self.answer = task_solve_response


class MCTSAgentSelector(object):
    def __init__(self,
                 logger: logging.Logger,
                 model: str,
                 endpoints: str,
                 api_key: str,
                 consume_token_path: Optional[str] = None,
                 max_tree_depth: int = 5) -> None:
        self.logger = logger
        self.model = model
        self.endpoints = endpoints
        self.api_key = api_key
        self.consume_token_path = consume_token_path
        self.max_tree_depth = max_tree_depth
    
    def set_root_agent(self,
                       agent_role: str,
                       untried_actions: list) -> ExecSubtaskAgent:
        return ExecSubtaskAgent(self.logger, agent_role, untried_actions)

    def add_path_agent_response(self,
                                current_agent: ExecSubtaskAgent,
                                path: list[tuple[int, ExecSubtaskAgent]]) -> None:
        for i, used_agent in path:
            if i == 0:
                continue
            current_agent.add_other_agent_response(used_agent.agent_role, used_agent.answer)
        
    def select(self, 
               root: ExecSubtaskAgent) -> list[tuple[int, ExecSubtaskAgent]]:
        current_depth = 0
        path = [(current_depth, root)]
        node = copy.deepcopy(root)
        while node.children:
            node = max(node.children, key=lambda c: c.ucb1())
            current_depth += 1
            path.append((current_depth, node))
        return path
    
    def expand(self,
               path: list[tuple[int, ExecSubtaskAgent]],
               leaf: ExecSubtaskAgent,
               untried_actions: list,
               task_type: str, 
               core_intent: str, 
               key_details: str, 
               optimize_user_prompt: str,
               current_subtask: str) -> None:
        for agent_role, agent_system_prompt in leaf.untried_actions:
            # create agent
            specific_agent = ExecSubtaskAgent(self.logger, agent_role, untried_actions, agent_system_prompt, task_type, core_intent, key_details, optimize_user_prompt, current_subtask)
            # set parent
            specific_agent.set_parent(leaf)
            # add history
            self.add_path_agent_response(specific_agent, path)
            # generate ans
            specific_agent.solve_task(self.model, self.endpoints, self.api_key, self.consume_token_path)
            # judge ans
            specific_agent.judge_ans(self.model, self.endpoints, self.api_key, self.consume_token_path)
            # eval ans
            specific_agent.eval_ans_score(self.model, self.endpoints, self.api_key, self.consume_token_path)
            # set children
            leaf.children.append(specific_agent)
    
    def simulate(self, 
                 leaf: ExecSubtaskAgent,
                 current_depth: int,
                 path: list[tuple[int, ExecSubtaskAgent]],
                 untried_actions: list,
                 task_type: str, 
                 core_intent: str, 
                 key_details: str, 
                 optimize_user_prompt: str,
                 current_subtask: str) -> bool:
        node = copy.deepcopy(leaf)
        if node.judge_flag == "success":
            return True
        elif node.judge_flag == "fail":
            return False
        elif current_depth == self.max_tree_depth and node.judge_flag == "continue":
            return False
        elif node.judge_flag == "continue":
            additional_path = []
            for depth in range(current_depth + 1, self.max_tree_depth + 1):
                agent_role, agent_system_prompt = random.choice(node.untried_actions)
                # create agent
                specific_agent = ExecSubtaskAgent(self.logger, agent_role, untried_actions, agent_system_prompt, task_type, core_intent, key_details, optimize_user_prompt, current_subtask)
                # set parent
                specific_agent.set_parent(node)
                # add path
                additional_path.append((depth - 1, node))
                # add history
                self.add_path_agent_response(specific_agent, path + additional_path)
                # generate ans
                specific_agent.solve_task(self.model, self.endpoints, self.api_key, self.consume_token_path)
                # judge ans
                specific_agent.judge_ans(self.model, self.endpoints, self.api_key, self.consume_token_path)
                # eval ans
                specific_agent.eval_ans_score(self.model, self.endpoints, self.api_key, self.consume_token_path)

                if specific_agent.judge_flag == "success":
                    return True
                elif specific_agent.judge_flag == "fail":
                    return False
                elif specific_agent.judge_flag == "continue":
                    # move value
                    node = specific_agent
                else:
                    raise ValueError('judge_flag must be in "success", "fail", "continue"')
            else:
                return False
                
        else:
            raise ValueError('judge_flag must be in "success", "fail", "continue"')
    
    def backpropagate(self, 
                      path: list[tuple[int, ExecSubtaskAgent]], 
                      success: bool,
                      penalty_weight: float = -0.3,
                      reward_weight: float = 0.7) -> None:
        if success:
            value = reward_weight
        else:
            value = penalty_weight
        for _, node in reversed(path):
            node.visits += 1

            children_num = 0
            children_value =  0
            for children in node.children:
                children_num += 1
                children_value += children.value
            
            node.value = (node.value * node.visits + value + children_value) / (node.visits + children_num)

    def search(self, 
               untried_actions: list,
               task_type: str, 
               core_intent: str, 
               key_details: str, 
               optimize_user_prompt: str,
               current_subtask: str,
               attempts: int = 5) -> tuple[str, bool]:
        # initialize root
        root = self.set_root_agent("__root__", untried_actions)

        while True:
            path = self.select(root)
            leaf_depth, leaf = path[-1]

            if 0 < leaf_depth <= self.max_tree_depth and leaf.judge_flag == "success":
                return leaf.answer, True
            elif 0 < leaf_depth <= self.max_tree_depth and leaf.judge_flag == "fail":
                return leaf.answer, False
            elif leaf_depth >= self.max_tree_depth and leaf.judge_flag == "continue":
                return leaf.answer, False
            else:
                if not leaf.children:
                    self.expand(path, leaf, untried_actions, task_type, core_intent, key_details, optimize_user_prompt, current_subtask)
                best_child = max(leaf.children, key=lambda c: c.ucb1())
                for _ in range(attempts):
                    success = self.simulate(best_child, leaf_depth + 1, path, untried_actions, task_type, core_intent, key_details, optimize_user_prompt, current_subtask)
                    self.backpropagate(path + [(leaf_depth + 1, best_child)], success)


class mainAgent(object):
    def __init__(self,
                 logger: logging.Logger,
                 model: str,
                 endpoints: str,
                 api_key: str,
                 consume_token_path: Optional[str] = None,
                 task: str = "open_ended") -> None:
        self.logger = logger
        self.model = model
        self.endpoints = endpoints
        self.api_key = api_key
        self.consume_token_path = consume_token_path
        self.task = task
        
        if self.task == "open_ended":
            self.compare_func = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= BLEU_THERSHOLD * 100
        elif self.task == "human_eval":
            self.compare_func = lambda x, y: sentence_bleu(x, [y], lowercase=True).score >= BLEU_THERSHOLD * 100
        elif self.task == "MMLU":
            self.compare_func = lambda x, y: x == y
        elif self.task == "MATH":
            self.compare_func = is_equiv
        else:
            raise ValueError(f"Invalid task type: {self.task}. Must be 'human_eval', 'open_ended', 'MMLU' or 'MATH'.")
    
    def optimize_prompt(self,
                        demand_type: str,
                        info: list[str],
                        use_prompt_opt_flag: bool = True) -> tuple[str] | str:
        prompt_agent = promptAgent(self.logger, demand_type, info, use_prompt_opt_flag)
        final_prompt = prompt_agent.generate_prompt(self.model, self.endpoints, self.api_key, self.consume_token_path)
        if demand_type == "user":
            task_type, core_intent, key_details, optimize_user_prompt = final_prompt["Task Type"], final_prompt["Core Intent"], final_prompt["Key Details"], final_prompt["Final Prompt"]
            return task_type, core_intent, key_details, optimize_user_prompt
        elif demand_type == "agent":
            return final_prompt["Final Prompt"]
        else:
            raise ValueError("Invalid demand_type. Must be 'user' or 'agent'.")
    
    def task_decompose(self,
                       task_type: str,
                       core_intent: str,
                       key_details: str,
                       optimize_user_prompt: str,
                       implementation_of_subtasks: list) -> str:
        task_decomposition_agent = TaskDecompositionAgent(self.logger, task_type, core_intent, key_details, optimize_user_prompt, implementation_of_subtasks)
        current_subtask = task_decomposition_agent.generate_subtask(self.model, self.endpoints, self.api_key, self.consume_token_path)
        return current_subtask["next subtask"]
    
    def generate_agent_for_subtask(self,
                                   task_type: str,
                                   core_intent: str,
                                   key_details: str,
                                   optimize_user_prompt: str,
                                   current_subtask: str) -> list:
        generation_role_agent = GenerationRoleAgent(self.logger, task_type, core_intent, key_details, optimize_user_prompt, current_subtask)
        agents_roles = generation_role_agent.generate_role(self.model, self.endpoints, self.api_key, self.consume_token_path)
        return agents_roles["agent roles"]
    
    def parse_answer(self,
                     text: str,
                     match_signal: Optional[str] = None) -> str:
        
        if self.task == "open_ended":
            ans = text.strip()
        elif self.task == "human_eval":
            python_code = extract_python_code_block(text)
            if python_code:
                if match_signal in python_code:
                    func_body = extract_function_body(python_code, match_signal)
                    if func_body:
                        ans = func_body
                    else:
                        ans = python_code
                else:
                    ans = python_code
            else:
                ans = text
        elif self.task == "MMLU":
            # parse MMLU answer
            parse_answer_agent = ParseMMLUAnswerAgent(self.logger, text)
            parsed_answer = parse_answer_agent.parse_answer(self.model, self.endpoints, self.api_key, self.consume_token_path)
            ans = parsed_answer["answer"]

            # no single choice
            if len(ans) != 1:
                ans = "C"

        elif self.task == "MATH":
            # parse MATH answer
            parse_answer_agent = ParseMATHAnswerAgent(self.logger, text)
            parsed_answer = parse_answer_agent.parse_answer(self.model, self.endpoints, self.api_key, self.consume_token_path)
            ans = parsed_answer["answer"]

            # no boxed
            if "boxed" not in ans:
                # but have answer
                if ans:
                    ans = f"boxed${ans}$"
                # empty string
                else:
                    ans = "boxed$0$"
            
            ans = find_math_answer(ans)
        else:
            raise ValueError("Invalid task type. Must be 'human_eval', 'open_ended', 'MMLU' or 'MATH'.")
        return ans

    def implement(self,
                  task_type: str, 
                  core_intent: str, 
                  key_details: str, 
                  optimize_user_prompt: str,
                  match_signal: Optional[str] = None,
                  max_task_decompose_times: int = 5,
                  max_tree_depth: int = 5,
                  attempts: int = 5,
                  use_high_level_agent: bool = True) -> str:
        implementation_of_subtasks = []
        extract_ans = []

        if not use_high_level_agent:
            max_task_decompose_times = 1


        for _ in range(max_task_decompose_times):
        
            # step 1. generate subtask for agent
            if use_high_level_agent:
                current_subtask = self.task_decompose(task_type, core_intent, key_details, optimize_user_prompt, implementation_of_subtasks)
                if current_subtask == "stop":
                    if not implementation_of_subtasks:
                        return "Failed to complete the task"
                    else:
                        return extract_ans[-1]
            else:
                current_subtask = optimize_user_prompt

            # step 2. generate agents for subtask
            agents_roles = self.generate_agent_for_subtask(task_type, core_intent, key_details, optimize_user_prompt, current_subtask)
            subtask_agents_list = []
            for agent_role in agents_roles:
                # generate prompt for agent
                agent_system_prompt = self.optimize_prompt("agent", [current_subtask, agent_role])
                subtask_agents_list.append((agent_role, agent_system_prompt))
            
            # step 3. MCTS
            mcts_agent = MCTSAgentSelector(self.logger, self.model, self.endpoints, self.api_key, self.consume_token_path, max_tree_depth)
            subtask_ans, subtask_result = mcts_agent.search(subtask_agents_list, task_type, core_intent, key_details, optimize_user_prompt, current_subtask, attempts)
            if subtask_result:
                implementation_of_subtasks.append((current_subtask, subtask_ans, "success"))
            else:
                implementation_of_subtasks.append((current_subtask, subtask_ans, "fail"))

            
            # step 4. early-stopping mechanism
            extract_ans.append(self.parse_answer(subtask_ans, match_signal))
            if not use_high_level_agent:
                return extract_ans[-1]

            if len(extract_ans) > 1:
                best_ans, early_stop_flag = most_frequent(extract_ans, self.compare_func)
                if early_stop_flag:
                    self.logger.info(f"Early stopping mechanism triggered")
                    return best_ans

        else:
            return "Failed to complete the task"

    def forword(self,
                user_init_query: str,
                match_signal: Optional[str] = None,
                max_task_decompose_times: int = 5,
                max_tree_depth: int = 5,
                attempts: int = 5,
                use_prompt_opt_flag: bool = True,
                use_high_level_agent: bool = True) -> str:
        self.logger.info(f'{DELIMITER}user query{DELIMITER}\n{user_init_query}')
        
        # optimize user query
        task_type, core_intent, key_details, optimize_user_prompt = self.optimize_prompt("user", [user_init_query], use_prompt_opt_flag)

        # implement task
        ans = self.implement(task_type, core_intent, key_details, optimize_user_prompt, match_signal, max_task_decompose_times, max_tree_depth, attempts, use_high_level_agent)

        if ans == "Failed to complete the task":
            if self.task == "human_eval":
                ans = "return None"
            elif self.task == "MMLU":
                ans = "C"
            elif self.task == "MATH":
                ans = "0"
            
        self.logger.info(f'{DELIMITER}final answer{DELIMITER}\n{ans}\n{DELIMITER}final answer ended{DELIMITER}')
        return ans
