import base64
import json
import math
import os
import random
import time
from typing import Optional, Callable
import re
import logging
from human_eval.data import read_problems
import ast
import pandas as pd
from utils.prompt import (
    MMLU_QUERY_QUESTION_PREFIX,
    MMLU_QUERY_FEW_SHOTS_PREFIX,
    MATH_FEW_SHOTS,
    MATH_QUERY_FEW_SHOTS_PREFIX
)


random.seed(10)


# read json
def read_json(file_path: str) -> dict | list:
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data


# write json
def write_json(file_path: str,
               input_data: dict | list,
               json_type: str = "dict") -> None:
    if json_type == "dict":
        data = input_data
    elif json_type == "list":
        try:
            data = read_json(file_path)
            if not isinstance(data, list):
                raise ValueError("JSON file need array format")
        except (FileNotFoundError, json.JSONDecodeError, ValueError):
            data = []
        data.append(input_data)
    else:
        raise ValueError("json_type must be 'dict' or 'list'")
    
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


# image encoding
def encode_image(image_path: str) -> bytes:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# apikey out of quota
class OutOfQuotaException(Exception):
    "Raised when the key exceeded the current quota"
    def __init__(self, 
                 key: str, 
                 cause: Optional[str] = None) -> None:
        super().__init__(f"No quota for key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


# api key with no permission
class AccessTerminatedException(Exception):
    "Raised when the key has been terminated"
    def __init__(self, 
                 key: str, 
                 cause: Optional[str] = None) -> None:
        super().__init__(f"Access terminated key: {key}")
        self.key = key
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{super().__str__()}. Caused by {self.cause}"
        else:
            return super().__str__()


def track_usage(res_json: dict) -> dict:
    usage = res_json['usage']
    prompt_tokens, completion_tokens, total_tokens = usage['prompt_tokens'], usage['completion_tokens'], usage['total_tokens']

    if "gpt-4o" in res_json['model']:
        prompt_token_price = (2.5 / 1000000) * prompt_tokens
        completion_token_price = (10 / 1000000) * completion_tokens
        return {
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": res_json.get('id', 'unknown'),
            "model": res_json.get('model', 'unknown'),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "prompt_token_price": prompt_token_price,
            "completion_token_price": completion_token_price,
            "total_price": prompt_token_price + completion_token_price
        }
    else:    
        return {
            "time": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            "id": res_json.get('id', 'unknown'),
            "model": res_json.get('model', 'unknown'),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }


def fix_common_json_issues(json_str: str) -> str:
    # 'key': 'value' → "key": "value"
    # json_str = re.sub(r"'([^']+)'\s*:\s*'([^']*)'", r'"\1": "\2"', json_str)

    json_str = re.sub(r'\\(?!["\\/bfnrt])', r'\\', json_str)

    if r'\\"""' in json_str:
        # \\""" -> \"\"\"
        json_str = json_str.replace(r'\\"""', r'\\"\\"\\"')
    if '"""' in json_str:
        # """ -> \"\"\"
        json_str = json_str.replace('"""', r'\\"\\"\\"')

    if r"\\'''" in json_str:
        # \\''' -> \\\'\'\'
        json_str = json_str.replace(r"\\'''", r"\\'\\'\\'")
    if "'''" in json_str:
        # ''' -> \'\'\'
        json_str = json_str.replace("'''", r"\\'\\'\\'")

    return json_str


def extract_json_format_string(text: str) -> str:
    # print(f"Original text: \n{text}")

    start = text.find('{')
    if start == -1:
        return ""
    
    in_string = False
    escape = False
    brace_count = 0

    for i in range(start, len(text)):
        char = text[i]
        
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and not escape:
            escape = True
            continue
        
        if not in_string:
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_string = text[start: i + 1]

                    try:
                        json.loads(json_string)
                        # print(f"Extracted JSON string: \n{json_string}")
                        return json_string
                    except json.JSONDecodeError:
                    
                        json_string = fix_common_json_issues(json_string)
                        # print(f"Extracted JSON string: \n{json_string}")
                        return json_string
        escape = False
    else:
        return ""


def setup_logger(log_path: str,
                 log_name: str = 'my_logger') -> logging.Logger:
    logger = logging.getLogger(log_name)
    logger.setLevel(logging.DEBUG)

    # create write file logging handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    # format
    formatter = logging.Formatter('%(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # create console logging handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # format
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


def most_frequent(agent_ans_list: list, 
                  cmp_func: Callable) -> tuple[str, bool]:
    counter = 0
    best_ans = agent_ans_list[0]

    for sub_ans in agent_ans_list:
        current_frequency = []
        for cmp_ans in agent_ans_list:
            if cmp_ans == sub_ans:
                continue
            current_frequency.append(cmp_func(sub_ans, cmp_ans))
            
        current_frequency = sum(current_frequency)
        if current_frequency > counter:
            counter = current_frequency
            best_ans = sub_ans
    
    if counter >= math.floor(2 / 3 * len(agent_ans_list)):
        return best_ans, True
    else:
        return best_ans, False


def get_human_eval_qa() -> list[str, str, str, dict]:
    problems = read_problems()
    qa = []
    for k, v in problems.items():
        qa.append((k, v["prompt"], v["entry_point"], v))

    return qa


def extract_python_code_block(text: str) -> str:
    pattern = r"```python(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""


def extract_function_body(code: str, 
                          function_name: str) -> str:
    tree = ast.parse(code)
    if not isinstance(tree, ast.Module):
        return ""

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            lines = code.splitlines()
            start_lineno = node.body[0].lineno - 1
            end_lineno = node.body[-1].end_lineno

            function_body_lines = lines[start_lineno:end_lineno]
            return "\n".join(function_body_lines)
    
    return ""


def get_subjects(mmlu_dataset_path: str,
                 ratio: float = 0.13) -> tuple[list[str], list[pd.DataFrame], list[pd.DataFrame]]:
    '''
    Randomly return ratio of the subjects(57 in total) and corresponding prompt words used for testing and dev prompts
    '''
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(mmlu_dataset_path, "test")) if "_test.csv" in f])

    dev_df_subject = []
    test_df_subject = []
    for subject in subjects:
        test_df = pd.read_csv(os.path.join(mmlu_dataset_path, "test", subject + "_test.csv"), header=None)
        total_rows = len(test_df)
        sample_size = int(total_rows * ratio)
        sample_indices = random.sample(range(total_rows), sample_size)
        sampled_test_df = test_df.iloc[sample_indices].reset_index(drop=True)
        dev_df = pd.read_csv(os.path.join(mmlu_dataset_path, "dev", subject + "_dev.csv"), header=None)
        test_df_subject.append(sampled_test_df)
        dev_df_subject.append(dev_df)

    return subjects, test_df_subject, dev_df_subject


def format_subject(subject: str) -> str:
    '''
    format the subject strings
    '''
    l = subject.split("_")
    s = ""
    for i, entry in enumerate(l):
        if i == 0:
            s += entry
        else:
            s += f" {entry}"
    return s


def format_example(df: pd.DataFrame, 
                   idx: int, 
                   include_answer: bool = True) -> str:
    '''
    format the subject example from a df into prompt, which conclude question, choices and correct choices
    '''
    choices = ["A", "B", "C", "D"]
    prompt = f"[Example {idx + 1}]\nquestion:\n```\n{df.iloc[idx, 0]}\n```\n"
    options = "options:\n```"
    k = df.shape[1] - 2
    for j in range(k):
        if j == k - 1:
            options += f"\n{choices[j]}. {df.iloc[idx, j + 1]}\n```\n"
        else:
            options += f"\n{choices[j]}. {df.iloc[idx, j + 1]}"
    answer = "answer:"
    if include_answer:
        answer += f" {df.iloc[idx, k + 1]}\n\n"
    prompt += options
    prompt += answer
    return prompt


def gen_prompt(train_df: pd.DataFrame, 
               subject: str, 
               k: int = -1) -> str:
    '''
    generate the prompt for few-shot learning
    '''
    process_subject = format_subject(subject)
    few_shots_prompt = ""
    if k == -1:
        k = train_df.shape[0]
    for i in range(k):
        few_shots_prompt += format_example(train_df, i)
    prompt = MMLU_QUERY_FEW_SHOTS_PREFIX.format(process_subject, few_shots_prompt)
    return prompt


def read_MMLU_line(line: pd.DataFrame) -> tuple[str, str, str]:
    choices = ["A", "B", "C", "D"]
    question = line[0]
    options = ""
    k = len(line) - 2
    for j in range(k):
        if j == k - 1:
            options += f"{choices[j]}. {line[j + 1]}"
        else:
            options += f"{choices[j]}. {line[j + 1]}\n"
    prompt = MMLU_QUERY_QUESTION_PREFIX.format(question, options)
    return prompt, question, options


def get_MMLU_qa(dev_df: pd.DataFrame, 
                test_df_line: pd.DataFrame, 
                subject: str,
                k: int = -1) -> tuple[str, str, str, str]:
    '''
    Return true label and prompt of MMMLU each line 
    '''
    prompt_end, question, options = read_MMLU_line(test_df_line)
    train_prompt = gen_prompt(dev_df, subject, k)
    prompt = train_prompt + prompt_end
    true_ans = test_df_line[5]
    return prompt, true_ans, question, options


def _remove_right_units(string: str) -> str:
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) >= 2
        return splits[0]
    else:
        return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) == 0:
                continue
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _strip_string(string: str) -> str:
    # linebreaks
    string = string.replace("\n", "")
    # print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    # print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    # print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    # print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    # print(string)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string


def find_math_answer(solution_string: str) -> str:
    assert 'boxed' in solution_string, f"no boxed found in solution: {solution_string}"
    ans = solution_string.split('boxed')[-1]
    if ans[0] == '{':
        stack = 1
        a = ''
        for c in ans[1:]:
            if c == '{':
                stack += 1
                a += c
            elif c == '}':
                stack -= 1
                if stack == 0:
                    break
                a += c
            else:
                a += c
    else:
        a = ans.split('$')[0].strip()
    
    a = _strip_string(a)
    return a


def is_equiv(str1: str, 
             str2: str, 
             verbose: bool = False) -> bool:
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


def get_MATH_question_type(file_dir: str,
                           csv_filename: str = "math_dataset_summary.csv",
                           total_count: int = 500) -> list[tuple[str, pd.DataFrame]]:
    '''
    Args:
        total_count: total number of samples for the overall test set (because rounding might be slightly less)
    '''
    file_path = os.path.join(file_dir, csv_filename)
    dataset = pd.read_csv(file_path)
    type_df = []
    types = dataset['type'].unique()
    type_ratios = [len(dataset[dataset['type'] == type_]) / len(dataset) for type_ in types]

    for type_, type_ratio in zip(types, type_ratios):
        subset = dataset[dataset['type'] == type_]
        sample_count = int(total_count * type_ratio)

        level_list = subset['level'].unique()
        level_ratios = [len(subset[subset['level'] == level_]) / len(subset) for level_ in level_list]

        sampled_levels = []
        for level_, level_ratio in zip(level_list, level_ratios):
            level_subset = subset[subset['level'] == level_]
            level_sample_count = int(sample_count * level_ratio)

            level_sample_count = min(level_sample_count, len(level_subset))

            sampled_level_subset = level_subset.sample(n=level_sample_count, random_state=10)
            sampled_levels.append(sampled_level_subset)

        sampled_subset = pd.concat(sampled_levels, ignore_index=True)
        type_df.append((type_, sampled_subset))

    return type_df


def get_MATH_prompt(line: pd.DataFrame) -> tuple[str, str, str, str]:
    '''
    Args:
        line: one row of dataframe, containing columns "problem" and "solution"
    Returns:
        prompt: the question string
        true_ans: the processed ground truth answer
    '''
    prob_content = line['problem']
    solution_content = line['solution']
    level = line['level']

    prompt = MATH_QUERY_FEW_SHOTS_PREFIX.format(MATH_FEW_SHOTS, prob_content)

    true_ans = find_math_answer(solution_content)

    return prompt, true_ans, prob_content, level


def extract_MATH_json_files(file_dir: str,
                            csv_filename: str = "math_dataset_summary.csv") -> None:
    file_test_dir = os.path.join(file_dir, "test")
    data = []
    for root, _, files in os.walk(file_test_dir):
        for file in files:
            if file.endswith('.json'):
                file_full_path = os.path.join(root, file)
                content = read_json(file_full_path)

                problem = content.get('problem', '')
                level = content.get('level', '')
                type = content.get('type', '')
                solution = content.get('solution', '')
                data.append({
                    'problem': problem,
                    'level': level,
                    'type': type, 
                    'solution': solution
                })

    df = pd.DataFrame(data)
    output_csv_path = os.path.join(file_dir, csv_filename)
    df.to_csv(output_csv_path, index=False)
    print(f"Saved {len(df)} entries to {output_csv_path}")
