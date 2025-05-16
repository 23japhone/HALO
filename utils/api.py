import requests
import time
from utils.basic_operation import (
    write_json,
    track_usage,
    encode_image,
    OutOfQuotaException,
    AccessTerminatedException
)
import json
from typing import Optional
import copy
import os
import re


META_JPEG = "image/jpeg"
META_PNG = "image/png"
META_GIF = "image/gif"
META_WEBP = "image/webp"


def add_response(role: str, 
                 prompt: str, 
                 chat_history: list,
                 image: list = [],
                 image_num_limit: int = 6) -> list:
    new_chat_history = copy.deepcopy(chat_history)
    content = [
        {
        "type": "text", 
        "text": prompt
        }
    ]
    if image:
        if len(image) > image_num_limit:
            raise ValueError(f"Image number exceeds the limit of {image_num_limit}.")
        
        for single_image_info in image:
            for img_type, img_path in single_image_info.items():
                # url
                if img_type == "url":
                    content.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": img_path
                        }
                    })
                # local
                elif img_type == "local":
                    base64_image = encode_image(img_path)
                    _, ext = os.path.splitext(img_path)
                    ext = ext.strip(".").lower()
                    if ext == "jpg" or ext == "jpeg":
                        meta_data = META_JPEG
                    elif ext == "png":
                        meta_data = META_PNG
                    elif ext == "gif":
                        meta_data = META_GIF
                    elif ext == "webp":
                        meta_data = META_WEBP
                    else:
                        raise ValueError(f"Unsupported image format: {ext}. Supported formats are jpg, jpeg, png, gif, webp.")
                    content.append({
                        "type": "image_url", 
                        "image_url": {
                            "url": f"data:{meta_data};base64,{base64_image}"
                        }
                    })
                else:
                    raise ValueError(f"Invalid image type: {img_type}. Supported types are 'url' and 'local'.")

    new_chat_history.append([role, content])
    return new_chat_history


def inference_chat(chat_history: list, 
                   model: str, 
                   endpoints: str, 
                   api_key: str, 
                   usage_tracking_path: Optional[str] = None, 
                   max_tokens: int = 2048, 
                   temperature: float = 0.8,
                   n: int = 1,
                   timeout: float = 600.0) -> str:    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": model,
        "messages": [],
        "max_tokens": max_tokens,
        'temperature': temperature,
        'n': n
    }

    # claude official
    if "claude" in model and "https://api.anthropic.com" in endpoints:
        # use claude official headers
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        for role, content in chat_history:
            if role == "system":
                assert content[0]['type'] == "text" and len(content) == 1, "system role must be a single text message"
                data['system'] = content[0]['text']
            else:
                converted_content = []
                for item in content:
                    if item['type'] == "text":
                        converted_content.append({
                            "type": "text", 
                            "text": item['text']
                        })
                    elif item['type'] == "image_url":
                        # url
                        if item['image_url']['url'].startswith("http"):
                            converted_content.append({
                                "type": "image",
                                "source": {
                                    "type": "url",
                                    "url": item['image_url']['url']
                                }
                            })
                        # base64
                        else:
                            # extract media_type
                            match = re.match(r"data:(image/\w+);base64,", item['image_url']['url'])
                            media_type = match.group(1) 
                            converted_content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": media_type,
                                    "data": item['image_url']['url'].replace(f"data:{media_type};base64,", "")
                                }
                            })
                    else:
                        raise ValueError(f"Invalid content type: {item['type']}")
                data["messages"].append({
                    "role": role, 
                    "content": converted_content
                })       
    # general
    else:
        for role, content in chat_history:
            data["messages"].append({
                "role": role, 
                "content": content
            })

    max_retry = 5
    sleep_sec = 20

    while max_retry > 0:
        try:
            if "claude" in model and "https://api.anthropic.com" in endpoints:
                res = requests.post(endpoints, headers=headers, data=json.dumps(data), timeout=timeout)
                res.raise_for_status()
                res_json = res.json()
                res_content = res_json['content'][0]['text']
            else:
                res = requests.post(endpoints, headers=headers, json=data, timeout=timeout)
                res.raise_for_status()
                res_json = res.json()
                res_content = res_json['choices'][0]['message']['content']

            if usage_tracking_path:
                usage = track_usage(res_json)
                write_json(usage_tracking_path, usage, json_type="list")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                if "You exceeded your current quota, please check your plan and billing details" in e.response.text:
                    raise OutOfQuotaException(api_key)
                elif "Your access was terminated due to violation of our policies" in e.response.text:
                    raise AccessTerminatedException(api_key)
                else:
                    print(f"Rate Limit Exceeded: {e.response.text}")
            else:
                print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Network Error: {e}")
        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
        except KeyError as e:
            print(f"Missing Key in Response: {e}")
        except Exception as e:
            print(f"Unexpected Error: {e}")
        else:
            # success break
            break

        print(f"Sleep {sleep_sec} before retry...")
        time.sleep(sleep_sec)
        max_retry -= 1

    else:
        print(f"Failed after {max_retry} retries...")
        raise RuntimeError("unable to connect to endpoints")
    
    return res_content
