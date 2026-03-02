import openai
import os
import requests
import json
import time
import re

# openai.api_key = "sk-9YkSkn0QXtEUDCb0ePpwAEfTbnBbwmOaZvvw47I8gQ50hAUG"
# openai.api_base = "https://api.chatanywhere.com.cn/v1"

model = ['gpt-4', 'gpt-4-32k', 'gpt-35-turbo']
model_choice = "gpt-35-turbo"
API_KEY = os.environ.get("OPENAI_API_KEY")
API_ENDPOINT = f"https://gcraoai5sw2.openai.azure.com/openai/deployments/{model_choice}/chat/completions?api-version=2023-03-15-preview"
headers = {'Content-Type': 'application/json', 'api-key': API_KEY}
max_wait_gpt4_time = 40


import json
import numpy as np
import time

import concurrent.futures

from timeout_decorator import timeout

achive22_list = ['collect coal', 'collect diamond', 'collect drink', \
  'collect iron', 'collect sapling', 'collect stone', 'collect wood', \
  'defeat skeleton', 'defeat zombie', 'eat cow', 'eat plant', \
  'make iron pickaxe', 'make iron sword', 'make stone pickaxe', \
  'make stone sword', 'make wood pickaxe', 'make wood sword', 'place furnace', \
  'place plant', 'place stone', 'place table', 'wake up']

num2tran_list = [
  "do nothing",
  "sleep",
  "wake up",
  "move",
  "drink water",
  "collect wood",
  "collect sapling",
  "collect stone",
  "collect coal",
  "collect iron",
  "collect diamond",
  "collect fence",
  "eat cow",
  "eat plant",
  "attack cow",
  "attack zombie",
  "attack skeleton",
  "place furnace",
  "place plant",
  "place stone",
  "place table",
  "make wood pickaxe",
  "make wood sword",
  "make stone pickaxe",
  "make stone sword",
  "make iron pickaxe",
  "make iron sword"
]

type_dict = {
    0: '??',
    1: 'water',
    2: 'grass',
    3: 'stone',
    4: 'path',
    5: 'sand',
    6: 'tree',
    7: 'lava',
    8: 'coal',
    9: 'iron',
    10: 'diamond',
    11: 'table',
    12: 'furnace',
    13: 'player',
    14: 'cow',
    15: 'zombie',
    16: 'skeleton',
    17: 'arrow',
    18: 'plant'
}

transition_sub_type_dict = {
    'do nothing': 0,
    'sleep': 1,
    'move': 2,
    'drink water': 3,
    'collect': 4,
    'place': 22,
    'make': 40,
    'attack': 59,
    'eat': 77
  }

transition_sub_type_dict_rev = {
    0: 'do nothing',
    1: 'sleep',
    2: 'move',
    3: 'drink water',
    4: 'collect',
    5: 'place',
    6: 'make',
    7: 'attack',
    8: 'eat'
  }

for i in range(94):
  if i not in transition_sub_type_dict_rev:
    index = (i - 4) // 18
    sub_index = (i - 4) % 18
    action = transition_sub_type_dict_rev[index + 4]
    obj = type_dict[sub_index + 1]
    tran_str = f"{action} {obj}"
    transition_sub_type_dict_rev[4 + index * 18 + sub_index] = tran_str
    transition_sub_type_dict[tran_str] = 4 + index * 18 + sub_index

reverse_type_dict = {value: key for key, value in type_dict.items()}

def _generate_prompt(**kargs):
  from crafter import constants
  achievements_c = constants.achievements.copy()
  prompt = 'Player see '
  if 'fov_token' in kargs.keys():
    n_item = kargs['fov_token'].shape[0]
    items = {type_dict[i + 1]: kargs['fov_token'][i] for i in range(n_item) if kargs['fov_token'][i]}
    for item, n in items.items():
      if n == 1:
        prompt += f'{item}, '
    prompt = prompt[:-2] + '.' if prompt[-2:] == ', ' else prompt
    prompt += ' Player have '
  if 'status_token' in kargs.keys():
    n_item = kargs['status_token'].shape[0]
    items = {type_dict[i]: kargs['status_token'][i] for i in range(n_item) if kargs['status_token'][i]}
    if items:
      for item, n in items.items():
        prompt += f"{n} {item}s " if n>1 else f"{n} {item}, "
      prompt = prompt[:-2] + '.' if prompt[-2:] == ', ' else prompt
    else:
      prompt += 'nothing.'
    prompt += ' '
  if 'achivement_token' in kargs.keys():
    n_item = kargs['achivement_token'].shape[0]
    items = [achievements_c[i] for i in range(n_item) if not kargs['achivement_token'][i]]
    if items:
      prompt += 'The achievements that have not been accomplished yet include: '
      for item in items:
        prompt += f'{item}, '
      prompt = prompt[:-2] + '.' if prompt[-2:] == ', ' else prompt
    else:
      prompt += 'All achievements have been accomplished.'
    prompt += ' '
  if 'transition_token' in kargs.keys():
    token = kargs['transition_token']
    if transition_sub_type_dict_rev[token]:
      prompt += f'Player {transition_sub_type_dict_rev[token]}.'
  return prompt

def generate_prompt(**kargs):
  assert bool(kargs.keys())
  n_prompts = list(kargs.values())[0].shape[0]
  prompts = []
  for i in range(n_prompts):
    prompts.append(_generate_prompt(**{key: kargs[key][i] for key in kargs.keys()}))
  return prompts
  
def traj2lang(traj: dict):
  token = {}
  length, bs, _ = list(traj.values())[0].shape
  if "achivement_token" in traj.keys():
    achivement_token = np.round(traj['achivement_token']).reshape((-1, traj["achivement_token"].shape[-1]))
    token["achivement_token"] = achivement_token
  if "fov_token" in traj.keys():
    fov_token = np.round(traj["fov_token"]).clip(0, 1).reshape((-1, traj["fov_token"].shape[-1]))
    token["fov_token"] = fov_token
  if "status_token" in traj.keys():
    status_token = np.round(traj['status_token']).reshape((-1, traj["status_token"].shape[-1]))
    token["status_token"] = status_token
  if "transition_token" in traj.keys():
    transition_token = np.argmax(traj['transition_token'], axis=-1).reshape(-1)
    token["transition_token"] = transition_token
  prompts = generate_prompt(**token)
  prompts = np.array(prompts).reshape((length, bs))
  return prompts

def traj2lang_small(traj: dict):
  token = {}
  length, _ = list(traj.values())[0].shape
  if "achivement_token" in traj.keys():
    achivement_token = np.round(traj['achivement_token']).reshape((-1, traj["achivement_token"].shape[-1]))
    token["achivement_token"] = achivement_token
  if "fov_token" in traj.keys():
    fov_token = np.round(traj["fov_token"]).clip(0, 1).reshape((-1, traj["fov_token"].shape[-1]))
    token["fov_token"] = fov_token
  if "status_token" in traj.keys():
    status_token = np.round(traj['status_token']).reshape((-1, traj["status_token"].shape[-1]))
    token["status_token"] = status_token
  if "transition_token" in traj.keys():
    transition_token = np.argmax(traj['transition_token'], axis=-1).reshape(-1)
    token["transition_token"] = transition_token
  prompts = generate_prompt(**token)
  prompts = np.array(prompts).reshape((length,))
  return prompts

def parse_transition_string(text):
  actions = text.strip().split(',')
  actions = [action.strip() for action in actions]
  return actions

def azure_api_call(prompts: list):
    data = {
        "messages": prompts,
        "temperature": 0.8,
    }
    while True:  # Add a loop to keep trying if there's an error
        response = requests.post(API_ENDPOINT, json=data, headers=headers)
        response = response.json()

        if 'error' in response:
            message = response['error']['message']
            sleep_time = int(re.findall(r'Please retry after (\w+) second', message)[0])
            #sleep_time = 5
            sleep_time = min(sleep_time, max_wait_gpt4_time)
            time.sleep(sleep_time + 1.0)
            # Continue the loop to retry
        else:
            choices_content = response['choices'][0]['message']['content']
            result_list = parse_transition_string(choices_content.lower())
            #print(result_list)
            return result_list


def gpt_35_api_stream(messages: list):
  cnt = 0
  while True:
    cnt += 1
    try:
        with concurrent.futures.ThreadPoolExecutor() as executor:
        #     future = executor.submit(openai.ChatCompletion.create,
        #                              model='gpt-3.5-turbo-1106',
        #                              messages=messages,
        #                              stream=True,
        #                              temperature=0, 
        #                             #  response_format={
        #                             #    "type": "json_object"
        #                             #   }
        #                              )
        #     response = future.result(timeout=10)
            future = executor.submit(azure_api_call, messages)
            response = future.result(timeout=10)
            return response
            # result = process_azure_response(response)
        #completion = process_azure_response(response)
        # completion = {'role': '', 'content': ''}
        # for event in response['choices'][0]['message']:
        #     if event['finish_reason'] == 'stop':
        #         if 'content' in completion:
        #             result = parse_transition_string(completion['content'].lower())
        #             return result
        #         else:
        #             break

        #     for delta_k, delta_v in event['message'].items():
        #         completion[delta_k] += delta_v
    # except concurrent.futures.TimeoutError:
    #     print("Time out")
    #     pass
    # except Exception as err:
    #     print(f"OpenAI API exception: {err}")
    #     pass
        # completion = {'role': '', 'content': ''}
        # for event in response:
        #     if event['choices'][0]['finish_reason'] == 'stop':
        #       if 'content' in completion:
        #         result = parse_transition_string(completion['content'].lower())
        #         return result
        #       else:
        #         break
        #     for delta_k, delta_v in event['choices'][0]['delta'].items():
        #         completion[delta_k] += delta_v
    except concurrent.futures.TimeoutError:
        print("Time out")
        pass
    # except Exception as err:
    #     print(f"OpenAI API exception: {err}")
    #     pass

def apply_intrinsic(response_dict):
  num_dict = np.zeros((27,))
  for i, name in enumerate(response_dict):
    for cnt, achi in enumerate(num2tran_list):
      if (name == achi):
        num_dict[cnt] = 0.25
  return num_dict


# messages = [
#   {
#     "role": "system", 
#     "content": "something like 1. collect sapling\n2. collect stone\n3. make wood pickaxe\n4. place table\n5. make stone pickaxe."
#   },
#   {
#     "role": "user", 
#     "content": "something like 1. collect sapling\n2. collect stone\n3. make wood pickaxe\n4. place table\n5. make stone pickaxe."
#   }
# ]


# gpt_35_api_stream(messages)
