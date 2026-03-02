import numpy as np

goal2num_dict = {
  "sleep": 1,
  "wake_up": 2,
  "move": 3,
  "drink_water": 4,
  "collect_wood": 5,
  "collect_sapling": 6,
  "collect_stone": 7,
  "collect_coal": 8,
  "collect_iron": 9,
  "collect_diamond": 10,
  "collect_fence": 11,
  "eat_cow": 12,
  "eat_plant": 13,
  "attack_cow": 14,
  "attack_zombie": 15,
  "attack_skeleton": 16,
  "place_furnace": 17,
  "place_plant": 18,
  "place_stone": 19,
  "place_table": 20,
  "make_wood_pickaxe": 21,
  "make_wood_sword": 22,
  "make_stone_pickaxe": 23,
  "make_stone_sword": 24,
  "make_iron_pickaxe": 25,
  "make_iron_sword": 26,
}

query_prompt =  "As a professional game analyst, your role is to oversee a DreamerV3 agent—a player in a game resembling Minecraft. \
                You will receive an unfinished achievement and a starting point that includes information in four parts: \
                what the player sees, the player's status, achievements yet to be completed, and the current transition. \
                For this starting point, \
                please provide the top 5 key actions the player should take in the next 16 time steps to complete this achievement. \
                Consider the feasibility of each action in the current state and its importance to achieving the achievement. \
                All valid actions include: \
                'sleep', 'drink water', 'wake up', \
                'collect wood', 'collect sapling', 'collect diamond', 'collect stone', \
                'collect coal', 'collect iron', 'collect diamond', \
                'place table', 'place stone', 'place furnace', \
                'make wood pickaxe', 'make wood sword', \
                'make stone pickaxe', 'make stone sword', \
                'attack cow', 'attack zombie', 'attack skeleton', \
                'eat cow', 'eat plant'. \n \
                The materials required to make different items are as follows. \
                You need to compare the corresponding material quantities with the quantity of materials in the inventory.\
                collect:\
                  tree: {require: {}, receive: {wood: 1}, leaves: grass}\
                  stone: {require: {wood_pickaxe: 1}, receive: {stone: 1}, leaves: path}\
                  coal: {require: {wood_pickaxe: 1}, receive: {coal: 1}, leaves: path}\
                  iron: {require: {stone_pickaxe: 1}, receive: {iron: 1}, leaves: path}\
                  diamond: {require: {iron_pickaxe: 1}, receive: {diamond: 1}, leaves: path}\
                  water: {require: {}, receive: {drink: 1}, leaves: water}\
                  grass: {require: {}, receive: {sapling: 1}, probability: 0.1, leaves: grass}\
                \
                place:\
                  stone: {uses: {stone: 1}, where: [grass, sand, path, water, lava], type: material}\
                  table: {uses: {wood: 2}, where: [grass, sand, path], type: material}\
                  furnace: {uses: {stone: 4}, where: [grass, sand, path], type: material}\
                  plant: {uses: {sapling: 1}, where: [grass], type: object}\
                \
                make:\
                  wood_pickaxe: {uses: {wood: 1}, nearby: [table], gives: 1}\
                  stone_pickaxe: {uses: {wood: 1, stone: 1}, nearby: [table], gives: 1}\
                  iron_pickaxe: {uses: {wood: 1, coal: 1, iron: 1}, nearby: [table, furnace], gives: 1}\
                  wood_sword: {uses: {wood: 1}, nearby: [table], gives: 1}\
                  stone_sword: {uses: {wood: 1, stone: 1}, nearby: [table], gives: 1}\
                  iron_sword: {uses: {wood: 1, coal: 1, iron: 1}, nearby: [table, furnace], gives: 1}\
                The actions in the response are sorted from low to high according to the recommended strength. \n \
                The response should only include valid actions separated by ','. DO NOT INCLUDE ANY OTHER LETTERS, SYMBOLS, OR WORDS.\n"

type_dict = {
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

goal2num_dict = {
  "do nothing":0,
  "sleep":1,
  "wake up":2,
  "move":3,
  "drink water":4,
  "collect wood":5,
  "collect sapling":6,
  "collect stone":7,
  "collect coal":8,
  "collect iron":9,
  "collect diamond":10,
  "collect fence":11,
  "eat cow":12,
  "eat plant":13,
  "attack cow":14,
  "attack zombie":15,
  "attack skeleton":16,
  "place furnace":17,
  "place plant":18,
  "place stone":19,
  "place table":20,
  "make wood pickaxe":21,
  "make wood sword":22,
  "make stone pickaxe":23,
  "make stone sword":24,
  "make iron pickaxe":25,
  "make iron sword":26,
}

sorted_achievement_list = [
  'wake up', 
  'collect drink', 
  'collect wood', 
  'place table', 
  'collect sapling', 
  'place plant', 
  'eat plant', 
  'make wood pickaxe', 
  'make wood sword', 
  'eat cow', 
  'defeat zombie', 
  'defeat skeleton',
  'collect stone', 
  'place stone', 
  'make stone pickaxe', 
  'make stone sword', 
  'collect coal', 
  'place furnace', 
  'collect iron',
  'make iron pickaxe', 
  'make iron sword', 
  'collect diamond', 
]

def get_fovs(info):
    pos = info['player_pos']
    obs = info['semantic']
    fov_size = np.array([9, 7])
    top_left = np.maximum(pos - fov_size // 2, 0)
    bottom_right = np.minimum(pos + fov_size // 2 + 1, obs.shape)
    fov = obs[top_left[0]:bottom_right[0], top_left[1]:bottom_right[1]]
    pad_top = top_left[0] - pos[0] + fov_size[0] // 2
    pad_bottom = pos[0] + fov_size[0] // 2 + 1 - bottom_right[0]
    pad_left = top_left[1] - pos[1] + fov_size[1] // 2
    pad_right = pos[1] + fov_size[1] // 2 + 1 - bottom_right[1]
    fov = np.pad(fov, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    types = np.unique(fov)
    type_strings = [type_dict[t] for t in types if (t != 13 and t != 0)]
    return type_strings

def get_current_achievement(locked):
  for sort_achi in sorted_achievement_list:
    if sort_achi in locked:
      return sort_achi