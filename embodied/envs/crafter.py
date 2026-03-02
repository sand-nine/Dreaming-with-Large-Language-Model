import embodied
import numpy as np

from embodied import dicts

import pathlib
import pickle

from embodied.gpt_api import _generate_prompt, generate_prompt, traj2lang, traj2lang_small, parse_transition_string, gpt_35_api_stream, apply_intrinsic

class Crafter(embodied.Env):

  def __init__(self, task, size=(64, 64), outdir=None, seed=None):
    assert task in ('reward', 'noreward')
    import crafter
    self._env = crafter.Env(size=size, reward=(task == 'reward'), seed=seed)
    if outdir:
      outdir = embodied.Path(outdir)
      self._env = crafter.Recorder(
          self._env, outdir,
          save_stats=True,
          save_video=False,
          save_episode=False,
      )
    self._achievements = crafter.constants.achievements.copy()
    self._done = True

    #----goal part>>>>
    directory = pathlib.Path(__file__).resolve().parent
    with open(directory / "crafter_embeds.pkl", "rb") as f:
      self.cache = pickle.load(f)
    self._step = 0
    self.goal_tokens = np.zeros((5, 384))
    self.goal_id = np.zeros((5,))
    #----goal part<<<<

  @property
  def obs_space(self):
    spaces = {
        'image': embodied.Space(np.uint8, self._env.observation_space.shape),
        'transition_tokens': embodied.Space(np.uint32, (384,)),
        'goal_tokens': embodied.Space(np.uint32, (5, 384)),
        'goal_id': embodied.Space(np.uint32, (5,)),
        'reward': embodied.Space(np.float32),
        'is_first': embodied.Space(bool),
        'is_last': embodied.Space(bool),
        'is_terminal': embodied.Space(bool),
        'log_reward': embodied.Space(np.float32),
    }
    spaces.update({
        f'log_achievement_{k}': embodied.Space(np.int32)
        for k in self._achievements})
    return spaces

  @property
  def act_space(self):
    return {
        'action': embodied.Space(np.int32, (), 0, self._env.action_space.n),
        'reset': embodied.Space(bool),
    }

  def get_goals(self, info):
    fov = ', '.join(dicts.get_fovs(info))
    status = ', '.join([f"{v} {k}" for k, v in info['inventory'].items() if v > 0])
    locked = ', '.join([k for k, v in info['achievements'].items() if v == 0])
    unlocked = ', '.join([k for k, v in info['achievements'].items() if v != 0])
    prompt2gpt = ''
    prompt2gpt += 'The current achievement to be completed is {}'.format(dicts.get_current_achievement(locked))
    prompt2gpt += 'The player see: \n'
    prompt2gpt += fov + '\n'
    prompt2gpt += 'The status of the player is:'
    prompt2gpt += status + '\n'
    prompt2gpt += 'The locked achievements of the player are:'
    prompt2gpt += locked + '\n'
    prompt2gpt += 'The unlocked achievements of the player are:'
    prompt2gpt += unlocked + '\n'
    messages = [
      {
        "role": "system", 
        "content": dicts.query_prompt
      }, 
      {
        "role": "user", 
        "content": prompt2gpt
      }
    ]
    response_dict = gpt_35_api_stream(messages)
    return response_dict

  def step(self, action):
    if action['reset'] or self._done:
      self._done = False
      image = self._env.reset()
      return self._obs(image, 0.0, {}, is_first=True)
    image, reward, self._done, info = self._env.step(action['action'])

    # #----get goals>>>>
    # if (self._step % 10 == 0):
    #   #>>>>
    #   #goals = self.get_goals(info)
    #   print("Hi, you did not redo the get_goals part. :-)")
    #   goals = ["Hi, you did not redo the get_goals part. :-)"]
    #   #<<<<
    #   self.goal_tokens = np.zeros((5, 384))
    #   for i, goal in enumerate(goals[:5]):
    #     if goal in self.cache:
    #       self.goal_tokens[i] = self.cache[goal]
    #       self.goal_id[i] = dicts.goal2num_dict[goal]
    # self._step += 1

    # #----get goals>>>>
    if (self._step % 10 == 0):
    #>>>>
      #goals = self.get_goals(info)
      goals = self.get_goals(info)
    #<<<<
      self.goal_tokens = np.zeros((5, 384))
      for i, goal in enumerate(goals[:5]):
        if goal in self.cache:
          self.goal_tokens[i] = self.cache[goal]
          self.goal_id[i] = dicts.goal2num_dict[goal]
    self._step += 1

    reward = np.float32(reward)
    return self._obs(
        image, reward, info,
        is_last=self._done,
        is_terminal=info['discount'] == 0)
    #----get goals<<<<

  def get_transition(self, info):
    if 'transition' in info:
      text = info['transition']
      if text in self.cache:
        return self.cache[text]
    return np.zeros(384)

  def _obs(
      self, image, reward, info,
      is_first=False, is_last=False, is_terminal=False):
    log_achievements = {
        f'log_achievement_{k}': info['achievements'][k] if info else 0
        for k in self._achievements}
    return dict(
        image=image,
        reward=reward,
        transition_tokens = self.get_transition(info),
        goal_tokens = self.goal_tokens,
        goal_id = self.goal_id,
        is_first=is_first,
        is_last=is_last,
        is_terminal=is_terminal,
        log_reward=np.float32(info['reward'] if info else 0.0),
        **log_achievements,
    )

  def render(self):
    return self._env.render()
