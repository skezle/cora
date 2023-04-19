import gym
from minihack import MiniHack

from envs import (
    skills_all,
    task_lavacross,
    task_medusa,
    task_mimic,
    task_seamonsters,
    task_simple,
)

ENVS = dict(
    # Skill Transfer Skills
    mini_skill_apply_frost_horn=skills_all.MiniHackSkillApplyFrostHorn,
    mini_skill_eat=skills_all.MiniHackSkillEat,
    mini_skill_fight=skills_all.MiniHackSkillFight,
    mini_skill_nav_blind=skills_all.MiniHackSkillNavigateBlind,
    mini_skill_nav_lava=skills_all.MiniHackSkillNavigateLava,
    mini_skill_nav_lava_to_amulet=skills_all.MiniHackSkillNavigateLavaToAmulet,
    mini_skill_nav_over_lava=skills_all.MiniHackSkillNavigateOverLava,
    mini_skill_nav_water=skills_all.MiniHackSkillNavigateWater,
    mini_skill_pick_up=skills_all.MiniHackSkillPickUp,
    mini_skill_put_on=skills_all.MiniHackSkillPutOn,
    mini_skill_take_off=skills_all.MiniHackSkillTakeOff,
    mini_skill_throw=skills_all.MiniHackSkillThrow,
    mini_skill_unlock=skills_all.MiniHackSkillUnlock,
    mini_skill_wear=skills_all.MiniHackSkillWear,
    mini_skill_wield=skills_all.MiniHackSkillWield,
    mini_skill_zap_cold=skills_all.MiniHackSkillZapColdWand,
    mini_skill_zap_death=skills_all.MiniHackSkillZapDeathWand,
    mini_skill_nav_blind_fixed=skills_all.MiniHackSkillNavigateBlindFixed,
    # Skill Transfer Tasks
    mini_lc_freeze=task_lavacross.MiniHackLCFreeze,
    mini_medusa=task_medusa.MiniHackMedusa,
    mini_mimic=task_mimic.MiniHackMimic,
    mini_seamonsters=task_seamonsters.MiniHackSeaMonsters,
    mini_simple_seq=task_simple.MiniHackSimpleSeq,
    mini_simple_intersection=task_simple.MiniHackSimpleIntersection,
    mini_simple_union=task_simple.MiniHackSimpleUnion,
    mini_simple_random=task_simple.MiniHackSimpleRandom,
)

from .image_task import ImageTask

# from skillhack agent/common/envs/wrapper.py

class CounterWrapper(gym.Wrapper):
    def __init__(self, env, state_counter="none"):
        # intialize state counter
        self.state_counter = state_counter
        if self.state_counter != "none":
            self.state_count_dict = defaultdict(int)
        # this super() goes to the parent of the particular task, not to object
        super().__init__(env)

    def step(self, action):
        # add state counting to step function if desired
        step_return = self.env.step(action)
        if self.state_counter == "none":
            # do nothing
            return step_return

        obs, reward, done, info = step_return

        if self.state_counter == "ones":
            # treat every state as unique
            state_visits = 1
        elif self.state_counter == "coordinates":
            # use the location of the agent in the dungeon to accumulate visits
            features = obs["blstats"]
            x = features[0]
            y = features[1]
            d = features[12]
            coord = (d, x, y)
            self.state_count_dict[coord] += 1
            state_visits = self.state_count_dict[coord]
        else:
            raise NotImplementedError("state_counter=%s" % self.state_counter)

        obs.update(state_visits=np.array([state_visits]))

        if done:
            self.state_count_dict.clear()

        return step_return

    def reset(self, wizkit_items=None):
        # reset state counter when env resets
        obs = self.env.reset(wizkit_items=wizkit_items)
        if self.state_counter != "none":
            self.state_count_dict.clear()
            # current state counts as one visit
            obs.update(state_visits=np.array([1]))
        return obs

def is_env_minihack(env_cls):
    return issubclass(env_cls, MiniHack)

def make_skillhack(
    env_name,
    flags,
    observation_keys=["pixel_crop"],
    reward_win=1,
    reward_lose=0,
    penalty_time=0.0,
    penalty_step=-0.001,  # MiniHack uses different than -0.01 default of NLE
    penalty_mode="constant",
    character="mon-hum-neu-mal",
    savedir=None,  # save_tty=False -> savedir=None, see https://github.com/MiniHackPlanet/MiniHack/blob/e124ae4c98936d0c0b3135bf5f202039d9074508/minihack/agent/common/envs/tasks.py#L168
    **kwargs,
):
    # env = gym.make(
    #     f"MiniHack-{env_name}",
    #     observation_keys=observation_keys,
    #     reward_win=reward_win,
    #     reward_lose=reward_lose,
    #     penalty_time=penalty_time,
    #     penalty_step=penalty_step,
    #     penalty_mode=penalty_mode,
    #     character=character,
    #     savedir=savedir,
    #     **kwargs,
    # ) 
    env_class = ENVS[env_name]

    if flags["save_tty"]:
        savedir = ""  # NLE choses location
    else:
        savedir = None

    kwargs = dict(
        savedir=savedir,
        archivefile=None,
        observation_keys=flags["obs_keys"].split(","),
        penalty_step=flags["penalty_step"],
        penalty_time=flags["penalty_time"],
        penalty_mode=flags["fn_penalty_step"],
    )
    if not is_env_minihack(env_class):
        kwargs.update(max_episode_steps=flags["max_num_steps"])
        kwargs.update(character=flags["character"])

    env = env_class(**kwargs)
    if flags["state_counter"] != "none":
        env = CounterWrapper(env, flags["state_counter"])
    if flags["seedspath"] is not None and len(flags["seedspath"]) > 0:
        raise NotImplementedError("seedspath > 0 not implemented yet.")

    return env

def get_single_skillhack_task(task_id, action_space_id, env_name, flags, num_timesteps, eval_mode=False, **kwargs):
    return ImageTask(
        task_id=task_id,
        action_space_id=action_space_id,
        env_spec=lambda: make_skillhack(env_name, flags, **kwargs),
        num_timesteps=num_timesteps,
        time_batch_size=1,  # no framestack
        eval_mode=eval_mode,
        image_size=[84, 84],
        grayscale=False,
    )
