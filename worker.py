import sys
from redis import Redis
from rlgym.envs import Match
from padded_obs import PaddedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter
from rlgym_tools.extra_state_setters.weighted_sample_setter import WeightedSampleSetter
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from N_Parser import NectoAction
from rl_five_reward import RLFiveReward
from torch import set_num_threads
set_num_threads(1)


if __name__ == "__main__":
    ts_index = int(sys.argv[1])
    team_sizes = [1, 2, 3, 1, 1, 2, 3, 1, 1, 2, 3, 1, 2, 1, 3, 1, 2, 3, 1, 2, 1, 3, 1, 2, 1, 3, 2, 1, 3, 2, 1, 1, 3, 2]
    team_size = team_sizes[ts_index]
    replay_options = ["platdiachampgcssl_1v1.npy", "platdiachampgcssl_2v2.npy", "platdiachampgcssl_3v3.npy"]

    match = Match(
        game_speed=100,
        self_play=True,
        team_size=team_size,
        state_setter=WeightedSampleSetter(
            (
                DefaultState(),
                AugmentSetter(ReplaySetter(replay_options[team_size-1])),
            ),
            (0.5, 1.0),
        ),
        obs_builder=PaddedObs(team_size=3, expanding=True),
        action_parser=NectoAction(),
        terminal_conditions=[TimeoutCondition(fps * 300), NoTouchTimeoutCondition(fps * 45), GoalScoredCondition()],
        reward_function=RLFiveReward()
    )

    r = Redis(password="none of your business ;p ")
    RedisRolloutWorker(r, "Impossibum", match, past_version_prob=0.1, sigma_target=1.5,
                       send_gamestates=False, evaluation_prob=0.01, force_paging=True).run()
