import math
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    VelocityPlayerToBallReward,
    FaceBallReward,
)
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import math as rl_math
import numpy as np
from misc_rewards import *


class SimplifiedBaseReward(RewardFunction):
    def __init__(self, team_spirit=0.1, boost_weight=1.0):
        super().__init__()
        self.team_spirit = team_spirit
        self.goal_reward = 10.0
        self.boost_weight = boost_weight
        self.reward = OncePerStepRewardWrapper(CombinedReward((
            CenterReward(non_participation_reward=self.team_spirit),
            ClearReward(non_participation_reward=self.team_spirit),
            EventReward(goal=self.goal_reward * (1 - self.team_spirit), team_goal=self.goal_reward * self.team_spirit,
                        shot=1.0, save=1.0, demo=1.0, boost_pickup=self.boost_weight),
            PositiveWrapperReward(VelocityBallToGoalReward()),
        ),
            (1.0, 1.0, 1.0, 0.334)))

    def reset(self, initial_state: GameState) -> None:
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)


class PersonalRewards(RewardFunction):
    """reward intended soley for the individual and not to be penalized"""
    def __init__(self, boost_weight=1.0):
        super().__init__()
        self.boost_weight = boost_weight
        self.boost_disc_weight = (self.boost_weight * 0.02222)*0.75
        self.reward = OncePerStepRewardWrapper(CombinedReward((
            TouchVelChange(),
            JumpTouchReward(min_height=130, exp=0.05),
            WallTouchReward(min_height=300, exp=0.025),
            VelocityPlayerToBallReward(),
        ),
            (0.5, 1.0, 1.0, 0.05)))

    def reset(self, initial_state: GameState) -> None:
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)


class RLFiveReward(RewardFunction):
    """Reward inspired by rewards in openai five paper: https://arxiv.org/abs/1912.06680"""
    def __init__(self, team_spirit=0.1):
        super().__init__()
        self.team_spirit = team_spirit
        self.blue_rewards = dict()
        self.orange_rewards = dict()
        self.prev_action_dummy = np.zeros(8)
        self.personal_rewards = dict()
        self.boost_weight = 0.75

    def reset(self, initial_state: GameState) -> None:
        for p in initial_state.players:
            if p.team_num == 0:
                self.blue_rewards[p.car_id] = SimplifiedBaseReward(team_spirit=self.team_spirit,
                                                                   boost_weight=self.boost_weight)
                self.blue_rewards[p.car_id].reset(initial_state)
            else:
                self.orange_rewards[p.car_id] = SimplifiedBaseReward(team_spirit=self.team_spirit,
                                                                     boost_weight=self.boost_weight)
                self.orange_rewards[p.car_id].reset(initial_state)
            self.personal_rewards[p.car_id] = PersonalRewards(boost_weight=self.boost_weight)
            self.personal_rewards[p.car_id].reset(initial_state)

    def get_enemy_average(self, enemy_team: int, state: GameState):  # previous action won't be available information
        if self.team_spirit > 0:
            enemies = [x for x in state.players if x.team_num == enemy_team]
            if len(enemies) < 1:
                return 0
            enemy_rewards = self.orange_rewards if enemy_team == 1 else self.blue_rewards
            reward_total = float(0)
            for e in enemies:
                reward_total += enemy_rewards[e.car_id].get_reward(e, state, self.prev_action_dummy)
            return reward_total/len(enemies) * self.team_spirit  # min(1, self.team_spirit * 2)
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        my_rewards = self.orange_rewards if player.team_num == 1 else self.blue_rewards
        base_reward = my_rewards[player.car_id].get_reward(player, state, previous_action)
        base_reward += self.personal_rewards[player.car_id].get_reward(player, state, previous_action)
        adjusted_reward = base_reward - self.get_enemy_average(1 if player.team_num == 0 else 0, state)
        return adjusted_reward


if __name__ == "__main__":
    pass
