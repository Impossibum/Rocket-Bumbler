import math
import numpy
from rlgym.utils.reward_functions import RewardFunction, CombinedReward
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import math as rl_math
import numpy as np
from misc_rewards import JumpTouchReward, WallTouchReward, BoostAcquisitions, \
    TouchVelChange, AerialTraining, RetreatReward, PositiveWrapperReward, CenterReward, ClearReward, BoostDiscipline, \
    FlatSpeedReward, OncePerStepRewardWrapper, GroundedReward, DemoPunish, VelocityBallToGoalReward, \
    VelocityPlayerToBallReward, EventReward, BoostTrainer, KickoffReward


class SimplifiedBaseReward(RewardFunction):
    def __init__(self, team_spirit=0.1, boost_weight=1.0):
        super().__init__()
        self.team_spirit = team_spirit
        self.goal_reward = 15.0
        self.boost_weight = boost_weight
        self.reward = None
        self.orange_count = 0
        self.blue_count = 0

    def setup_reward(self, initial_state: GameState) -> None:

        for p in initial_state.players:
            # no access to player team makes proper assignments difficult.
            # luckily this setup will work as long as teams are even
            if p.team_num == 1:
                self.orange_count += 1
            else:
                self.blue_count += 1

        self.goal_reward = 10 + (5 * self.blue_count)

        tgv = (self.goal_reward * self.team_spirit)/self.blue_count
        gv = (self.goal_reward * (1 - self.team_spirit)) - tgv
        self.reward = OncePerStepRewardWrapper(CombinedReward((
            EventReward(goal=gv, team_goal=tgv, demo=self.boost_weight, boost_pickup=self.boost_weight),
            TouchVelChange(),
            PositiveWrapperReward(VelocityBallToGoalReward()),
        ),
            # (1.0, 1.0, 0.05, 1.0, 0.2)))
            (1.0, 0.2, 0.05)))

    def reset(self, initial_state: GameState) -> None:
        if self.reward is None:
            self.setup_reward(initial_state)
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)


class PersonalRewards(RewardFunction): #reward intended soley for the individual and not to be penalized
    def __init__(self, boost_weight=1.0):
        super().__init__()
        self.boost_weight = boost_weight
        self.boost_disc_weight = self.boost_weight * 0.02222
        self.reward = OncePerStepRewardWrapper(CombinedReward((
            JumpTouchReward(min_height=500, exp=0.02),
            # VelocityPlayerToBallReward(), , 0.005
            BoostDiscipline(),
            RetreatReward(),
            KickoffReward(),
        ),
            (1.0, self.boost_disc_weight, 0.03334, 0.3334)))

    def reset(self, initial_state: GameState) -> None:
        self.reward.reset(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.reward.get_reward(player, state, previous_action)

class RLFiveReward(RewardFunction):
    def __init__(self, team_spirit=0.3):
        super().__init__()
        self.team_spirit = team_spirit
        self.blue_rewards = dict()
        self.orange_rewards = dict()
        self.prev_action_dummy = np.zeros(8)
        self.personal_rewards = dict()
        self.boost_weight = 3.0

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
            return reward_total/len(enemies)
        return 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        my_rewards = self.orange_rewards if player.team_num == 1 else self.blue_rewards
        base_reward = my_rewards[player.car_id].get_reward(player, state, previous_action)
        base_reward += self.personal_rewards[player.car_id].get_reward(player, state, previous_action)
        adjusted_reward = base_reward - self.get_enemy_average(1 if player.team_num == 0 else 0, state)
        return adjusted_reward
