import math
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.reward_functions.common_rewards.player_ball_rewards import (
    VelocityPlayerToBallReward,
    FaceBallReward,
)
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils import math as rl_math
from rlgym.utils.reward_functions.common_rewards.ball_goal_rewards import VelocityBallToGoalReward
import numpy as np

SIDE_WALL_X = 4096  # +/-
BACK_WALL_Y = 5120  # +/-
CEILING_Z = 2044
BACK_NET_Y = 6000  # +/-

GOAL_HEIGHT = 642.775

ORANGE_GOAL_CENTER = (0, BACK_WALL_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_CENTER = (0, -BACK_WALL_Y, GOAL_HEIGHT / 2)

# Often more useful than center
ORANGE_GOAL_BACK = (0, BACK_NET_Y, GOAL_HEIGHT / 2)
BLUE_GOAL_BACK = (0, -BACK_NET_Y, GOAL_HEIGHT / 2)
BALL_RADIUS = 92.75

BALL_MAX_SPEED = 6000
CAR_MAX_SPEED = 2300
SUPERSONIC_THRESHOLD = 2200
CAR_MAX_ANG_VEL = 5.5

BLUE_TEAM = 0
ORANGE_TEAM = 1
NUM_ACTIONS = 8


class EventReward(RewardFunction):
    def __init__(self, goal=0., team_goal=0., concede=-0., touch=0., shot=0., save=0., demo=0., boost_pickup=0.):
        """
        :param goal: reward for goal scored by player.
        :param team_goal: reward for goal scored by player's team.
        :param concede: reward for goal scored by opponents. Should be negative if used as punishment.
        :param touch: reward for touching the ball.
        :param shot: reward for shooting the ball (as detected by Rocket League).
        :param save: reward for saving the ball (as detected by Rocket League).
        :param demo: reward for demolishing a player.
        :param boost_pickup: reward for picking up boost. big pad = +1.0 boost, small pad = +0.12 boost.
        """
        super().__init__()
        self.weights = np.array([goal, team_goal, concede, touch, shot, save, demo, boost_pickup])

        # Need to keep track of last registered value to detect changes
        self.last_registered_values = {}

    @staticmethod
    def _extract_values(player: PlayerData, state: GameState):
        if player.team_num == BLUE_TEAM:
            team, opponent = state.blue_score, state.orange_score
        else:
            team, opponent = state.orange_score, state.blue_score

        return np.array([player.match_goals, team, opponent, player.ball_touched, player.match_shots,
                         player.match_saves, player.match_demolishes, player.boost_amount])

    def reset(self, initial_state: GameState, optional_data=None):
        # Update every reset since rocket league may crash and be restarted with clean values
        self.last_registered_values = {}
        for player in initial_state.players:
            self.last_registered_values[player.car_id] = self._extract_values(player, initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray, optional_data=None):
        old_values = self.last_registered_values[player.car_id]
        new_values = self._extract_values(player, state)

        diff_values = new_values - old_values
        diff_values[diff_values < 0] = 0  # We only care about increasing values

        reward = np.dot(self.weights, diff_values)

        self.last_registered_values[player.car_id] = new_values
        return reward


class JumpTouchReward(RewardFunction):
    def __init__(self, min_height=92.75):
        self.min_height = min_height
        self.max_height = 2044-92.75
        self.range = self.max_height - self.min_height

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and not player.on_ground and state.ball.position[2] >= self.min_height:
            return (state.ball.position[2] - self.min_height) / self.range

        return 0


class CenterReward(RewardFunction):
    def __init__(self, centered_distance=1200, punish_area_exit=False, non_participation_reward=0.0):
        self.centered_distance = centered_distance
        self.punish_area_exit = punish_area_exit
        self.non_participation_reward = non_participation_reward
        self.centered = False
        self.goal_spot = np.array([0, 5120, 0])

    def reset(self, initial_state: GameState):
        self.centered = False

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball_loc = state.ball.position
        if player.team_num != 0:
            ball_loc = state.inverted_ball.position

        coord_diff = self.goal_spot - ball_loc
        ball_dist_2d = np.linalg.norm(coord_diff[:2])
        #ball_dist_2d = math.sqrt(coord_diff[0] * coord_diff[0] + coord_diff[1] * coord_diff[1])
        reward = 0

        if self.centered:
            if ball_dist_2d > self.centered_distance:
                self.centered = False
                if self.punish_area_exit:
                    reward -= 1
        else:
            if ball_dist_2d < self.centered_distance:
                self.centered = True
                if state.last_touch == player.car_id:
                    reward += 1
                else:
                    reward += self.non_participation_reward
        return reward


class ClearReward(RewardFunction):
    def __init__(self, protected_distance=1200, punish_area_entry=False, non_participation_reward=0.0):
        self.protected_distance = protected_distance
        self.punish_area_entry=punish_area_entry
        self.non_participation_reward = non_participation_reward
        self.needs_clear = False
        self.goal_spot = np.array([0, -5120, 0])

    def reset(self, initial_state: GameState):
        self.needs_clear = False

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        ball_loc = state.ball.position
        if player.team_num != 0:
            ball_loc = state.inverted_ball.position

        coord_diff = self.goal_spot - ball_loc
        ball_dist_2d = np.linalg.norm(coord_diff[:2])
        #ball_dist_2d = math.sqrt(coord_diff[0]*coord_diff[0] + coord_diff[1]*coord_diff[1])
        reward = 0

        if self.needs_clear:
            if ball_dist_2d > self.protected_distance:
                self.needs_clear = False
                if state.last_touch == player.car_id:
                    reward += 1
                else:
                    reward += self.non_participation_reward
        else:
            if ball_dist_2d < self.protected_distance:
                self.needs_clear = True
                if self.punish_area_entry:
                    reward -= 1
        return reward


class WallTouchReward(RewardFunction):
    def __init__(self, min_height=92, exp=0.2):
        self.min_height = min_height
        self.exp = exp
        self.max = math.inf

    def reset(self, initial_state: GameState):
        pass

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        if player.ball_touched and player.on_ground and state.ball.position[2] >= self.min_height:
            return (clamp(self.max, 0.0001, state.ball.position[2] - 92) ** self.exp)-1

        return 0


class TouchVelChange(RewardFunction):
    def __init__(self):
        self.last_vel = np.zeros(3)

    def reset(self, initial_state: GameState):
        self.last_vel = np.zeros(3)

    def get_reward(
        self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        reward = 0
        if player.ball_touched:
            vel_difference = abs(np.linalg.norm(self.last_vel - state.ball.linear_velocity))
            reward = vel_difference / 4600.0

        self.last_vel = state.ball.linear_velocity

        return reward


class PositiveWrapperReward(RewardFunction):
    """A simple wrapper to ensure a reward only returns positive values"""
    def __init__(self, base_reward):
        super().__init__()
        #pass in instantiated reward object
        self.rew = base_reward

    def reset(self, initial_state: GameState):
        self.rew.reset(initial_state)

    def get_reward(
            self, player: PlayerData, state: GameState, previous_action: np.ndarray
    ) -> float:
        rew = self.rew.get_reward(player, state, previous_action)
        return 0 if rew < 0 else rew


class OncePerStepRewardWrapper(RewardFunction):
    def __init__(self, reward):
        super().__init__()
        self.reward = reward
        self.gs = None
        self.rv = 0

    def reset(self, initial_state: GameState):
        self.reward.reset(initial_state)
        self.gs = None
        self.rv = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state == self.gs:
            return self.rv

        self.gs = state
        reward = self.reward.get_reward(player, state, previous_action)
        self.rv = reward
        return self.rv


if __name__ == "__main__":
    pass
