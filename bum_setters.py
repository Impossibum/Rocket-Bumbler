from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
import random
import numpy as np
from numpy import random as rand


class AerialSetup(StateSetter):

    SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
    SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                      0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
    SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                        [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
    SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                        np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]

    X_MAX = 3000
    Y_MAX = 4500
    Z_MAX_BALL = 1200
    Z_MAX_CAR = 1900
    PITCH_MAX = np.pi / 2
    YAW_MAX = np.pi
    ROLL_MAX = np.pi

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to simulate an aerial setup opportunity
        """
        ball_x = random.randrange(-self.X_MAX, self.X_MAX)#rand.random() * self.X_MAX - self.X_MAX / 2
        ball_y = random.randrange(-self.Y_MAX, self.Y_MAX)#rand.random() * self.Y_MAX - self.Y_MAX / 2
        state_wrapper.ball.set_pos(ball_x, ball_y, 140)
        state_wrapper.ball.set_lin_vel(x=random.randrange(-75, 75), y=random.randrange(-75, 75), z=1400)

        for car in state_wrapper.cars:
            ground_spawn = random.randint(0, 6) > 4
            c_z = 17
            if ground_spawn:
                c_z = 35
            c_x = random.randrange(ball_x-500, ball_x+500)
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                c_y = random.randrange(ball_y-500, ball_y)
                yaw = 0.5 * np.pi

            else:
                # select a unique spawn state from pre-determined values
                c_y = random.randrange(ball_y, ball_y + 500)
                yaw = -0.5 * np.pi

            pos = [c_x, c_y, c_z]

            car.set_pos(*pos)
            if not ground_spawn:
                car.set_lin_vel(*[0, 0, random.randint(100, 1000)])
            car.set_rot(yaw=yaw)
            car.boost = random.random()



