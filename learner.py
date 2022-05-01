import wandb
import torch.jit
from torch.nn import Linear, Sequential, GELU
from torch import nn
from redis import Redis
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from padded_obs import PaddedObs
from rlgym.utils.gamestates import PlayerData, GameState
from N_Parser import NectoAction
import numpy as np
from rl_five_reward import RLFiveReward


if __name__ == "__main__":
    frame_skip = 8
    half_life_seconds = 12
    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    run_id = None
    config = dict(
        actor_lr=5e-5,
        critic_lr=5e-5,
        n_steps=2_000_000,
        batch_size=200_000,
        minibatch_size=20_000,
        epochs=25,
        gamma=gamma,
        iterations_per_save=5,
        ent_coef=0.01,
    )

    wandb.login(key="NOPE")
    logger = wandb.init(name="NOPE", project="NOPE", entity="NOPE", id=run_id, config=config)

    r = Redis(password="NOPE")
    rollout_gen = RedisRolloutGenerator(r, lambda: PaddedObs(team_size=3, expanding=True), lambda: RLFiveReward(), lambda: NectoAction(),
                                        save_every=config["iterations_per_save"], logger=logger, clear=run_id is None, max_age=1)

    critic = Sequential(Linear(237, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 512),
                        GELU(), Linear(512, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 512), GELU(),
                        Linear(512, 1))

    actor = Sequential(Linear(237, 512), GELU(), Linear(512, 512), GELU(),
               Linear(512, 512), GELU(), Linear(512, 512), GELU(), Linear(512, 512), GELU(),
               Linear(512, 512), GELU(), Linear(512, 90))

    actor = DiscretePolicy(actor, (90,))

    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": config['actor_lr']},
        {"params": critic.parameters(), "lr": config['critic_lr']}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)

    alg = PPO(
        rollout_gen,
        agent,
        ent_coef=config['ent_coef'],
        n_steps=config['n_steps'],
        batch_size=config['batch_size'],
        minibatch_size=config['minibatch_size'],
        epochs=config['epochs'],
        gamma=config['gamma'],
        logger=logger,
    )

    try:
        if run_id is not None:
            alg.load("bumbler/.../checkpoint.pt")

    except Exception as e:
        print(e)
        print("Previous save not found.")

    alg.run(iterations_per_save=config['iterations_per_save'], save_dir="bumbler")
