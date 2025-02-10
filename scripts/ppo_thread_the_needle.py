import gym
import lightning as pl
import thread_the_needle as ttn
from pytorch_lightning.loggers import TensorBoardLogger

from vi_ppo.actor_critic import ActorCritic
from vi_ppo.modules import ThreadTheNeedleModule
from vi_ppo.nets.cnn import Cnn
from vi_ppo.nets.mlp import Mlp

N_SIMULATIONS = 16


def make_agent():
    hidden_dims = 16

    feature_extractor_config = Cnn.config_cls(
        input_channels=1,
        channels=[8, 16, 1],
        kernel_sizes=[8, 4, 1],
        strides=[2, 2, 1],
        padding=[0, 0, 0],
        flatten_output=True,
        activation="silu",
    )

    feature_extractor = Cnn(feature_extractor_config)

    embedding_dims = feature_extractor.calculate_output_shape(input_shape=(1, 64, 64))[
        1
    ]

    actor_config = Mlp.config_cls(
        input_dims=embedding_dims,
        output_dims=n_a,
        hidden_dims=hidden_dims,
        n_layers=1,
        activation="silu",
    )
    critic_config = Mlp.config_cls(
        input_dims=embedding_dims,
        output_dims=1,
        hidden_dims=hidden_dims,
        n_layers=1,
        activation="silu",
    )
    ac_config = ActorCritic.config_cls(
        clip_epsilon=0.2, value_coeff=0.5, entropy_coeff=0.01
    )

    model = ActorCritic(
        ac_config,
        actor_net=Mlp(actor_config),
        critic=Mlp(critic_config),
        feature_extractor=feature_extractor,
    )

    return model


def main():

    # Create the environment
    env = ttn.make("thread_the_needle")

    # make the actor critic model
    d = env.observation_space.shape
    n_a = env.action_space.n

    print("Observation space: ", d)
    print("Action space: ", n_a)

    for _ in range(N_SIMULATIONS):
        model = make_agent()

        config = ThreadTheNeedleModule.config_class(lr=3e-4)
        module = ThreadTheNeedleModule(actor_critic=model, env=env, config=config)

        logger = TensorBoardLogger("../lightning_logs", name="thread_the_needle")
        trainer = pl.Trainer(max_epochs=100, logger=logger)

        trainer.fit(module)


if __name__ == "__main__":
    main()
