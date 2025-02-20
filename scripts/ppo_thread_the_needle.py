import argparse
from pathlib import Path

import lightning as pl
import thread_the_needle as ttn
from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from vi_ppo.actor_critic import ActorCritic
from vi_ppo.modules import ThreadTheNeedleModule
from vi_ppo.nets.cnn import Cnn
from vi_ppo.nets.mlp import Mlp


def make_agent(n_a, hidden_dims):
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


def main(args):
    # Create the environment
    env = ttn.make("thread_the_needle")

    # make the actor critic model
    d = env.observation_space.shape
    n_a = env.action_space.n

    print("Observation space: ", d)
    print("Action space: ", n_a)

    for _ in range(args.n_simulations):
        model = make_agent(n_a, args.hidden_dims)

        config = ThreadTheNeedleModule.config_class(lr=args.lr)
        module = ThreadTheNeedleModule(actor_critic=model, env=env, config=config)

        logger = TensorBoardLogger(args.log_dir, name="thread_the_needle/lr_3e-4")
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"{args.checkpoint_dir}/thread_the_needle/lr_3e-4",
            filename="thread_the_needle",
            save_top_k=1,
            monitor="train/total_reward",
            mode="max",
        )
        trainer = pl.Trainer(
            callbacks=[checkpoint_callback],
            max_epochs=args.max_epochs,
            logger=logger,
        )

        trainer.fit(module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Thread the Needle")
    parser.add_argument(
        "--n_simulations",
        type=int,
        default=25,
        help="Number of simulations",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        default=32,
        help="Hidden dimensions for MLP",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "lightning_logs"),
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=50,
        help="Maximum number of epochs",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "checkpoints"),
        help="Directory for model checkpoints",
    )

    args = parser.parse_args()
    main(args)
