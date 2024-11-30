import os
from pathlib import Path
from typing import Annotated

import tensorflow as tf
import typer
import yaml
from loguru import logger
from pydantic import BaseModel

import mingpt


class MinGPTConfig(BaseModel):
    dataset_url: str
    data_filepath: Path
    model_filepath: Path
    batch_size: int
    block_size: int
    embedding_dim: int
    num_heads_per_block: int
    num_attention_blocks: int
    dropout_rate: float
    learning_rate: float
    num_tokens_to_generate: int


# Default values
pretrained = True
config_filepath = "./superhero.yml"


app = typer.Typer(name="mingpt", help="Train minGPT with TensorFlow")


ConfigFilePathOption = Annotated[Path, typer.Option(help="Path to the config file.")]
PretrainedOption = Annotated[
    bool, typer.Option(help="If to use the pretrained model or not")
]


def load_config_file(filepath: Path) -> MinGPTConfig:
    with open(filepath, "r") as _file:
        config_data = yaml.safe_load(_file)
    return MinGPTConfig(**config_data)


def ensure_load_data(filepath: Path, dataset_url: str) -> str:
    if not os.path.isfile(filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        mingpt.data.download_example_data(filepath=filepath, dataset_url=dataset_url)
    return mingpt.data.load_data(filepath=filepath)


@app.command()
def generate(
    text_filepath: Annotated[
        Path, typer.Argument(help="Path to input text file for generating sequence.")
    ],
    config_filepath: ConfigFilePathOption = config_filepath,
    pretrained: PretrainedOption = pretrained,
):
    config = load_config_file(config_filepath)
    raw_data = ensure_load_data(config.data_filepath, config.dataset_url)
    vocab, encoder, decoder = mingpt.data.create_vocab(raw_data)

    model = mingpt.model.create_language_model(
        len(vocab),
        config.block_size,
        config.embedding_dim,
        config.num_heads_per_block,
        config.num_attention_blocks,
        config.dropout_rate,
        config.learning_rate,
    )

    if pretrained:
        model_filepath = str(config.model_filepath)
        model.load_weights(model_filepath)
        logger.info("Model checkpoint loaded")

    raw_input_text = mingpt.data.load_data(text_filepath)

    logger.info(f"Ending sequence: {raw_input_text[-config.block_size:]}")

    input_tokens = encoder(raw_input_text)
    num_tokens = len(input_tokens)

    if num_tokens > config.block_size:
        input_tokens = input_tokens[-config.block_size :]
    elif num_tokens < config.block_size:
        input_tokens = [encoder(" ")[0]] * (
            config.block_size - num_tokens
        ) + input_tokens

    sequence = mingpt.generate.generate_sequence(
        model=model,
        tokens=input_tokens,
        block_size=config.block_size,
        num_tokens_to_generate=config.num_tokens_to_generate,
    )

    logger.info(decoder(sequence))


@app.command()
def train(
    config_filepath: ConfigFilePathOption = config_filepath,
    pretrained: PretrainedOption = pretrained,
):
    config = load_config_file(config_filepath)
    raw_data = ensure_load_data(config.data_filepath, config.dataset_url)
    vocab, encoder, _ = mingpt.data.create_vocab(raw_data)

    train_data, valid_data = mingpt.data.create_dataset(encoder(raw_data))
    train_generator = mingpt.data.batch_generator(
        data=train_data, block_size=config.block_size
    )
    valid_generator = mingpt.data.batch_generator(
        data=valid_data, block_size=config.block_size
    )

    mirrored_strategy = tf.distribute.MirroredStrategy()

    num_gpus = mirrored_strategy.num_replicas_in_sync

    logger.info(f"Number of devices: {num_gpus}")

    # The examples will be spread out over all gpus, so per epoch steps will
    # reduce accordingly, let's run through all examples per epoch roughly twice
    num_training_examples = 2 * len(train_generator) // (num_gpus * config.batch_size)
    num_validation_examples = 2 * len(valid_generator) // (num_gpus * config.batch_size)

    train_generator = train_generator.batch(num_gpus * config.batch_size).repeat()
    valid_generator = valid_generator.batch(num_gpus * config.batch_size).repeat()

    train_generator = mirrored_strategy.experimental_distribute_dataset(train_generator)
    valid_generator = mirrored_strategy.experimental_distribute_dataset(valid_generator)

    model_filepath = str(config.model_filepath)

    with mirrored_strategy.scope():
        model = mingpt.model.create_language_model(
            len(vocab),
            config.block_size,
            config.embedding_dim,
            config.num_heads_per_block,
            config.num_attention_blocks,
            config.dropout_rate,
            config.learning_rate,
        )

        if pretrained and os.path.isfile(model_filepath):
            model.load_weights(model_filepath)
            logger.info("Model checkpoint loaded")

    logger.info(model.summary())

    mingpt.train.train_model(
        model,
        model_filepath,
        train_generator,
        valid_generator,
        steps=num_training_examples,
        validation_steps=num_validation_examples,
    )


if __name__ == "__main__":
    app()
