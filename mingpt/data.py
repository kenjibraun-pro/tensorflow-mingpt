"""Load and preprocess data for minGPT."""

from pathlib import Path
from typing import Callable

import requests
import tensorflow as tf
from loguru import logger


def download_example_data(filepath: Path, dataset_url: str):
    """
    Downloads example data from a given URL and saves it to the specified filepath.

    Args:
    ----
        filepath (Path): The path to save the downloaded file.
        dataset_name (str): Name of the dataset to be downloaded.

    Returns:
    -------
        None

    Example:
    -------
        >>> download_example_data(Path("data/input.txt"))
    """

    timeout = 1200

    response = requests.get(url=dataset_url, timeout=timeout)
    content = response.content.decode()

    with open(filepath, "w") as _file:
        _file.write(content)

    logger.info(f"File downloaded to {filepath}")


def load_data(filepath: Path) -> str:
    """
    Loads data from a given file.

    Args:
    ----
        filepath (Path): The path of the file to load.

    Returns:
    -------
        str: The loaded data as a string.

    Example:
    -------
        >>> data = load_data(Path("data/input.txt"))
    """
    data = None

    with open(filepath, "r") as _file:
        data = _file.read()

    logger.info(f"Data loaded from {filepath}")
    return data


def _encoder(char_to_index: dict[str, int]) -> Callable[[str], list[int]]:
    """
    Returns an encoding function that converts text into a list of encoded tokens.

    Args:
    ----
        char_to_index (dict[str, int]): A dictionary mapping characters to their corresponding indices.

    Returns:
    -------
        Callable[[str], list[str]]: The encoding function.

    Example:
    -------
        >>> encoder = _encoder(char_to_index)
        >>> encoded_text = encoder("Hello")
    """

    def _encode_text(text: str) -> list[int]:
        return [char_to_index[char] for char in text]

    return _encode_text


def _decoder(index_to_char: dict[int, str]) -> Callable[[list[int]], str]:
    """
    Returns a decoding function that converts a list of tokens into text.

    Args:
    ----
        index_to_char (dict[int, str]): A dictionary mapping indices to their corresponding characters.

    Returns:
    -------
        Callable[[list[int]], str]: The decoding function.

    Example:
    -------
        >>> decoder = _decoder(index_to_char)
        >>> decoded_text = decoder([0, 1, 2, 3, 4])
    """

    def _decode_tokens(tokens: list[int]) -> str:
        return "".join([index_to_char[token] for token in tokens])

    return _decode_tokens


def create_vocab(
    data: str,
) -> tuple[list[str], Callable[[str], list[str]], Callable[[list[int]], str]]:
    """
    Creates a vocabulary and returns the vocabulary list, an encoding function, and a decoding function.

    Args:
    ----
        data (str): The text corpus to create the vocabulary from.

    Returns:
    -------
        tuple[list[str], Callable[[str], list[str]], Callable[[list[int]], str]]: A tuple containing the vocabulary list,
        the encoding function, and the decoding function.

    Example:
    -------
        >>> vocab, encoder, decoder = create_vocab("abcd1234")
    """
    vocab = sorted(list(set(data)))
    char_to_index = {char: index for index, char in enumerate(vocab)}
    index_to_char = {value: key for key, value in char_to_index.items()}
    encoder = _encoder(char_to_index)
    decoder = _decoder(index_to_char)

    logger.info("Vocab, encoder and decoder created")

    return vocab, encoder, decoder


def create_dataset(tokens: list[int], split: float = 0.8) -> tuple[tf.Tensor]:
    """Create a dataset by splitting tokens into training and validation sets.

    Args:
    ----
        tokens (list[int]): A list of integers representing tokens.
        split (float, optional): The proportion of data to allocate for training.
            Defaults to 0.8.

    Returns:
    -------
        tuple[tf.Tensor]: A tuple containing the training and validation datasets
            as TensorFlow tensors.

    Raises:
    ------
        AssertionError: If the split value is not between 0 and 1 (inclusive).
    """

    if split < 0.0 or split > 1.0:
        raise AssertionError("split must be between 0 and 1")

    num_training_examples = int(len(tokens) * split)

    training_data = tokens[:num_training_examples]
    validation_data = tokens[num_training_examples:]

    training_data = tf.convert_to_tensor(training_data, dtype=tf.int64)
    validation_data = tf.convert_to_tensor(validation_data, dtype=tf.int64)

    return training_data, validation_data


def batch_generator(data: tf.Tensor, block_size: int) -> tf.data.Dataset:
    """Generate TensorFlow Dataset from tensor slices. Each item in the Dataset
    is a pair of examples and labels.

    Args:
    ----
        data (tf.Tensor): The input data as a TensorFlow tensor.
        block_size (int): The size of each example block. This is the number
            of consecutive tokens to be considered as a single example.

    Returns:
    -------
        tf.data.Dataset: A dataset containing the examples and labels.
    """

    num_examples = len(data) // block_size

    if len(data) % block_size == 0:
        data = tf.concat([data, [0]])

    examples = data[: num_examples * block_size]
    labels = data[1 : num_examples * block_size + 1]

    examples = tf.reshape(examples, [num_examples, block_size])
    labels = tf.reshape(labels, [num_examples, block_size])

    examples = tf.data.Dataset.from_tensor_slices(examples)
    labels = tf.data.Dataset.from_tensor_slices(labels)

    return tf.data.Dataset.zip((examples, labels))
