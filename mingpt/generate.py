import numpy as np
import tensorflow as tf


def generate_sequence(
    model: tf.keras.Model,
    tokens: list[int],
    block_size: int,
    num_tokens_to_generate: int = 100,
) -> list[int]:
    """Create a sequence of tokens given input tokens and model to be used
    for predicting next tokens.

    Args:
    ----
        model (tf.keras.Model): A tf.keras model to be used for predicting
            probability distributions for next token which are then sampled
            from to select a single token.
        tokens (list[int]): A list of tokens to be used as sequence input.
        block_size (int): The size of each example block. This is the number
            of consecutive tokens to be considered as a single example.
        num_tokens_to_generate (int): Number of tokens to be generated.

    Returns:
    -------
        list[int]: A sequence of tokens that include the predicted tokens
            following the input tokens.
    """
    tokens = tokens[-block_size:]
    tokens = tf.convert_to_tensor(tokens, dtype=tf.int64)

    generation_model = tf.keras.models.Model(
        model.input, model.get_layer("logits").output
    )

    for _ in range(0, num_tokens_to_generate):
        inputs = tokens[-block_size:]
        inputs = tf.expand_dims(inputs, axis=0)

        predictions = generation_model.predict(inputs, verbose=False)
        prediction = predictions[:, -1, :]

        next_index = tf.raw_ops.Multinomial(logits=prediction, num_samples=1)

        tokens = tf.concat([tokens, next_index[0]], axis=-1)

    return np.array(tokens).tolist()
