import numpy as np
import tensorflow as tf


EPS = 1e-7


class AffinityLayer(tf.keras.layers.Layer):
    """A custom Keras layer for computing attention-based affinities.

    This layer takes three input tensors: query, key, and value, and computes
    attention-based affinities between them. It applies a weighted dot product
    attention mechanism to calculate the affinities and returns the weighted sum
    of the value tensor based on the computed affinities.

    Args:
    ----
        embedding_dim (int): The dimension of the embedding space.
        block_size (int): The size of the block for computing affinities.
        dropout_rate (float): The dropout rate to apply to the affinities.
        trainable (bool): Whether the layer's variables should be trainable.
        name (str): Optional name for the layer.
        dtype (tf.DType): The data type of the layer's variables.
        dynamic (bool): Whether the layer should be executed eagerly.
        **kwargs: Additional keyword arguments passed to the base class.

    Attributes:
    ----------
        _embedding_dim (int): The dimension of the embedding space.
        _dropout_rate (float): The dropout rate applied to the affinities.
        _tril (tf.Tensor): Lower triangular matrix used for masking affinities.

    Returns:
    -------
        tf.Tensor: The weighted sum of the value tensor based on the computed
            affinities.

    Example:
    -------
        Create an instance of AffinityLayer
        >>> layer = AffinityLayer(embedding_dim=64, block_size=8, dropout_rate=0.2)

        Assuming query, key, and value are properly defined, compute the
        attention-based affinities
        >>> output = layer([query, key, value])

        The output will be the weighted sum of the value tensor based on the
        computed affinities.
    """

    def __init__(
        self,
        embedding_dim: int,
        block_size: int,
        dropout_rate: float,
        trainable=True,
        name=None,
        dtype=None,
        dynamic=False,
        **kwargs,
    ):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)

        self._embedding_dim = embedding_dim
        self._dropout_rate = dropout_rate

        self._tril = tf.linalg.LinearOperatorLowerTriangular(
            tf.cast(tf.ones((block_size, block_size)), dtype=tf.bool)
        ).to_dense()

    def call(self, inputs: tuple[tf.Tensor]) -> tf.Tensor:
        """Computes the attention-based affinities and returns the weighted sum of
        the value tensor.

        Args:
        ----
            inputs (tuple[tf.Tensor]): A tuple containing the query, key, and
                value tensors.

        Returns:
        -------
            tf.Tensor: The weighted sum of the value tensor based on the computed
                affinities.
        """

        query, key, value = inputs
        # both are of shape batch_dim, block_size, head_size

        key = tf.transpose(key, perm=[0, 2, 1])
        # batch dot product
        affinities = tf.matmul(query, key)
        # The following weighting is done to retain the variance
        affinities = affinities * self._embedding_dim**-0.5
        # At this point affinities is of shape batch_size, block_size, block size
        affinities = tf.where(
            self._tril, affinities, tf.ones(self._tril.shape) * np.inf * -1
        )
        affinities = tf.nn.softmax(affinities)
        # Softmax is applied on the last dim, shape remains the same: batch_size, block_size, block size
        # dot product between this and value will give us the final affinities, return batch, block_size, head_size
        affinities = tf.nn.dropout(affinities, self._dropout_rate)
        return tf.matmul(affinities, value)

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {"_embedding_dim": self._embedding_dim, "_dropout_rate": self._dropout_rate}
        )
        return config


def create_single_head_attention_block(
    head_size: int, block_size: int, embedding_dim: int, dropout_rate: float
) -> tf.keras.Model:
    """Creates a single-head attention block model.

    This function creates a single-head attention block model using the given
    parameters. The attention block takes an input tensor of shape
    (batch_size, block_size, embedding_dim) and applies self-attention to
    compute the affinities between the input elements per example. The weighted
    sum of the value tensor based on the computed affinities is the final output.

    Args:
    ----
        head_size (int): The size of the attention head.
        block_size (int): The size of the attention block (context length).
        embedding_dim (int): The dimension of the embedding space.
        dropout_rate (float): The dropout rate applied to the attention weights.

    Returns:
    -------
        tf.keras.Model: A Keras model representing the single-head attention block.

    Example:
    -------
        Create a single-head attention block model
        >>> model = create_single_head_attention_block(
        >>>    head_size=64, block_size=8, embedding_dim=256, dropout_rate=0.1
        >>> )
        >>> output = model(input_tensor)
    """

    inputs = tf.keras.layers.Input(shape=(block_size, embedding_dim))

    key = tf.keras.layers.Dense(head_size, use_bias=False)(
        inputs
    )  # batch_dim, block_size, head_size
    query = tf.keras.layers.Dense(head_size, use_bias=False)(inputs)
    value = tf.keras.layers.Dense(head_size, use_bias=False)(inputs)

    affinities = AffinityLayer(embedding_dim, block_size, dropout_rate)(
        [query, key, value]
    )

    return tf.keras.models.Model(inputs, affinities)


def create_feed_forward_network(
    block_size: int, embedding_dim: int, dropout_rate: float
):
    """Creates a feed-forward network.

    This function creates a feed-forward network model with two dense layers and a
    dropout layer. The model takes an input tensor of shape (batch_size, block_size,
    embedding_dim), applies the dense layers with ReLU activation, and applies dropout
    regularization to the output.

    Args:
    ----
        block_size (int): The size of the input block.
        embedding_dim (int): The dimension of the embedding space.
        dropout_rate (float): The dropout rate applied to the network output.

    Returns:
    -------
        tf.keras.models.Sequential: A Keras sequential model representing the feed-forward network.
    """

    return tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(
                4 * embedding_dim,
                activation="relu",
                input_shape=(block_size, embedding_dim),
            ),
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dropout(dropout_rate),
        ]
    )


def create_multi_head_attention_block(
    num_heads: int,
    head_size: int,
    block_size: int,
    embedding_dim: int,
    dropout_rate: float,
) -> tf.keras.Model:
    """Creates a multi-head attention block.

    This function creates a multi-head attention block with multiple
    attention heads. The model takes an input tensor of shape
    (batch_size, block_size, embedding_dim), applies layer normalization to the
    input tensor, and then creates multiple single-head attention blocks. The
    outputs from the single-head attention blocks are concatenated along the last
    dimension, followed by a dense projection layer and a dropout layer.

    Args:
    ----
        num_heads (int): The number of attention heads.
        head_size (int): The size of each attention head.
        block_size (int): The size of the attention block.
        embedding_dim (int): The dimension of the embedding space.
        dropout_rate (float): The dropout rate applied to the projection layer.

    Returns:
    -------
        tf.keras.Model: A Keras model representing the multi-head attention block.
    """

    inputs = tf.keras.layers.Input(shape=(block_size, embedding_dim))

    normalised_inputs = tf.keras.layers.LayerNormalization(epsilon=EPS)(inputs)

    outputs = [
        create_single_head_attention_block(
            head_size, block_size, embedding_dim, dropout_rate
        )(normalised_inputs)
        for _ in range(0, num_heads)
    ]

    mh_attention_outputs = tf.keras.layers.concatenate(outputs, axis=-1)

    projection = tf.keras.layers.Dense(embedding_dim, activation="relu")(
        mh_attention_outputs
    )
    projection = tf.keras.layers.Dropout(dropout_rate)(projection)
    return tf.keras.models.Model(inputs, projection)


def create_full_block(
    block_size: int,
    embedding_dim: int,
    num_heads: int,
    head_size: int,
    dropout_rate: float,
):
    """Creates a full attention block.

    This function creates a full attention block that combines multi-head
    attention and feed-forward network layers. The model takes an input tensor of
    shape (batch_size, block_size, embedding_dim) followed by a multi-head attention block.
    The output from the attention block is added to the input tensor, and the result is
    normalized. Next, the normalized tensor is passed through a feed-forward network.
    The output from the network is added to the previous sum, resulting in the final output
    of the block.

    Args:
    ----
        block_size (int): The size of the attention block.
        embedding_dim (int): The dimension of the embedding space.
        num_heads (int): The number of attention heads.
        head_size (int): The size of each attention head.
        dropout_rate (float): The dropout rate applied to the networks.

    Returns:
    -------
        tf.keras.Model: A Keras model representing the full attention block.
    """

    inputs = tf.keras.layers.Input(shape=(block_size, embedding_dim))

    # normalised = tf.keras.layers.LayerNormalization()(inputs)
    mh_output = create_multi_head_attention_block(
        num_heads, head_size, block_size, embedding_dim, dropout_rate
    )(inputs)
    added = tf.keras.layers.Add()([mh_output, inputs])
    normalised = tf.keras.layers.LayerNormalization(epsilon=EPS)(added)
    outputs = create_feed_forward_network(block_size, embedding_dim, dropout_rate)(
        normalised
    )
    outputs = tf.keras.layers.Add()([outputs, added])

    return tf.keras.models.Model(inputs, outputs)


def create_language_model(
    vocab_size: int,
    block_size: int,
    embedding_dim: int,
    num_heads: int,
    num_attention_blocks: int,
    dropout_rate: float,
    learning_rate: float,
) -> tf.keras.Model:
    """Creates a language model based on the Transformer architecture.

    This function creates a language model using the Transformer architecture. The
    model consists of a stack of attention blocks, each containing a multi-head
    attention layer and a feed-forward network. The input to the model is a tensor
    of shape (batch_size, block_size), where each value represents a token from a
    vocabulary of size `vocab_size`. The model is trained to predict the next token
    in a sequence.

    Args:
    ----
        vocab_size (int): The size of the vocabulary.
        block_size (int): The length of the input block.
        embedding_dim (int): The dimension of the token embeddings.
        num_heads (int): The number of attention heads.
        num_attention_blocks (int): The number of attention blocks in the model.
        dropout_rate (float): The dropout rate applied to the attention and
            feed-forward layers.
        learning_rate (float): The learning rate for model training.

    Returns:
    -------
        tf.keras.Model: A Keras model representing the language model.
    """

    head_size = embedding_dim // num_heads

    inputs_tokens = tf.keras.layers.Input(shape=(block_size,))

    token_embeddings = tf.keras.layers.Embedding(
        input_dim=vocab_size, output_dim=embedding_dim
    )(inputs_tokens)
    position_embeddings = tf.keras.layers.Embedding(
        input_dim=block_size, output_dim=embedding_dim
    )(tf.range(0, block_size))
    position_embeddings = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=0))(
        position_embeddings
    )

    outputs = tf.keras.layers.Add()([token_embeddings, position_embeddings])

    for _ in range(0, num_attention_blocks):
        block = create_full_block(
            block_size, embedding_dim, num_heads, head_size, dropout_rate
        )

        outputs = block(outputs)

    outputs = tf.keras.layers.LayerNormalization(epsilon=EPS)(outputs)
    outputs = tf.keras.layers.Dense(vocab_size, name="logits")(outputs)
    outputs = tf.keras.layers.Softmax()(outputs)

    model = tf.keras.models.Model(inputs_tokens, outputs, name="mingpt")

    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=learning_rate, clipvalue=1.0),
        loss=tf.losses.SparseCategoricalCrossentropy(),
    )

    return model
