import tensorflow as tf


def train_model(
    model: tf.keras.Model,
    model_filepath: str,
    training_generator: tf.data.Dataset,
    validation_generator: tf.data.Dataset,
    epochs: int = 1000,
    steps: int = 500,
    validation_steps: int = 200,
):
    """Train a model using the provided training and validation generators.

    Args:
    ----
        model (tf.keras.Model): The model to be trained.
        model_filepath (str): The file path to save the best model.
        training_generator (tf.data.Dataset): The training data generator.
        validation_generator (tf.data.Dataset): The validation data generator.
        epochs (int, optional): The number of training epochs. Defaults to 1000.
        steps (int, optional): The number of steps per epoch. Defaults to 500.
        validation_steps (int, optional): The number of steps for validation.
            Defaults to 200.

    Returns:
    -------
        None.
    """

    _ = model.fit(
        training_generator,
        validation_data=validation_generator,
        steps_per_epoch=steps,
        epochs=epochs,
        validation_steps=validation_steps,
        callbacks=[
            tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3, min_lr=1e-5),
            tf.keras.callbacks.EarlyStopping(patience=7),
            tf.keras.callbacks.ModelCheckpoint(model_filepath, save_best_only=True),
        ],
        verbose=True,
    )
