import tensorflow as tf
import tensorflow_datasets as tfds

BATCH_SIZE = 50
IMAGE_DIM = 224


def preprocess_image(image, label):
    return tf.image.resize_with_pad(image, IMAGE_DIM, IMAGE_DIM), label


def process_image(ds):
    return (
        ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )


def get_base_model():
    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(224, 224, 3), include_top=False, weights="imagenet"
    )
    base_model.trainable = False
    return base_model


def get_model():
    base_model = get_base_model()
    inputs = tf.keras.Input(shape=(IMAGE_DIM, IMAGE_DIM, 3))
    rescale = tf.keras.applications.mobilenet_v3.preprocess_input(inputs)
    x = base_model(rescale, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(128)(x)
    return tf.keras.Model(inputs, outputs)


def main(epochs: int, version: int):
    (ds_train, ds_test), ds_info = tfds.load(
        "stanford_dogs",
        split=["train", "test"],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    ds_train = process_image(ds_train.shuffle(buffer_size=tf.data.AUTOTUNE))
    ds_test = process_image(ds_test)

    model = get_model()
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    model.fit(
        ds_train,
        epochs=epochs,
        validation_data=ds_test,
    )

    model.save(f"models/test_model/{version}")


if __name__ == "__main__":
    import typer

    typer.run(main)
