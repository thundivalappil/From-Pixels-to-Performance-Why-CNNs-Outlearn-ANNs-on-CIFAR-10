
#CIFAR-10 ANN (Baseline) + CNN 

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# 1) Load CIFAR-10

(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# 2) Normalize images

X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0


# 3) One-hot encode labels

y_train_oh = to_categorical(y_train, 10)
y_test_oh  = to_categorical(y_test, 10)

# Plot Accuracy & Loss

def plot_history(history, title_prefix):
    # Accuracy
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val Accuracy")
    plt.title(f"{title_prefix} Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Loss
    plt.figure()
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title(f"{title_prefix} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


ann = Sequential([
    Flatten(input_shape=(32, 32, 3)),
    Dense(256, activation="relu"),
    Dropout(0.30),
    Dense(10, activation="softmax")
])

ann.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

hist_ann = ann.fit(
    X_train, y_train_oh,
    validation_split=0.2,
    epochs=8,
    batch_size=64,
    verbose=1
)

loss_ann, acc_ann = ann.evaluate(X_test, y_test_oh, verbose=0)
print("\n==============================")
print("ANN RESULTS (Baseline)")
print("==============================")
print("Test Accuracy:", round(float(acc_ann) * 100, 2), "%")
print("Test Loss    :", round(float(loss_ann), 4))

plot_history(hist_ann, "ANN (Baseline)")


# B) CNN 

cnn = Sequential([
    Conv2D(32, (3,3), activation="relu", padding="same", input_shape=(32,32,3)),
    BatchNormalization(),
    Conv2D(32, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    Conv2D(64, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.30),

    Conv2D(128, (3,3), activation="relu", padding="same"),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Dropout(0.35),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.40),
    Dense(10, activation="softmax")
])

cnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

early_stop = EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)

hist_cnn = cnn.fit(
    X_train, y_train_oh,
    validation_split=0.2,
    epochs=40,
    batch_size=64,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

loss_cnn, acc_cnn = cnn.evaluate(X_test, y_test_oh, verbose=0)
print("\n==============================")
print("CNN RESULTS (Best)")
print("==============================")
print("Test Accuracy:", round(float(acc_cnn) * 100, 2), "%")
print("Test Loss    :", round(float(loss_cnn), 4))

plot_history(hist_cnn, "CNN (Best)")


improvement = (acc_cnn - acc_ann) * 100
print("\n==============================")
print("FINAL COMPARISON")
print("==============================")
print("ANN Accuracy :", round(float(acc_ann) * 100, 2), "%")
print("CNN Accuracy :", round(float(acc_cnn) * 100, 2), "%")
print("Improvement  :", round(float(improvement), 2), "%")
