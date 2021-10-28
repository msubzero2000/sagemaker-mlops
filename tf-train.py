import tensorflow as tf
import numpy as np
import os
import argparse
import tarfile
import tempfile


def tardir(path, tar_name):
    with tarfile.open(tar_name, "w:gz") as tar_handle:
        for root, dirs, files in os.walk(path):
            for file in files:
                tar_handle.add(os.path.join(root, file))


parser = argparse.ArgumentParser()

parser.add_argument('--model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
parser.add_argument('--train_dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
parser.add_argument('--eval_dir', type=str, default=os.environ['SM_CHANNEL_EVAL'])
parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])
parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])

args = parser.parse_args()

print(f"Model directory at {args.model_dir}")
print(f"Training directory at {args.train_dir}")
print(f"Eval directory at {args.eval_dir}")
print(f"Output data directory at {args.output_data_dir}")
print(f"Num GPUS {args.num_gpus}")

mnist = tf.keras.datasets.mnist

x_train = np.load(f"{args.train_dir}/x_train.dat.npy")
y_train = np.load(f"{args.train_dir}/y_train.dat.npy")

x_test = np.load(f"{args.eval_dir}/x_test.dat.npy")
y_test = np.load(f"{args.eval_dir}/y_test.dat.npy")

x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation='relu'),
tf.keras.layers.Dropout(0.2),
tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
loss='sparse_categorical_crossentropy',
metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
with tempfile.TemporaryDirectory() as tmpdir:
    tf.saved_model.save(model, f"/opt/ml/model/1/")

model.evaluate(x_test, y_test)