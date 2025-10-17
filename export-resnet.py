import sys
from pathlib import Path

sys.path.append(str(Path(".") / "tensorflow2/tf2cv/models"))

import tensorflow as tf
from resnet import resnet200b

model = resnet200b()

try:
    # for keras
    tf.saved_model.save(model, "./export_model")

    # for tf.Model
    # tf.saved_model.save(
    #     model,
    #     export_dir: "./export_model",
    #     signatures=None,
    # )
    print("[JIT] Export model by TensorFlow successed.")
except:
    print("[JIT] Export model by TensorFlow failed.")
