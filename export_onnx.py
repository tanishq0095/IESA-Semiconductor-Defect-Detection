import tensorflow as tf
import tf2onnx
import os

# 1. Load the model you just trained
model = tf.keras.models.load_model("temp_model.keras")

# 2. Use the most stable conversion method
print("Converting to ONNX... Final step for Phase 1!")
spec = (tf.TensorSpec((None, 224, 224, 1), tf.float32, name="input"),)
output_path = "semiconductor_defect_model.onnx"

model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

with open(output_path, "wb") as f:
    f.write(model_proto)

print(f"DONE! File created: {os.path.abspath(output_path)}")