import tf2onnx

tf2onnx.convert.from_saved_model(
    saved_model_dir="saved_model",
    output_path="handwriting_model.onnx",
    opset=13
)
