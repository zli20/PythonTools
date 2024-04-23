# -------------------------------------
# --------修改onnx模型的op版本------------
# -------------------------------------

import onnx
from onnx import version_converter, helper

# 加载模型
model_path = "static_model.onnx"
original_model = onnx.load(model_path)

# 获取模型的opset_import信息
opset_imports = original_model.opset_import

# 输出模型的op版本信息
print("Model opset versions:")
for opset in opset_imports:
    print("Domain: {}, Version: {}".format(opset.domain, opset.version))

onnx.checker.check_model(original_model)

# 将模型转换为需要的版本
target_opset_version = 11

converted_model = version_converter.convert_version(original_model, target_opset_version)

# 保存转换后的模型
output_model_path = 'model_v11.onnx'
onnx.save(converted_model, output_model_path)
