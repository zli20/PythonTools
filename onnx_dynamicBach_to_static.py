# -----------------------------------------
# --------修改模型动态batch改为静态------------
# -----------------------------------------

import onnx
from onnx import helper


def convert_dynamic_to_static_batch(model_path, input_shapes):
    # 加载模型
    model = onnx.load(model_path)
    # 假设我们要将第一个输入的批处理大小改为 1
    # 获取第一个输入的形状（假设第一个输入是我们要修改的输入）
    input_shape = model.graph.input[0].type.tensor_type.shape
    # 更改批处理大小（假设批处理大小是第一个维度）
    input_shape.dim[0].dim_value = input_shapes[0]
    input_shape.dim[1].dim_value = input_shapes[1]
    input_shape.dim[2].dim_value = input_shapes[2]
    input_shape.dim[3].dim_value = input_shapes[3]
    # 保存修改后的模型
    onnx.save(model, "static_model.onnx")


def convert_dynamic_to_static_layer(model_path, static_input_shape):
    # 加载模型
    model = onnx.load(model_path)

    # 更新模型中所有层的输入形状
    for node in model.graph.node:
        # 遍历节点的输入
        for i, input_name in enumerate(node.input):
            for input_tensor in model.graph.input:
                if input_tensor.name == input_name:
                    # 更新输入形状
                    input_tensor.type.tensor_type.shape.ClearField("dim")
                    for dim_size in static_input_shape:
                        dim = input_tensor.type.tensor_type.shape.dim.add()
                        dim.dim_value = dim_size

                    break

    # 保存修改后的模型
    onnx.save(model, "static_model.onnx")


if __name__ == '__main__':
    m_path = ""
