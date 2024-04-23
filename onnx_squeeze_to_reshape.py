# snpe的squeeze存在bug，所以想将onnx中的squeeze替换为reshape

import onnx
from onnx import helper, shape_inference
import numpy as np

def trans():
    # 加载ONNX模型
    model_path = "rtmpose_t_256_192.onnx"
    model = onnx.load(model_path)

    new_nodes = []
    squeeze_node = ['Squeeze_159', 'Squeeze_160']
    for node_name in squeeze_node:
        for node in model.graph.node:
            if node.op_type == 'Squeeze' and node.name == node_name:
                squeeze_node = node
                break

        if squeeze_node is None:
            raise ValueError("Squeeze node not found")

        # 获取Squeeze节点的输入和输出
        input_name = squeeze_node.input[0]
        output_name = squeeze_node.output[0]
        name = "Reshape" + node_name[-4:]
        original_shape = [1, 17, 1, 128]
        new_shape = [1, 17, 128]

        # 创建一个常量节点来表示新的形状
        new_shape_tensor = helper.make_tensor(
            name='new_shape',
            data_type=onnx.TensorProto.INT64,
            dims=[len(new_shape)],
            vals=new_shape,  # 这里vals是临时的，因为我们稍后会用make_node创建常量
        )
        new_shape_node = helper.make_node(
            'Constant',
            inputs=[],
            outputs=['new_shape_output'+name],
            value=new_shape_tensor,
        )

        # 创建新的Reshape节点
        reshape_node = helper.make_node(
            'Reshape',
            inputs=[input_name, 'new_shape_output'+name],
            outputs=[output_name],
            name=name,
        )

        # 将新节点添加到模型中
        # 首先，添加Constant节点
        model.graph.node.insert(0, new_shape_node)  # 插入到图的开始位置

        # 然后，替换Squeeze节点为Reshape节点
        index = model.graph.node.index(squeeze_node)
        model.graph.node.insert(index, reshape_node)
        model.graph.node.pop(index + 1)  # 移除旧的Squeeze节点

        # # 将新节点添加到图中
        # model.graph.node.extend([new_shape_node, reshape_node])
        #将新节点添加到列表中
        # new_nodes.extend([new_shape_node, reshape_node])
        # model.graph.node.append(new_shape_node)
        # model.graph.node.append(reshape_node)
        # 从图中删除旧的Squeeze节点
        # model.graph.node.remove(squeeze_node)
        print('trans', node_name, 'to', name)

    # 扩展图的节点列表
    # model.graph.node.extend(new_nodes)
    model = shape_inference.infer_shapes(model)

    onnx.save(model, "rtmpose_t_256_192_reshape.onnx")


def test():
    model_path = "rtmpose_t_256_192_reshape.onnx"

    import onnxruntime as ort
    ort_session = ort.InferenceSession(model_path)
    input_data = np.random.randn(1, 3, 256, 192).astype(np.float32)
    output = ort_session.run(None, {'input': input_data})

    from onnx import checker
    model = onnx.load(model_path)
    # 检查模型结构
    checker.check_model(model)

    print(output)


if __name__ == '__main__':
    trans()
    test()