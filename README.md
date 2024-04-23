# PythonTools
一些常用的工具

`huggingface_hub_download.py `  
从huggingface 下载模型，没有vpn，需要使用镜像；  

`onnx_dynamicBach_to_static.py`  
部分推理框架不支持动态输入，将onnx模型的动态输入转化为静态；  

`onnx_opset_converter.py`  
为了适配推理框架，更改onnx模型的op版本；  

`onnx_squeeze_to_reshape.py`  
snpe2.10 在模型转换时，squeeze层存在bug，将squeeze转换成reshpe层。  