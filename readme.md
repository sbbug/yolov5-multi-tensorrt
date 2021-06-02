

## 边缘设备模型量化部署

#### 硬件：英伟达JetSon AGX Xavier
#### 模型：YOLOv5

#### 步骤：

###### 1、模型量化 .onnx --> .trt 
          export_tensorrt.py

###### 2、模型推理  输入图像，输入结果
         ./lib/demo_test.py