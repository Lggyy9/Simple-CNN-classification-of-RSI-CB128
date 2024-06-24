# 简单卷积神经网络对RSI-CB128的分类
利用简单CNN对RSI-CB128中45个子类进行分类 代码包含数据集划分、模型训练、模型验证三个过程  
用到的简单CNN网络结构可视化如下：  
![模型可视化](https://github.com/Lggyy9/Simple-CNN-classification-of-RSI-CB128/assets/135825262/44961654-08bf-42c0-8fba-1dac4d49a72e)  
数据集源：https://github.com/lehaifeng/RSI-CB  
**添加使用指南和简单说明**  
## validate.py说明
* load_model(model_path)函数：加载预训练好的模型参数
* validate_model(model, dataloader)函数:前向传播计算预测结果，比较标签统计正确预测的数量，计算并打印模型在验证集上的准确率
  >  由于此文件用于模型的最后一步“验证”，无需被其它文件调用因此没有编写main函数对主代码保护 直接运行即可

# Simple-CNN-classification-of-RSI-CB128
This project uses a simple CNN to classify 45 subclasses in RSI-CB128. The code includes three processes: data set division, model training, and model validation.  
The simple CNN network structure used is visualized as shown in the picture in the Chinese part.  
You can find the original dataset in:https://github.com/lehaifeng/RSI-CB  

