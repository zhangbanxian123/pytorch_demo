# pytorch_demo

根据斯坦福大学一个pytorch项目修改的图像分类模板
## 库安装
pip install -r requirements.txt

## 任务
给定手做代表0、1、2、3、4或5的符号的图像，则预测正确的标签。

## 下载SIGNS数据集
点击下载[百度云](https://pan.baidu.com/s/1IVCPVKElIcXJK7RNT2VITQ).
提取码：jkhf 
将数据集放置在dataset目录下，目录结构如：
```
SIGNS/
    train_signs/
        0_IMG_5864.jpg
        ...
    test_signs/
        0_IMG_5942.jpg
        ...
```
## 开始训练
python train.py
## 测试模型
python evaluate.py
若要使用预训练模型，可以从[这里下载](https://pan.baidu.com/s/1IVCPVKElIcXJK7RNT2VITQ) 
并将它放在checkpoint文件夹下
