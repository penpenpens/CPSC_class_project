## READ ME

### 训练：

**method 1:**

运行*Train.ipynb*，更改data_path为训练集目录，文件中更改参数lead=0和lead=1各训练一次.

再运行*extraTrain.ipynb*用阵发性房颤数据集额外训练，data_path为训练集目录，lead=0和lead=1各训练一次.。

**method 2:**

```python
python Train.py   ##更改data_path为训练集目录，文件中更改参数lead=0和lead=1各训练一次.
```

```python
python extraTrain.py   ##更改data_path为训练集目录，文件中更改参数lead=0和lead=1各训练一次.
```

### 预测：

**普通双模型：**

```python
python entry_2021.py [DATA_PATH] [RESULT_PATH] 
```

**小波变换+LSTM模型用于分类：**

```python
python entry_2021_wave.py [DATA_PATH] [RESULT_PATH] 
```

用train_set和test_set各测试一次。

### 计算得分：

**method 1:**

```python{
python score_2021.py [TESTSET_PATH] [RESULT_PATH]  ##仅得到最终得分，注意更改
```

**method 2:**

```python{
python score1_2021.py [TESTSET_PATH] [RESULT_PATH] ##得到分类错误的数据和错误类型，错分类矩阵，两种得分以及最终得分
```

**method 3:**

运行*score.ipynb*，需更改TESTSET_PATH和RESULT_PATH，结果同method2。