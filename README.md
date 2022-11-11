# GP-GNN test

## 转换CMeIE数据
```bash
python3 convert.py
```



## 生成Bert向量
```bash
python3 bert_features.py
```



## 训练 NER
```bash
cd ner
python3 convert_cmeie.py
python3 baseline_train.py
```



## 训练 图网络
```bash
python3 train2.py
```



## 生成结果
```bash
cd ner; echo "import baseline_train" | python3; cd ..
python3 conver_test.py
python3 test2.py
```



## 测试结果

|       |   F1   |   P    |   R    |
| :---: | :----: | :----: | :----: |
| CMeIE | 36.365 | 32.722 | 40.920 |

