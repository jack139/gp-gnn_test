# GP-GNN test

## 转换CMeIE数据
```
python3 convert.py
```



## 生成Bert向量
```
python3 bert_features.py
```



## 训练
```
python3 train2.py
```



## 生成结果
```
cd ner; echo "import baseline_train" | python3; cd ..
python3 conver_test.py

```