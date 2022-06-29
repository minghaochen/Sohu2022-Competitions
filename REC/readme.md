## 目录结构

```
├── code
│   ├── feats
│   ├── model
│   ├── notebook
│   │   ├── 1_preproccess.py
│   │   ├── 2_time_feats.py
│   │   ├── 3_static_feats.py
│   │   ├── 4_pos_gap_feats.py
│   │   ├── 5_tfidf_feats.py
│   │   ├── 6_w2v.py
│   │   ├── 7_click_feats.py
│   │   ├── 8_emotion_feats.py
│   │   ├── dataset.py
│   │   ├── main.py
│   │   └── utils.py
│   ├── result
│   └── tmp
├── data
│   └── Sohu2022_data
│       └── rec_data
│           ├── recommend_content_entity_0317.txt
│           ├── recommend_content_entity_B.txt
│           ├── recommend_content_entity_sup.txt
│           ├── test-dataset_A.csv
│           ├── test-dataset_B.csv
│           ├── train-dataset_A.csv
│           └── train-dataset_sup.csv
└── readme.md
```

## 特征说明

+ 1_preproccess.py

  主要利用5号和6号的数据，恢复部分A榜测试集，4号的数据

+ 2_time_feats.py

  时间方面的一些特征

+ 3_static_feats.py

  一些统计特征，包括count、nunique、click等

+ 4_pos_gap_feats.py

  pos、gap、itmeId的序列特征

+ 5_tfidf_feats.py

  用户pvId、suv的tfidf embedding

+ 6_w2v.py

  w2v模型的训练

+ 7_click_feats.py

  点击率特征、包括历史和全局的

+ 8_emotion_feats.py

  情感特征

+ dataset.py

  产生训练数据

+ main.py

  训练并生成结果

## 模型

每折模型250M，由于大小限制，只使用前8折

## 运行说明

1-8为特征文件，先运行dataset.py 生成训练数据集，在运行main.py

## 其他说明

数据文件仅提供占位符的作用

## 百度云

链接: https://pan.baidu.com/s/11Dy8ruESaC8xU9eb7rxJbQ 提取码: p9jj 
