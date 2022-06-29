# NLP赛题

建模思路：

Prompt+固定模板：再这句话中，{实体}是[MASK]

训练思路：

- 针对样本不平衡问题，使用了标签平滑损失函数
- 每个 mini-batch 中，每个数据样本过两次带有 Dropout 的同一个模型，再使用 KL-divergence 约束两次的输出一致
- 迭代训练，基于第一次训练的模型对测试集进行预测，然后取置信度较高的样本重新加入训练过程

后处理：

- 针对评价指标f1-score进行预测权重的优化

## 执行说明

1. 根据 生成复赛训练数据.ipynb 得到训练数据
2. 根据 train_nlp_rdrop_final.py 进行第一轮模型训练
3. 根据 test_nlp_for_pseudo.py 进行测试集预测
4. 根据 train_nlp_rdrop_pseudo_final.py 进行第二轮模型训练
5. 根据 train_nlp_f1_opt.py 进行预测权重的优化
6. 根据 test_nlp_f1.py 得到测试集最终预测结果

## 网盘

链接：https://pan.baidu.com/s/1n0IUQJjF_W00Gpd7dZBEDA 
提取码：123a

# 推荐赛题

建模思路：DeepFM + DIN序列建模

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

- 1_preproccess.py

  主要利用5号和6号的数据，恢复部分A榜测试集，4号的数据

- 2_time_feats.py

  时间方面的一些特征

- 3_static_feats.py

  一些统计特征，包括count、nunique、click等

- 4_pos_gap_feats.py

  pos、gap、itmeId的序列特征

- 5_tfidf_feats.py

  用户pvId、suv的tfidf embedding

- 6_w2v.py

  w2v模型的训练

- 7_click_feats.py

  点击率特征、包括历史和全局的

- 8_emotion_feats.py

  情感特征

- dataset.py

  产生训练数据

- main.py

  训练并生成结果

## 模型

每折模型250M，由于大小限制，只使用前8折

## 运行说明

1-8为特征文件，先运行dataset.py 生成训练数据集，在运行main.py

## 其他说明

数据文件仅提供占位符的作用

## 百度云

链接: https://pan.baidu.com/s/11Dy8ruESaC8xU9eb7rxJbQ 提取码: p9jj 