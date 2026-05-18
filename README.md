# 数据格式说明
这是一个用于 电子健康记录（EHR）深度学习任务 的预处理数据集，主要面向 药物推荐 等临床预测问题。数据来源于 MIMIC 数据库，经过清洗、编码和序列化处理，存储为 Python Pickle 格式。
一、整体结构
加载后得到一个 SampleEHRDataset 对象，核心结构如下：
├── samples: List[dict]         # 样本列表，每个样本代表一次就诊
├── dataset_name: str            # 数据集名称
├── task_name: str               # 任务类型
├── code_vocs: dict              # 各类临床事件的编码映射
├── input_info: dict             # 输入特征的元信息
├── patient_to_index: dict       # 患者ID → 样本索引
└── visit_to_index: dict         # 就诊ID → 样本索引
