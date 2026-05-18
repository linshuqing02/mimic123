# Processed EHR Dataset 


## 📖 简介 (Introduction)
本数据集是一个经过预处理的电子健康记录（EHR）样本数据集，保存为 Python 的 `pickle` 格式。它聚合了患者的就诊记录（Visits），包含了疾病诊断（Conditions）、医疗操作（Procedures）、处方药物（Drugs，采用 ATC 编码）以及实验室检查项目（Lab Items）。该数据结构非常适合用于医疗深度学习模型的训练（如：疾病预测、药物推荐、住院时间预测等）。

## 🗂 数据结构 (Data Structure)

读取该 `.pkl` 文件后，将获得一个包含医疗特征数据的对象（或字典结构）。主要由以下两部分组成：

- `samples`: `List[Dict]`，主要的数据载体。列表中的每一个字典代表一次患者的就诊记录。
- `patient_to_index`: `Dict[str, List[int]]`，辅助索引字典。键为 `patient_id`，值为该患者在 `samples` 列表中的索引（Index）列表，用于快速实现患者维度的时间序列查询。

### `samples` 字段详情

单条就诊记录包含以下键值：

| 字段名 | 数据类型 | 描述说明 | 示例值 |
| :--- | :--- | :--- | :--- |
| `visit_id` | `str` / `int` | 单次就诊/住院的唯一标识 ID | `182383` |
| `patient_id` | `str` / `int` | 患者的唯一标识 ID | `107` |
| `conditions` | `List[List[str]]` | 疾病诊断代码（如 ICD 编码） | `[['237', '158', ...]]` |
| `procedures` | `List[List[str]]` | 手术或医疗操作代码 | `[['58']]` |
| `drugs` | `List[str]` | 本次就诊开具的药物代码（ATC 标准） | `['B05X', 'V03A', ...]` |
| `drugs_hist` | `List[List[str]]` | 患者历史用药记录序列 | `[]` |
| `lab_item` | `List[List[str]]` | 实验室检查/化验项目代码（如 MIMIC ItemID）| `[['50809', '50813', ...]]` |

---

## 🎯 常见下游任务与评价指标 (Tasks & Evaluation Metrics)

针对此类包含诊断、操作和药物的纵向医疗数据，常见的深度学习任务及对应的标准评价指标如下：

### 1. 药物推荐 (Medication Recommendation)
**任务描述**：根据患者的当前诊断、过往病史和手术操作，推荐本次就诊应该开具的药物组合（多标签分类任务）。
**标准评价指标**：
- **Jaccard Similarity (Jaccard 相似度)**: 衡量预测药物集合与真实药物集合的重合度（最重要的核心指标）。
- **PRAUC (PR 曲线下面积)**: 衡量模型在类别不平衡情况下的综合性能。
- **F1-Score (Micro / Macro)**: 衡量精确率和召回率的调和平均值。
- **DDI Rate (药物相互作用率)**: 衡量推荐药物组合中存在的禁忌药物相互作用比例（越低越好，用于评估模型的安全性）。

### 2. 疾病/再入院预测 (Disease / Readmission Prediction)
**任务描述**：根据患者的历史就诊序列（诊断、药物等），预测患者在未来一段时间内是否会患某种特定疾病，或是否会再次入院（二分类或多分类任务）。
**标准评价指标**：
- **AUROC (ROC 曲线下面积)**: 整体预测能力的评估指标。
- **AUPRC**: 对于阳性样本较少（即不平衡数据）的预测更具参考价值。
- **Accuracy / Recall / Precision**: 基础的二分类/多分类指标。

---

## 🏆 标准基线模型 (Standard Baselines)

你可以使用以下经典的 Baseline 模型在当前数据集上进行跑分对比：

1. **传统机器学习方法**: 
   - **LR (Logistic Regression) / RF (Random Forest)**: 将历史序列特征展平（Flatten）或求和后进行预测，作为最基础的对照。
2. **时序深度学习模型**:
   - **LSTM / GRU**: 最常用的序列建模基线，将每次就诊作为一个 Time Step 依次输入。
   - **RETAIN**: 经典的医疗可解释性时序模型，基于双向注意力机制，能解释是哪次就诊或哪个具体诊断导致了最终的预测结果。
3. **先进药物推荐模型** (如果核心任务是药物推荐):
   - **LEAP**: 基于强化学习或序列到序列 (Seq2Seq) 的药物推荐模型。
   - **GAMENet**: 结合图神经网络（GNN）和药物相互作用（DDI）知识图谱的经典安全药物推荐模型。
   - **SafeDrug**: 利用分子结构和二分图提升药物推荐准确性与安全性的 SOTA 模型。
4. **预训练模型**:
   - **Transformer / Med-BERT**: 将诊断和药物编码视作 NLP 中的 Token，利用自注意力机制（Self-Attention）进行建模，捕获复杂的医疗概念关联。

---

## 🚀 如何加载和使用 (Usage Tutorial)

以下代码展示了如何使用 Python 加载数据，并将其封装为适合深度学习框架（如 **PyTorch**）使用的数据迭代器。

```python
import pickle

file_path = 'processed_developer_data.pkl'

with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

# 查看第一位患者的所有就诊记录
first_patient_id = list(dataset.patient_to_index.keys())[0]
visit_indices = dataset.patient_to_index[first_patient_id]
patient_visits = [dataset.samples[i] for i in visit_indices]

print(f"患者 {first_patient_id} 共有 {len(patient_visits)} 次就诊记录。")

### PyTorch Dataset 封装示例 (以药物推荐为例)
import torch
from torch.utils.data import Dataset, DataLoader

class EHRDataset(Dataset):
    def __init__(self, ehr_data):
        # 假设传入的是加载好的 dataset 对象
        self.samples = ehr_data.samples
        
        # 在实际训练中，你需要在这里构建 vocab (词表) 
        # 将条件代码 (如 '237') 和药物代码 (如 'B05X') 映射为唯一的整数 ID
        self.vocab = self._build_mock_vocab()

    def _build_mock_vocab(self):
        # 此处省略具体词表构建逻辑，需自行根据数据集所有出现过的代码进行映射
        return {'condition2id': {}, 'drug2id': {}}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        visit = self.samples[idx]
        
        # 提取当前就诊的诊断 (展平嵌套列表)
        conditions = [c for sublist in visit['conditions'] for c in sublist]
        # 提取当前就诊的药物 (Labels)
        drugs = visit['drugs']
        
        # 将代码转化为数值 ID (伪代码)
        # condition_ids = [self.vocab['condition2id'][c] for c in conditions]
        # label_ids = [self.vocab['drug2id'][d] for d in drugs]
        
        return {
            'visit_id': visit['visit_id'],
            'conditions': conditions, # 模型输入 Input
            'drugs': drugs            # 模型目标标签 Target
        }

# 使用 DataLoader
my_dataset = EHRDataset(dataset)
dataloader = DataLoader(my_dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    # 开始你的训练流程
    # outputs = model(batch['conditions'])
    # loss = criterion(outputs, batch['drugs'])
    break
