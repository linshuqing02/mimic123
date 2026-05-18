# Processed EHR Dataset 

## 📖 简介 (Introduction)
本数据集是一个经过预处理的电子健康记录（EHR）样本数据集，保存为 Python 的 `pickle` 格式。它聚合了患者的就诊记录（Visits），包含了疾病诊断（Conditions）、医疗操作（Procedures）、处方药物（Drugs，采用 ATC 编码）以及实验室检查项目（Lab Items）。该数据结构非常适合用于医疗深度学习模型的训练（如：疾病预测、药物推荐、住院时间预测等）。

## 🗂 数据结构 (Data Structure)

读取该 `.pkl` 文件后，将获得一个包含医疗特征数据的对象（`SampleEHRDataset`）。其主要由以下两个核心部分组成：

- `samples`: `List[Dict]`，主要的数据载体。列表中的每一个字典代表一次患者的就诊记录。
- `patient_to_index`: `Dict[str, List[int]]`，辅助索引字典。键为 `patient_id`，值为该患者在 `samples` 列表中的索引（Index）列表，用于快速实现患者维度的序列查询。

### `samples` 字段详情

单条就诊记录（Sample Dict）包含以下键值：

| 字段名 | 数据类型 | 描述说明 | 示例值 |
| :--- | :--- | :--- | :--- |
| `visit_id` | `str` / `int` | 单次就诊/住院的唯一标识 ID | `182383` |
| `patient_id` | `str` / `int` | 患者的唯一标识 ID | `107` |
| `conditions` | `List[List[str]]` | 疾病诊断代码（如 ICD 编码） | `[['237', '158', '108', ...]]` |
| `procedures` | `List[List[str]]` | 手术或医疗操作代码 | `[['58']]` |
| `drugs` | `List[str]` | 本次就诊开具的药物代码（ATC 标准） | `['B05X', 'V03A', 'N02B', ...]` |
| `drugs_hist` | `List[List[str]]` | 患者历史用药记录序列 | `[]` |
| `lab_item` | `List[List[str]]` | 实验室检查/化验项目代码（如 MIMIC ItemID）| `[['50809', '50813', '50822', ...]]` |

## 🚀 如何加载和使用 (Usage)

你可以使用 Python 标准库中的 `pickle` 模块来加载和查看该数据集：

```python
import pickle

# 定义文件路径
file_path = 'processed_developer_data.pkl'

# 如果该文件是自定义类的实例，建议确保当前命名空间有对应的类定义
# 或者在读取遇到 ModuleNotFoundError 时构建 mock class

with open(file_path, 'rb') as f:
    dataset = pickle.load(f)

# 查看数据基本信息
# 如果 dataset 是类对象：
print(f"总就诊记录数: {len(dataset.samples)}")
print(f"第一条记录信息: {dataset.samples[0]}")

# 如果 dataset 解析为字典形式：
# print(len(dataset['samples']))
