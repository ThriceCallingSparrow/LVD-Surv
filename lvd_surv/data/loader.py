"""data 模块：提供项目内部的明确、可复用实现。"""
import pandas as pd
import numpy as np
import os

class CMAPSSLoader:
    '''
    许丹, 肖小琦, 冯至昕等 . 未知载荷条件下机械系统剩余寿命预测方法[J]. 北京航空航天大学学报, 2022, 48(3): 376-383. doi: 10.13700/j.bh.1001-5965.2020.0582
    '''
    _CONDITION_CENTERS = np.array([
        [42.0, 0.84, 100.0],  # 工况1：高空高速巡航（最大推力状态）
        [10.0, 0.25, 100.0],  # 工况2：低空低速巡航
        [25.0, 0.62, 60.0],   # 工况3：中空中速巡航（部分推力状态）
        [20.0, 0.70, 100.0],  # 工况4：中高空亚音速巡航
        [35.0, 0.84, 100.0],  # 工况5：高空亚音速巡航
        [0.0, 0.00, 100.0]    # 工况6：地面怠速（起飞前/降落后状态）
    ])
    
    def __init__(self, root_path):
        """
        root_path: CMAPSSData folder path
        """
        self.root_path = root_path

    def _get_file_path(self, data_type, fd):
        """得到数据路径"""
        filename = f"{data_type}_{fd}.txt"
        return os.path.join(self.root_path, filename)

    def _load_txt(self, file_path):
        '''加载文件'''
        df = pd.read_csv(file_path, sep=r"\s+", header=None)
        df = df.dropna(axis=1, how='all')
        return df
    
    def _add_columns(self, df):
        '''添加头'''
        columns = ['unit_id', 'cycle']
        columns += [f'op_{i}' for i in range(1, 4)]
        columns += [f'sensor_{i}' for i in range(1, 22)]
        df.columns = columns
        return df
    
    def _label_conditions(self, df):
        """欧氏距离匹配法自动识别并标记每个样本的工况"""
        # 提取三个操作参数
        ops_data = df[['op_1', 'op_2', 'op_3']].values
        
        # 计算每个样本到 6 个工况中心点的欧氏距离
        distances = np.sqrt(
            np.sum((ops_data[:, np.newaxis] - self._CONDITION_CENTERS) ** 2, axis=2)
        )
        
        # 找到距离最近的工况编号
        df['condition'] = np.argmin(distances, axis=1) + 1
        
        return df
    
    def _compute_rul(self, df):
        '''计算当前时刻剩余寿命'''
        max_cycle = df.groupby('unit_id')['cycle'].max().reset_index()
        max_cycle.columns = ['unit_id', 'max_cycle']
        df = df.merge(max_cycle, on='unit_id')
        df['RUL'] = df['max_cycle'] - df['cycle']
        return df.drop('max_cycle', axis=1)
    
    def load_train(self, fd):
        '''加载训练集'''
        path = self._get_file_path("train", fd)
        df = self._load_txt(path)
        
        # 添加列索引
        df = self._add_columns(df)
        # 添加 conditions 列
        df = self._label_conditions(df)
        # 添加 rul 列
        df = self._compute_rul(df)
        
        return df
    
    def load_test(self, fd):
        '''加载测试集'''
        path = self._get_file_path("test", fd)
        df = self._load_txt(path)
        
        # 添加列索引
        df = self._add_columns(df)
        # 添加 conditions 列
        df = self._label_conditions(df)
        
        return df
    
    def load_rul(self, fd):
        '''加载测试集的真实寿命周期'''
        path = self._get_file_path("RUL", fd)
        df = self._load_txt(path)
        df.columns = ['RUL']
        return df
    
    def load_all(self, fd):
        """
        一次性加载 train + test + RUL
        """
        train = self.load_train(fd)
        test = self.load_test(fd)
        rul = self.load_rul(fd)
        
        return train, test, rul


def load_cmapss_bundle(root_path, dataset):
    """加载 C-MAPSS train/test/RUL 三类数据"""
    from lvd_surv.data.cmapss import normalize_cmapss_schema
    loader = CMAPSSLoader(str(root_path))
    train_df, test_df, rul_df = loader.load_all(str(dataset))
    return {
        "train_df": normalize_cmapss_schema(train_df, add_condition_from_ops=True),
        "test_df": normalize_cmapss_schema(test_df, add_condition_from_ops=True) if test_df is not None else None,
        "rul_df": rul_df,
        "dataset": str(dataset),
    }


def load_cmapss_training_data(root_path, dataset):
    """只返回训练集，供轻量测试和旧式调用使用"""
    return load_cmapss_bundle(root_path, dataset)["train_df"]
