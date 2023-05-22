import pandas as pd
import tqdm
from tensorflow.keras.models import Model
from layers import DNN

def MIND(user_feature_columns, item_feature_columns, num_sampled=5, k_max=2, p=1.0, dynamic_k=False, 
         user_dnn_hidden_units=(64, 32), dnn_activation='relu', dnn_use_bn=False, l2_reg_dnn=0, 
         l2_reg_embedding=1e-6, dnn_dropout=0, output_activation='linear', seed=1024):
    """
    MIND采用动态路由算法将历史商品聚成多个集合,每个集合的历史行为进一步推断对应特定兴趣的用户表示向量.
    对于一个特定的用户,MND输出多个表示向量代表用户的不同兴趣;当用户再有新的交互时,通过胶囊网络实时的改变用户的兴趣表示向量
    做到在召回阶段的实时个性化
    """
    pass