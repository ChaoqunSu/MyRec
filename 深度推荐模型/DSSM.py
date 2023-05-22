# DSSM召回
"""
user侧主要包含一些户画像属性(用户性别,年龄,所在省市,使用设备及系统)
item侧主要包括创建时间,题目,级别,关键词等
"""

import pandas as pd
import datetime
import multiprocessing
import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Concatenate
from features import FeatureEncoder
from layers import DNN, CosinSimilarity, PredictLayer

# 预处理
def proccess(file):
    if file=="user_info_data_5w.csv":
        data = pd.read_csv(file_path + file, sep="\t",index_col=0)
        data["age"] = data["age"].map(lambda x: get_pro_age(x))
        data["gender"] = data["gender"].map(lambda x: get_pro_age(x))

        data["province"]=data["province"].fillna(method='ffill')
        data["city"]=data["city"].fillna(method='ffill')

        data["device"] = data["device"].fillna(method='ffill')
        data["os"] = data["os"].fillna(method='ffill')
        return data

    elif file=="doc_info.txt":
        data = pd.read_csv(file_path + file, sep="\t")
        data.columns = ["article_id", "title", "ctime", "img_num","cate","sub_cate", "key_words"]
        select_column = ["article_id", "title_len", "ctime", "img_num","cate","sub_cate", "key_words"]

        # 去除时间为nan的新闻以及除脏数据
        data= data[(data["ctime"].notna()) & (data["ctime"] != 'Android')]
        data['ctime'] = data['ctime'].astype('str')
        data['ctime'] = data['ctime'].apply(lambda x: int(x[:10]))
        data['ctime'] = pd.to_datetime(data['ctime'], unit='s', errors='coerce')


        # 这里存在nan字符串和异常数据
        data["sub_cate"] = data["sub_cate"].astype(str)
        data["sub_cate"] = data["sub_cate"].apply(lambda x: pro_sub_cate(x))
        data["img_num"] = data["img_num"].astype(str)
        data["img_num"] = data["img_num"].apply(photoNums)
        data["title_len"] = data["title"].apply(lambda x: len(x) if isinstance(x, str) else 0)
        data["cate"] = data["cate"].fillna('其他')

        return data[select_column]


def process_feature(user_feature_columns, item_feature_columns, feature_encode):
    """
    根据FeatureEncoder获取所有输入的Input层或者Embedding层,然后根据实际场景的业务数据，对不同的特征进行处理.
    """
    group_embedding_dict = feature_encode.sparse_feature_dict

    user_emb_name = [fc.embedding_name for fc in user_feature_columns]
    item_emb_name = [fc.embedding_name for fc in item_feature_columns]

    user_dnn_input = [v for k, v in group_embedding_dict['default_group'].items() 
        if k in user_emb_name]
    item_dnn_input = [v for k, v in group_embedding_dict['default_group'].items() 
        if k in item_emb_name]

    return user_dnn_input, item_dnn_input


# 构造样本,交互日志中前6天作为训练集,第七天为测试集
def dealsample(file, doc_data, user_data, s_data_str = "2021-06-24 00:00:00", e_data_str="2021-06-30 23:59:59", neg_num=5):
    # 先处理时间问题
    data = pd.read_csv(file_path + file, sep="\t",index_col=0)
    data['expo_time'] = data['expo_time'].astype('str')
    data['expo_time'] = data['expo_time'].apply(lambda x: int(x[:10]))
    data['expo_time'] = pd.to_datetime(data['expo_time'], unit='s', errors='coerce')

    s_date = datetime.datetime.strptime(s_data_str,"%Y-%m-%d %H:%M:%S")
    e_date = datetime.datetime.strptime(e_data_str,"%Y-%m-%d %H:%M:%S") + datetime.timedelta(days=-1)
    t_date = datetime.datetime.strptime(e_data_str,"%Y-%m-%d %H:%M:%S")

    # 选取训练和测试所需的数据
    all_data_tmp = data[(data["expo_time"]>=s_date) & (data["expo_time"]<=t_date)]

    # 处理训练数据集  防止穿越样本
    # merge item信息，得到曝光时间和item创建时间; inner join 去除doc_data之外的item
    all_data_tmp = all_data_tmp.join(doc_data.set_index("article_id"),on="article_id",how='inner')

    # 除去在时间上不符合业务逻辑的数据
    all_data_tmp = all_data_tmp[(all_data_tmp["ctime"]<=all_data_tmp["expo_time"])]
    train_data = all_data_tmp[(all_data_tmp["expo_time"]>=s_date) & (all_data_tmp["expo_time"]<=e_date)]
    train_data = train_data[(train_data["ctime"]<=e_date)]

    print("有效的样本数：",train_data["expo_time"].count())

    # 负采样
    if os.path.exists(file_path + "neg_sample.pkl") and os.path.getsize(file_path + "neg_sample.pkl"):
        neg_samples = pd.read_pickle(file_path + "neg_sample.pkl")
    else:
        # 进行负采样的时候对于样本进行限制，只对一定时间范围之内的样本进行负采样
        doc_data_tmp = doc_data[(doc_data["ctime"]>=datetime.datetime.strptime("2021-06-01 00:00:00","%Y-%m-%d %H:%M:%S"))]
        neg_samples = negSample_like_word2vec(train_data, doc_data_tmp[["article_id"]].values, user_data[["user_id"]].values, neg_num=neg_num)
        neg_samples = pd.DataFrame(neg_samples, columns= ["user_id","article_id","click"])
        neg_samples.to_pickle(file_path + "neg_sample.pkl")

    train_pos_samples = train_data[train_data["click"] == 1][["user_id","article_id", "expo_time", "click"]]    # 取正样本

    neg_samples_df = train_data[train_data["click"] == 0][["user_id","article_id", "click"]]
    train_neg_samples = pd.concat([neg_samples_df.sample(n=train_pos_samples["click"].count()) ,neg_samples],axis=0)  # 取负样本

    print("训练集正样本数：",train_pos_samples["click"].count())
    print("训练集负样本数：",train_neg_samples["click"].count())

    train_data_df = pd.concat([train_neg_samples,train_pos_samples],axis=0)
    train_data_df = train_data_df.sample(frac=1)  # shuffle

    print("训练集总样本数：",train_data_df["click"].count())

    test_data_df =  all_data_tmp[(all_data_tmp["expo_time"]>e_date) & (all_data_tmp["expo_time"]<=t_date)][["user_id","article_id", "expo_time", "click"]]

    print("测试集总样本数：",test_data_df["click"].count())
    print("测试集总样本数：",test_data_df["click"].count())

    all_data_df =  pd.concat([train_data_df, test_data_df],axis=0)

    print("总样本数：",all_data_df["click"].count())

    return all_data_df



# 负样本采样
# 基于item的曝光次数对全局item进行负采样
def negSample_like_word2vec(train_data, items, users, neg_nums=10):
    # 频次出现越多，采样概率越低，打压热门item, 为每个用户采样 neg_num 个负样本
    pos_sample = train_data[train_data['click']==1][['user_id', 'article_id']]

    pos_sample_dic = {}
    for idx, u in enumerate(pos_sample['user_id'].unique().tolist()):
        pos_list = list(pos_sample[pos_sample['user_id']==u]['article_id'].unique().tolist())
        if len(pos_list)>=30:
            pos_sample_dic[u]=pos_list[:30]
        else:
            pos_sample_dic=pos_list
    
    # 统计item出现频次
    article_counts = train_data['article_id'].value_counts()
    df_article_counts = pd.DataFrame(article_counts)
    dic_article_counts = dict(zip(df_article_counts.index.values.tolist(), df_article_counts.article_id.tolist()))

    for item in items:
        if item[0] not in dic_article_counts.keys():
            dic_article_counts[item[0]]=0
    
    # 根据频次排序
    tmp = sorted(list(dic_article_counts.items()), lambda x:x[1], reverse=True)
    n_art = len(tmp)
    article_prob={}
    for idx, item in enumerate(tmp):
        article_prob[item[0]]=cal_pos(idx, n_art)
    

    article_id_list = [a[0] for a in article_prob.items()]
    article_pro_list = [a[1] for a in article_prob.items()]
    pos_sample_users = list(pos_sample_dic.keys())
    all_users_list = [u[0] for u in all_users]

    print("start negative sampling !!!!!!")
    pool = multiprocessing.Pool(core_size)
    res = pool.map(SampleOneProb((pos_sample_users,article_id_list,article_pro_list,pos_sample_dic,neg_nums)), tqdm(all_users_list))

    pool.close()
    pool.join()

    neg_sample_dic = {}
    for idx, u in tqdm(enumerate(all_users_list)):
        neg_sample_dic[u] = res[idx]

    return [[k,i,0] for k,v in neg_sample_dic.items() for i in v]

def DSSM(user_feature_columns, item_feature_columns, dnn_units=[64, 32], 
        temp=10, task='binary'):
    """
    模型构建部分主要是将输入的user特征以及item特征处理完之后分别送入两侧的DNN结构
    """
    feature_encode = FeatureEncoder(user_feature_columns + item_feature_columns)
    feature_input_layers_list = list(feature_encode.feature_input_layer_dict.values())

    # 特征处理
    user_dnn_input, item_dnn_input = process_feature(user_feature_columns,\
        item_feature_columns, feature_encode)

    # 构建模型的核心层
    if len(user_dnn_input) >= 2:
        user_dnn_input = Concatenate(axis=1)(user_dnn_input)
    else:
        user_dnn_input = user_dnn_input[0]
    if len(item_dnn_input) >= 2:
        item_dnn_input = Concatenate(axis=1)(item_dnn_input)
    else:
        item_dnn_input = item_dnn_input[0]

    user_dnn_input = Flatten()(user_dnn_input) 
    item_dnn_input = Flatten()(item_dnn_input)

    user_dnn_out = DNN(dnn_units)(user_dnn_input)
    item_dnn_out = DNN(dnn_units)(item_dnn_input)

    # 计算相似度
    scores = CosinSimilarity(temp)([user_dnn_out, item_dnn_out]) # (B,1)

    # 确定拟合目标
    output = PredictLayer()(scores)

    # 根据输入输出构建模型
    model = Model(feature_input_layers_list, output)
    # model.summary()

    return model 