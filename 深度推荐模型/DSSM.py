# DSSM召回
"""
user侧主要包含一些户画像属性(用户性别,年龄,所在省市,使用设备及系统)
item侧主要包括创建时间,题目,级别,关键词等
"""

import pandas as pd
import datetime

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
    # merge item信息，得到曝光时间和item创建时间； inner join 去除doc_data之外的item
    all_data_tmp = all_data_tmp.join(doc_data.set_index("article_id"),on="article_id",how='inner')

    # 若存在ctime大于expo_time的交互存在就去除这部分错误数据
    all_data_tmp = all_data_tmp[(all_data_tmp["ctime"]<=all_data_tmp["expo_time"])]