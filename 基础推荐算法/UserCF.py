import pandas as pd
import numpy as np
import warnings
import random, math, os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


# 评价指标
# 推荐系统推荐正确的商品数量占用户实际点击的商品数量
def Recall(Rec_dict,Val_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...} 
    Val_dict: 用户实际点击的商品列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...}
    '''
    hit_items=0
    all_items=0
    for uid, items in Val_dict:
        rec_set = Rec_dict[uid]
        real_set = items
        for item in rec_set:
            if item in real_set:
                hit_items+=1
        all_items += len(real_set)
    
    return round(hit_items/all_items *100, 2)


# 推荐系统推荐正确的商品数量占给用户实际推荐的商品数
def Precision(Rec_dict, Val_dict):
    """
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...} 
    Val_dict: 用户实际点击的商品列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...}
    """
    hit_items=0
    all_items=0
    for uid, items in Val_dict:
        rec_set = Rec_dict[uid]
        real_set = items
        for item in rec_set:
            if item in real_set:
                hit_items+=1
        all_items += len(rec_set)
    
    return round(hit_items/all_items *100, 2)


# 所有被推荐的用户中,推荐的商品数量占用户实际点击的商品数量
def Coverage(Rec_dict, Trn_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...} 
    Trn_dict: 训练集用户实际点击的商品列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...}
    '''
    rec_items = set()
    all_items = set()
    for uid in Rec_dict:
        for item in Trn_dict[uid]:
            all_items.add(item)
        for item in Rec_dict[uid]:
            rec_items.add(item)
    return round(len(rec_items) / len(all_items) * 100, 2)


# 使用平均流行度度量新颖度,如果平均流行度很高(即推荐的商品比较热门),说明推荐的新颖度比较低
# 从训练集中用户实际点击的商品列表得到每个item的点击次数,再匹配推荐算法返回的item占的次数,得到新颖度
def Popularity(Rec_dict, Trn_dict):
    '''
    Rec_dict: 推荐算法返回的推荐列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...} 
    Trn_dict: 训练集用户实际点击的商品列表, 形式:{uid: [item1, item2,...], uid: [item1, item2,...], ...}
    '''
    pop_items = {}
    for uid in Trn_dict:
        for item in Trn_dict[uid]:
            if item not in pop_items:
                pop_items[item] = 0
            pop_items[item] += 1
    
    pop, num = 0, 0
    for uid in Rec_dict:
        for item in Rec_dict[uid]:
            pop += math.log(pop_items[item] + 1) # 物品流行度分布满足长尾分布,取对数可以使得平均值更稳定
            num += 1  
    return round(pop / num, 3)


# 将几个评价指标指标函数一起调用,注意不同指标中用的是训练集还是验证集
def rec_eval(val_rec_items, val_user_items, trn_user_items):
    print('recall:',Recall(val_rec_items, val_user_items))
    print('precision',Precision(val_rec_items, val_user_items))
    print('coverage',Coverage(val_rec_items, trn_user_items))
    print('Popularity',Popularity(val_rec_items, trn_user_items))


def get_data(root_path):
    """
    数据类型是这样的:1::1193::5::978300760,即user_id::movie_id::rating::timestamp
    """
    # 读数据
    rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
    ratings = pd.read_csv('../data/ml-1m/ratings.dat', sep='::', engine='python', names=rnames)

    # 划分数据集
    train_data, val_data, _, _ = train_test_split(ratings, ratings, test_size=0.2)

    # 依据user_id进行分组,将对应的所有movie_id转成list格式
    train_data = train_data.groupby('user_id')['movie_id'].apply(list).reset_index()
    val_data = val_data.groupby('user_id')['movie_id'].apply(list).reset_index()

    # 将数据构造成字典的形式,{user_id:[item_id1,item_id2,...,item_idn]}
    trn_user_items = {}
    val_user_items = {}

    for user, movies in zip(*(list(train_data['user_id']), list(train_data['movie_id']))):
        trn_user_items[user]=set(movies)
    
    for user, movies in zip(*(list(val_data['user_id']), list(val_data['movie_id']))):
        val_user_items[user]=set(movies)

    return trn_user_items, val_user_items


def User_CF_Rec(trn_user_items, val_user_items, K, N):
    '''
    trn_user_items: 表示训练数据, 格式为：{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
    val_user_items: 表示验证数据, 格式为：{user_id1: [item_id1, item_id2,...,item_idn], user_id2...}
    K: K表示的是相似用户的数量, 每个用户都选择与其最相似的K个用户
    N: N表示的是给用户推荐的商品数量, 给每个用户推荐相似度最大的N个商品
    '''
    # 建立item->users倒排表
    # 倒排表的格式为: {item_id1: [user_id1, user_id2, ... , user_idn], item_id2: ...} 也就是每个item对应有那些用户有过点击
    # 建立倒排表的目的就是为了更好的统计用户之间共同交互的商品数量
    print('建立倒排表...')
    item_users = {}
    for uid, items in tqdm(trn_user_items.items()): # 遍历每一个用户的数据,其中包含了该用户所有交互的item
        for item in items: # 遍历该用户的所有item, 给这些item对应的用户列表添加对应的uid
            if item not in item_users:
                item_users[item] = set()
            item_users[item].add(uid)

    # 计算用户协同过滤矩阵
    # item-users倒排表统计用户之间交互的商品数量，用户协同过滤矩阵为：sim = {user_id1: {user_id2: num1}, user_id3:{user_id4: num2}, ...}
    # 协同过滤矩阵是一个双层的字典，用来表示用户之间共同交互的商品数量
    # 在计算用户协同过滤矩阵的同时还需要记录每个用户所交互的商品数量，其表示形式为: num = {user_id1：num1, user_id2:num2, ...}
    sim = {}
    num = {}
    print('构建协同过滤矩阵...')
    # 因为前面已经建立好了倒排表,倒排表形式:{item_id1: [user_id1, user_id2, ... , user_idn], item_id2: ...}
    for item, users in tqdm(item_users.items()):
        # 先遍历users,记录下来每个用户交互商品的数量
        for user in users:
            if not num[user]:
                num[user]=0
            # 统计每一个用户交互的总的item的数量
            num[user]+=1
            # 现在遍历的是某个确定item的user列表,与确定user不同的用户v都要加入到共同交互计数中
            for v in users:
                if user!=v:
                    if not sim[user]:
                        sim[user][v]=0
                    sim[user][v]+=1
    
    # 计算用户相似度矩阵
    # sim = {user_id1: {user_id2: num1}, user_id3:{user_id4: num2}, ...}
    # 用户协同过滤矩阵sim其实相当于是余弦相似度的分子部分,还需要除以分母,即两个用户分别交互的item数量的乘积
    # 两个用户分别交互的item数量的乘积就是上面统计的num字典
    print('计算相似度...')
    for u, users in tqdm(sim.items()):
        for v, score in users.items():
            # 余弦相似度
            sim[u][v] =  score / math.sqrt(num[u] * num[v]) 
    
    """
    对验证集中的每个用户进行topN推荐
    先通过相似度矩阵得到与当前user最相似的topk个用户,对这k个用户交互过的商品中  除去该测试user在训练集中交互过的商品 剩下的商品,去计算相似度分数
    最终我们推荐的候选商品的相似度分数是由多个用户对该商品分数的一个累加和
    """
    print('给测试用户进行推荐...')
    items_rank = {}
    # 遍历测试集用户, 给测试集中的每个用户进行推荐
    # val_user_items:{user_id:[item_id1,item_id2,...,item_idn]}
    for u, _ in tqdm(val_user_items.items()):
        # 初始化用户u的候选item的字典
        items_rank[u]={}
        # 选出与用户u最相思的k个用户, sim[u].items()对应u那一行所有的user
        for v,score in sorted(sim[u].items(), key=lambda x:x[1], reverse=True):
            # 再遍历选出的这k个用户交互过的item/商品
            for item in trn_user_items[v]:
                # 若相似用户交互过的商品在测试用户的训练集中出现过,就不用推荐,只推荐给测试用户没推荐过的
                if item not in trn_user_items[u]:
                    # 初始化用户u对item的相似度分数为０
                    if item not in items_rank[u]:
                        items_rank[u][item]=0
                    items_rank[u][item]+=score
    
    print('为每个用户筛选出相似度分数最高的Ｎ个商品...')
    items_rank = {k:sorted(v.items(), key=lambda x:x[1], reverse=True)[:N] for k,v in items_rank.items()}
    # 整理格式
    items_rank = {k:set([x[0] for x in v]) for k,v in items_rank.items()}

    return items_rank



if __name__=="__main__":
    root_path = '../data/ml-1m/'
    trn_user_items, val_user_items = get_data(root_path)
    rec_items = User_CF_Rec(trn_user_items, val_user_items, K=80, N=10)
    rec_eval(rec_items, val_user_items, trn_user_items)