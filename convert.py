# 转换 CMeIE 数据到图数据
import os
import json
from bert4keras.tokenizers import Tokenizer

schemas_path = 'resources/CMeIE/53_schemas.jsonl'
P_filepath = "data/P_with_labels.txt"
Q_filepath = "data/Q_with_labels.txt"


#data_path = 'resources/CMeIE/CMeIE_train.jsonl'
#newfile_path = 'data/cmeie_train.json'
#data_path = 'resources/CMeIE/CMeIE_dev.jsonl'
#newfile_path = 'data/cmeie_dev.json'
data_path = 'data/example_test.jsonl'
newfile_path = 'data/test.json'


maxlen = None # 不限长度
dict_path = '../../nlp/nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True, 
    token_start=None, token_end=None) # 不要起始标记 [CLS] [SEP]

# >>> token_ids, segment_ids = tokenizer.encode("1978-03-23 3.担忧过去的行为；", maxlen=maxlen)
# >>> token_ids
# [8774, 118, 8140, 118, 8133, 124, 119, 2857, 2569, 6814, 1343, 4638, 6121, 711, 8039]
# >>> tokenizer.ids_to_tokens(token_ids)
# ['1978', '-', '03', '-', '23', '3', '.', '担', '忧', '过', '去', '的', '行', '为', '；']


P_nn = Q_nn = 0

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1

# 加载已有P/Q文件
p2id, id2p = {}, {}
q2id, id2q = {}, {}

if os.path.exists(P_filepath):
    with open(P_filepath) as f:
        for l in f:
            if len(l)==0:
                continue
            pid, P = l.split()
            id2p[pid] = P
            p2id[P] = pid

if os.path.exists(Q_filepath):
    with open(Q_filepath) as f:
        for l in f:
            if len(l)==0:
                continue
            qid, Q = l.split()
            id2q[qid] = Q
            q2id[Q] = qid

# 加载关系
with open(schemas_path) as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in p2id:
            print("append P:", l['predicate'])
            P_nn += 1
            id2p[f"P{len(p2id)+6}"] = l['predicate']
            p2id[l['predicate']] = f"P{len(p2id)+6}"


# 转换数据
D = []
with open(data_path, encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)

        token_ids, segment_ids = tokenizer.encode(l["text"], maxlen=maxlen)
        ids = tokenizer.ids_to_tokens(token_ids)
        
        new_item = {
            #"tokens" : [Tokenizer.stem(i) for i in ids], # 去掉前缀 '##'
            "tokens" : ids,
            "vertexSet" : [],
            "edgeSet" : []
        }

        for spo in l['spo_list']:
            s = spo['subject']
            p = spo['predicate']
            o = spo['object']["@value"]

            p_kbID = p2id[p]

            # 查找并转换token, 使用token查找，因为token不一定是单字符
            s = tokenizer.encode(s)[0]
            o = tokenizer.encode(o)[0]
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = token_ids[s_idx : s_idx + len(s)]
                o = token_ids[o_idx : o_idx + len(o)]
                s_ids = tokenizer.ids_to_tokens(s)
                o_ids = tokenizer.ids_to_tokens(o)
                #s_str = ''.join([Tokenizer.stem(i) for i in s_ids]) # 去掉前缀 '##'
                #o_str = ''.join([Tokenizer.stem(i) for i in o_ids]) # 去掉前缀 '##'
                s_str = ''.join(s_ids)
                o_str = ''.join(o_ids)
            else:
                print("search fail:", l)
                print(tokenizer.ids_to_tokens(token_ids))
                print(tokenizer.ids_to_tokens(s))
                print(tokenizer.ids_to_tokens(o))
                continue

            # 生成 Q 标记
            if s_str not in q2id:
                print("append Q:", s_str)
                Q_nn += 1
                id2q[f"Q{len(q2id)}"] = s_str
                q2id[s_str] = f"Q{len(q2id)}"

            s_kbID = q2id[s_str]

            if o_str not in q2id:
                print("append Q:", o_str)
                Q_nn += 1
                id2q[f"Q{len(q2id)}"] = o_str
                q2id[o_str] = f"Q{len(q2id)}"

            o_kbID = q2id[o_str]

            # 检查 s o 是否在 vertexSet, 不在则添加
            def in_vertexSet(kbID):
                for xx in new_item["vertexSet"]:
                    if kbID==xx["kbID"]:
                        return True
                return False

            if not in_vertexSet(s_kbID):
                new_item["vertexSet"].append({
                    "kbID": s_kbID,
                    "lexicalInput": s_str,
                    "namedEntity": True,
                    "tokenpositions": [ i for i in range(s_idx, s_idx + len(s)) ],
                    "numericalValue": 0.0,
                    "variable": False,
                    "unique": False,
                    "type": "LEXICAL"
                })

            if not in_vertexSet(o_kbID):
                new_item["vertexSet"].append({
                    "kbID": o_kbID,
                    "lexicalInput": o_str,
                    "namedEntity": True,
                    "tokenpositions": [ i for i in range(o_idx, o_idx + len(o)) ],
                    "numericalValue": 0.0,
                    "variable": False,
                    "unique": False,
                    "type": "LEXICAL"
                })

            # 添加 关系到 edgeSet
            new_item["edgeSet"].append({
                "kbID": p_kbID,
                "right": [ i for i in range(s_idx, s_idx + len(s)) ],
                "left": [ i for i in range(o_idx, o_idx + len(o)) ]
            })

        D.append(new_item)

# 保存文件
with open(newfile_path, "w", encoding='utf-8') as f:
    json.dump(D, f, 
        #indent=4, 
        ensure_ascii=False)
    print(newfile_path, "saved.")

if P_nn > 0:
    with open(P_filepath, "w", encoding='utf-8') as f:
        for k in id2p.keys():
            f.write(f"{k}\t{id2p[k]}\n")
    print(P_filepath, "saved.")

if Q_nn > 0:
    with open(Q_filepath, "w", encoding='utf-8') as f:
        for k in id2q.keys():
            f.write(f"{k}\t{id2q[k]}\n")

    print(Q_filepath, "saved.")

print(len(D), len(id2p), len(id2q))
