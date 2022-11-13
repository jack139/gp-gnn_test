# 转换 CMeIE 数据到图数据
import os
import json
from bert4keras.tokenizers import Tokenizer

schemas_path = 'resources/CMeIE/53_schemas.jsonl'
P_filepath = "data/P_with_labels.txt"
Q_filepath = "data/Q_with_labels.txt"

MAX_vertex_num = 9

#data_path = 'resources/CMeIE/CMeIE_train.jsonl'
#newfile_path = 'data/cmeie_train.json'
data_path = 'resources/CMeIE/CMeIE_dev.jsonl'
newfile_path = 'data/cmeie_dev.json'
#data_path = 'data/example_test.jsonl'
#newfile_path = 'data/test.json'


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
max_token_len = max_vertex_n = max_edge_n = 0
max_vertex_n_9 = [0] * 40

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

if "P1" not in id2p: # P1 用于标记“无用”的边
    id2p["P1"] = "P1"
    p2id["P1"] = "P1"


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

        entity_map = { # 只使用 一个标签
            "检查"    : [],
            "疾病"    : [],
            "症状"    : [],
            "手术治疗" : [],
            "其他治疗" : [],
            "部位"    : [],
            "药物"    : [],
            "流行病学" : [],
            "社会学"  : [],
            "预后"    : [],
            "其他"    : [],
        }

        spo_schemas = []

        for spo in l['spo_list']:
            s = spo['subject']
            p = spo['predicate']
            o = spo['object']["@value"]

            p_kbID = p2id[p]

            # 查找并转换token, 使用token查找，因为token不一定是单字符
            s = tokenizer.encode(spo['subject'])[0]
            o = tokenizer.encode(spo['object']["@value"])[0]
            try_again = 0
            while  True:
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
                    if try_again==0: # 这里假设 s_idx 和 o_idx 不会同时为 -1
                        if s_idx==-1:
                            s = tokenizer.encode('⑤'+spo['subject'])[0] # ⑤a 会变成  ⑤##a, 所以转换一下，进行匹配
                            s = s[1:]
                            try_again += 1
                            continue
                        elif o_idx==-1:
                            o = tokenizer.encode('⑤'+spo['object']["@value"])[0] # ⑤a 会变成  ⑤##a, 所以转换一下，进行匹配
                            o = o[1:]
                            try_again += 1
                            continue

                    print("search fail:", l)
                    print(tokenizer.ids_to_tokens(token_ids))
                    print(tokenizer.ids_to_tokens(s))
                    print(tokenizer.ids_to_tokens(o))
                    try_again += 1 # 此时应为 2

                break

            if try_again > 1: # 放弃这个数据
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

            if not in_vertexSet(s_kbID) and len(new_item["vertexSet"])<MAX_vertex_num: # 节点数超过，不再添加边
                tokenpositions = [ i for i in range(s_idx, s_idx + len(s)) ]
                new_item["vertexSet"].append({
                    "kbID": s_kbID,
                    "lexicalInput": s_str,
                    "namedEntity": True,
                    "tokenpositions": tokenpositions,
                    "numericalValue": 0.0,
                    "variable": False,
                    "unique": False,
                    "type": "LEXICAL"
                })

                entity_map[spo["subject_type"]].append((tokenpositions, spo['subject']))

            if not in_vertexSet(o_kbID) and len(new_item["vertexSet"])<MAX_vertex_num:
                tokenpositions = [ i for i in range(o_idx, o_idx + len(o)) ]
                new_item["vertexSet"].append({
                    "kbID": o_kbID,
                    "lexicalInput": o_str,
                    "namedEntity": True,
                    "tokenpositions": tokenpositions,
                    "numericalValue": 0.0,
                    "variable": False,
                    "unique": False,
                    "type": "LEXICAL"
                })

                entity_map[spo["object_type"]["@value"]].append((tokenpositions, spo['object']["@value"]))

            # 添加 关系到 edgeSet
            if in_vertexSet(s_kbID) and in_vertexSet(o_kbID): 
                new_item["edgeSet"].append({
                    "kbID": p_kbID,
                    "right": [ i for i in range(s_idx, s_idx + len(s)) ],
                    "left": [ i for i in range(o_idx, o_idx + len(o)) ]
                })

            # 记录已添加的边
            spo_schemas.append(f"{spo['subject']}_{spo['object']['@value']}")


        # 生成边，“无用的”
        for d in entity_map["疾病"]: # 疾病为主语的
            for k in entity_map.keys():
                if k=="疾病":
                    continue
                for j in entity_map[k]:
                    if f"{d[1]}_{j[1]}" in spo_schemas: # 已有边
                        continue
                    new_item["edgeSet"].append({
                        "kbID": "P1",
                        "right": d[0],
                        "left": j[0],
                    })

        for k in entity_map.keys(): # 同义词
            if len(entity_map[k])<2:
                continue
            for i in range(len(entity_map[k])-1):
                for j in range(i+1, len(entity_map[k]), 1):
                    if f"{entity_map[k][i][1]}_{entity_map[k][j][1]}" in spo_schemas: # 已有边
                        continue
                    new_item["edgeSet"].append({
                        "kbID": "P1",
                        "right": entity_map[k][i][0],
                        "left": entity_map[k][j][0],
                    })


        if len(new_item["vertexSet"])<2: # 忽略只有一个节点的数据
            continue

        D.append(new_item)

        max_token_len = max(max_token_len, len(new_item["tokens"]))
        max_vertex_n = max(max_vertex_n, len(new_item["vertexSet"]))
        max_edge_n = max(max_edge_n, len(new_item["edgeSet"]))

        max_vertex_n_9[len(new_item["vertexSet"])] += 1



# 保存文件
with open(newfile_path, "w", encoding='utf-8') as f:
    json.dump(D, f, 
        indent=4, 
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

print(f"total= {len(D)}\tP= {len(id2p)}\tQ= {len(id2q)}")
print(f"token_len= {max_token_len}\tvertex_n= {max_vertex_n}\tedge_n= {max_edge_n}")
print(f"max_vertex_n_9= {max_vertex_n_9}")

'''
未用MAX_vertex_num过滤前：

Train:
total= 14336    P= 44   Q= 30464
token_len= 297  vertex_n= 37    edge_n= 36

Dev:
total= 3585 P= 44   Q= 30464
token_len= 286  vertex_n= 31    edge_n= 32

过滤后

Train:
total= 14336    P= 45   Q= 29475
token_len= 297  vertex_n= 9 edge_n= 47

total= 3585 P= 45   Q= 29475
token_len= 286  vertex_n= 9 edge_n= 43

'''