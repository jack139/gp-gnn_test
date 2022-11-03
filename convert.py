# 转换 CMeIE 数据到图数据

import json
from bert4keras.tokenizers import Tokenizer

schemas_path = 'data/CMeIE/53_schemas.jsonl'
data_path = 'data/CMeIE/example_gold.jsonl'

maxlen = None # 不限长度
dict_path = '../../nlp/nlp_model/chinese_bert_L-12_H-768_A-12/vocab.txt'
tokenizer = Tokenizer(dict_path, do_lower_case=True, 
    token_start=None, token_end=None) # 不要起始标记 [CLS] [SEP]

# >>> token_ids, segment_ids = tokenizer.encode("1978-03-23 3.担忧过去的行为；", maxlen=maxlen)
# >>> token_ids
# [8774, 118, 8140, 118, 8133, 124, 119, 2857, 2569, 6814, 1343, 4638, 6121, 711, 8039]
# >>> tokenizer.ids_to_tokens(token_ids)
# ['1978', '-', '03', '-', '23', '3', '.', '担', '忧', '过', '去', '的', '行', '为', '；']


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


# 加载关系
p2id, id2p = {}, {}
with open(schemas_path) as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in p2id:
            id2p[f"P{len(p2id)}"] = l['predicate']
            p2id[l['predicate']] = f"P{len(p2id)}"


# 转换数据
D = []
q2id, id2q = {}, {}

with open(data_path, encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)

        token_ids, segment_ids = tokenizer.encode(l["text"], maxlen=maxlen)
        ids = tokenizer.ids_to_tokens(token_ids)
        
        new_item = {
            "tokens" : ids,
            "vertexSet" : [],
            "edgeSet" : []
        }

        for spo in l['spo_list']:
            s = spo['subject']
            p = spo['predicate']
            o = spo['object']["@value"]

            p_kbID = p2id[p]

            # 查找并转换token
            s = tokenizer.encode(s)[0][1:-1]
            o = tokenizer.encode(o)[0][1:-1]
            s_idx = search(s, token_ids)
            o_idx = search(o, token_ids)
            if s_idx != -1 and o_idx != -1:
                s = token_ids[s_idx : s_idx + len(s) - 1]
                o = token_ids[o_idx : o_idx + len(o) - 1]
                s_ids = tokenizer.ids_to_tokens(s)
                o_ids = tokenizer.ids_to_tokens(o)
                s_str = ''.join(s_ids)
                o_str = ''.join(o_ids)
            else:
                print("search fail:", l)
                continue

            # 生成 Q 标记
            if s_str not in q2id:
                id2q[f"Q{len(q2id)}"] = s_str
                q2id[s_str] = f"Q{len(q2id)}"

            s_kbID = q2id[s_str]

            if o_str not in q2id:
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
                    "tokenpositions": [ s_idx ],
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
                    "tokenpositions": [ o_idx ],
                    "numericalValue": 0.0,
                    "variable": False,
                    "unique": False,
                    "type": "LEXICAL"
                })

            # 添加 关系到 edgeSet
            new_item["edgeSet"].append({
                "kbID": p_kbID,
                "right": [ s_idx ],
                "left": [ o_idx ]
            })

        D.append(new_item)

with open("data/convert.json", "w", encoding='utf-8') as f:
    json.dump(D, f, indent=4, ensure_ascii=False)

with open("data/P_with_labels.txt", "w", encoding='utf-8') as f:
    for k in id2p.keys():
        f.write(f"{k}\t{id2p[k]}\n")

with open("data/Q_with_labels.txt", "w", encoding='utf-8') as f:
    for k in id2q.keys():
        f.write(f"{k}\t{id2q[k]}\n")
