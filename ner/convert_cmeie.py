import os
import json


categories = set()

new_ratio = 0.1


cate_map = { # 只使用 一个标签
    "检查" : "entity",
    "疾病" : "entity",
    "症状" : "entity",
    "手术治疗" : "entity",
    "其他治疗" : "entity",
    "部位" : "entity",
    "药物" : "entity",
    "流行病学": "entity", 
    "社会学" : "entity", 
    "预后" : "entity",
    "其他" : "entity"
}

def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


def get_data(infile, include_blank=True):

    D = []

    with open(infile, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)

            text = l['text']
            entities = []

            for e in l['spo_list']:
                s = e['subject']
                s_type = e['subject_type']
                o = e['object']['@value']
                o_type = e['object_type']['@value']

                categories.add(s_type)
                categories.add(o_type)

                s_type = cate_map[s_type] # 转换标签
                o_type = cate_map[o_type]

                s_idx = search(s, text)
                o_idx = search(o, text)

                if s_idx == -1 or o_idx == -1:
                    print('fail: ', s_idx, o_idx, text, e) # 未找到
                    continue

                entities.append({
                    "start_idx": s_idx,
                    "end_idx": s_idx + len(s) - 1,
                    "type": s_type,
                    "entity": s,
                })

                entities.append({
                    "start_idx": o_idx,
                    "end_idx": o_idx + len(o) - 1,
                    "type": o_type,
                    "entity": o,
                })

            # 加入数据集
            if include_blank or len(entities)>0:
                entities_ = []
                for e in entities: # 去除一样的实体
                    equal = False
                    for e2 in entities_:
                        if e['start_idx']==e2['start_idx'] and e['end_idx']==e2['end_idx'] and e['type']==e2['type']:
                            equal = True
                            break

                    if not equal:
                        entities_.append(e)

                D.append({
                    'text' : text,
                    'entities' : entities_,
                })


    return D


if __name__ == '__main__':

    D_train = get_data('../resources/CMeIE/CMeIE_train.jsonl', False)
    D_dev = get_data('../resources/CMeIE/CMeIE_dev.jsonl', False)

    D_test = get_data('../resources/CMeIE/CMeIE_test.jsonl', True)

    # 重新分割
    train_num = len(D_train)
    dev_num = len(D_dev)
    test_num = len(D_test)

    print(f"train: {train_num}\tdev: {dev_num}\ttest: {test_num}")

    new_dev_num = int((train_num + dev_num) * new_ratio)

    print(f"new train: {train_num+(dev_num-new_dev_num)}\tnew dev: {new_dev_num}")

    json.dump(
        D_train + D_dev[:-new_dev_num],
        open('../data/ner/cmeie_train.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    json.dump(
        D_dev[-new_dev_num:],
        open('../data/ner/cmeie_dev.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    json.dump(
        D_test,
        open('../data/ner/cmeie_test.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )

    json.dump(
        list(categories),
        open('../data/ner/cmeie_categories.json', 'w', encoding='utf-8'),
        indent=4,
        ensure_ascii=False
    )
