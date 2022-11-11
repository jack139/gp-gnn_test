# coding: utf-8
# Copyright (C) 2017 Hao Zhu
#
# Author: Hao Zhu (ProKil.github.io)
#

from utils import evaluation_utils, embedding_utils
from semanticgraph import io
from parsing import legacy_sp_models as sp_models
from models import baselines
import numpy as np
from sacred import Experiment
import json
import torch
from torch import nn
from torch.autograd import Variable
from tqdm import *
import ast
from models.factory import get_model

import torch.nn.functional as F
try:
    from functools import reduce
except:
    pass

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ex = Experiment("test")

np.random.seed(1)

p0_index = 1

def to_np(x):
    return x.data.cpu().numpy()

@ex.config
def main_config():
    """ Main Configurations """
    model_name = "GPGNN"
    load_model = "GPGNN-e027-f1_0.7134.pt" # you should choose the proper model to load
    device_id = 0

    data_folder = "data/"
    save_folder = "data/models/"
    model_params = "model_params.json"
    word_embeddings = "bert_features.txt"

    test_set = "cmeie_test.json"
    #test_set = "test.json"
    
    # a file to store property2idx
    # if is None use model_name.property2idx
    property_index = None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


@ex.automain
def main(model_params, model_name, data_folder, word_embeddings, test_set, property_index, 
    save_folder, load_model):
    
    with open(model_params) as f:
        model_params = json.load(f)

    embeddings, word2idx = embedding_utils.load(data_folder + word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    def check_data(data):
        for g in data:
            if(not 'vertexSet' in g):
                print("vertexSet missed\n")


    print("Reading the property index")
    with open(save_folder + model_name + ".property2idx") as f:
        property2idx = ast.literal_eval(f.read())

    with open(save_folder + "labels.json") as f:
        all_labels = json.load(f)

    max_sent_len = 300 # max tokens
    print("Max sentence length set to: {}".format(max_sent_len))

    graphs_to_indices = sp_models.to_indices_and_entity_pair
    if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_and_entity_pair
    elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask_and_entity_pair  
    elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_entity_pair
    elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding_and_entity_pair

    _, position2idx, _ = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)


    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out).to(device)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(save_folder + load_model))
    else:
        model.load_state_dict(torch.load(save_folder + load_model, map_location=torch.device('cpu')))
    print("Testing")


    print("Results on the test set")
    test_set_data, _ = io.load_relation_graphs_from_file(data_folder + test_set)
    test_as_indices = list(graphs_to_indices(test_set_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    
    #print(test_as_indices)

    # 测试文件数据
    outfile_f = open(save_folder + "CMeIE_test.jsonl", 'w', encoding='utf-8')
    with open(data_folder + test_set, encoding='utf-8') as f:
        test_dataset = json.load(f)
    test_nn = 0

    # P 数据
    p2id, id2p = {}, {}
    P_filepath = "data/P_with_labels.txt"
    if os.path.exists(P_filepath):
        with open(P_filepath) as f:
            for l in f:
                if len(l)==0:
                    continue
                pid, P = l.split()
                id2p[pid] = P
                p2id[P] = pid

    print("Start testing!")
    result_file = open(data_folder + f"result_{model_name}.txt", "w")

    for i in tqdm(range(int(test_as_indices[0].shape[0] / model_params['batch_size']))):
        sentence_input = test_as_indices[0][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        entity_markers = test_as_indices[1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        labels = test_as_indices[2][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]

        if model_name == "GPGNN":
            with torch.no_grad():
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).to(device),
                            Variable(torch.from_numpy(entity_markers.astype(int))).to(device),
                            test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])
        elif model_name == "PCNN":
            with torch.no_grad():
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).to(device), 
                            Variable(torch.from_numpy(entity_markers.astype(int))).to(device), 
                            Variable(torch.from_numpy(np.array(test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False).to(device))        
        else:
            with torch.no_grad():
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).to(device),
                            Variable(torch.from_numpy(entity_markers.astype(int))).to(device))

        _, predicted = torch.max(output, dim=1)
        labels = labels.reshape(-1)
        p_indices = labels != 0
        predicted = np.array(predicted.cpu())[p_indices].tolist()
        labels = labels[p_indices].tolist()

        #for l, p in zip(labels, predicted):
        #    print(all_labels[l-1], '---', all_labels[p-1])

        predicted_nn = 0
        while test_nn < len(test_dataset) and predicted_nn < len(predicted):
            tt = test_dataset[test_nn]
            item = {
                "text" : tt["text"],
                "spo_list" : [],
            }
            for x in tt["edgeSet"]:
                assert predicted_nn<len(predicted), f"{predicted_nn} >= {len(predicted)} !!!"

                predicate = id2p[all_labels[predicted[predicted_nn]-1]]

                if x["right_type"]==x["left_type"] and predicate!="同义词":
                    pass
                else:
                    item["spo_list"].append({
                        "Combined": False,
                        "predicate": predicate,
                        "subject": x["right_text"],
                        "subject_type": x["right_type"],
                        "object": {"@value": x["left_text"]},
                        "object_type": {"@value": x["left_type"]}
                    })
                predicted_nn += 1


            outfile_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            test_nn += 1

        assert predicted_nn==len(predicted), f"{predicted_nn} != {len(predicted)} !!!"

        #break # for test

    outfile_f.close()