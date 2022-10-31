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

ex = Experiment("test")

np.random.seed(1)

p0_index = 1

def to_np(x):
    return x.data.cpu().numpy()

@ex.config
def main_config():
    """ Main Configurations """
    model_name = "GPGNN"
    load_model = "GPGNN-xx.out" # you should choose the proper model to load
    device_id = 0

    data_folder = "data/"
    save_folder = "data/models/"
    result_folder = "result/"
    model_params = "model_params.json"
    word_embeddings = "glove.6B.50d.txt"

    test_set = "test.json"
    
    # a file to store property2idx
    # if is None use model_name.property2idx
    property_index = None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


@ex.automain
def main(model_params, model_name, data_folder, word_embeddings, test_set, property_index, save_folder, load_model, result_folder):
    
    with open(model_params) as f:
        model_params = json.load(f)

    embeddings, word2idx = embedding_utils.load(data_folder + word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    def check_data(data):
        for g in data:
            if(not 'vertexSet' in g):
                print("vertexSet missed\n")


    print("Reading the property index")
    with open(data_folder + "models/" + model_name + ".property2idx") as f:
        property2idx = ast.literal_eval(f.read())
    

    max_sent_len = 36
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

    _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)


    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out).cuda()
    model.load_state_dict(torch.load(save_folder + load_model))
    print("Testing")


    print("Results on the test set")
    test_set, _ = io.load_relation_graphs_from_file(data_folder + test_set)
    test_as_indices = list(graphs_to_indices(test_set, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    
    print("Start testing!")
    result_file = open(result_folder + "_" + model_name, "w")
    for i in tqdm(range(int(test_as_indices[0].shape[0] / model_params['batch_size']))):
        sentence_input = test_as_indices[0][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        entity_markers = test_as_indices[1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
        labels = test_as_indices[2][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]

        if model_name == "GPGNN":
            output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).cuda(),
                            Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).cuda(),
                            test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])
        elif model_name == "PCNN":
            output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).cuda(), 
                            Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).cuda(), 
                            Variable(torch.from_numpy(np.array(test_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False, volatile=True).cuda())        
        else:
            output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).cuda(),
                            Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).cuda())

        score = F.softmax(output)
        score = to_np(score).reshape(-1, n_out)
        labels = labels.reshape(-1)
        p_indices = labels != 0
        score = score[p_indices].tolist()
        labels = labels[p_indices].tolist()
        if(model_name != "LSTM" and model_name != "PCNN" and model_name != "CNN"):
            entity_pairs = test_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            entity_pairs = reduce(lambda x,y :x+y , entity_pairs)
        else:
            entity_pairs = test_as_indices[-1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]    
        for (i, j, entity_pair) in zip(score, labels, entity_pairs):
            for index, k in enumerate(i):
                result_file.write(str(index) + "\t" + str(k) + "\t" + str(1 if index == j else 0) + "\t" + entity_pair[0] + "\t" + entity_pair[1] + "\n")

