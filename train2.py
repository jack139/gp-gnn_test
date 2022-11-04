# coding: utf-8
# Copyright (C) 2017 Hao Zhu
#
# Author: Hao Zhu (ProKil.github.io)
#
import pickle
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
    device_id = 0
    # 
    model_name = "GPGNN"
    data_folder = "data/"
    save_folder = "data/models/"

    model_params = "model_params.json"
    word_embeddings = "bert_features.txt"
    train_set = "test.json" 
    val_set = "test.json" 

    # a file to store property2idx
    # if is None use model_name.property2idx
    property_index = None
    learning_rate = 1e-3
    shuffle_data = True
    save_model = True
    grad_clip = 0.25
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)


@ex.automain
def main(model_params, model_name, data_folder, word_embeddings, train_set, val_set, property_index, learning_rate, shuffle_data, save_folder, save_model, grad_clip):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    with open(model_params) as f:
        model_params = json.load(f)

    embeddings, word2idx = embedding_utils.load(data_folder + word_embeddings)
    print("Loaded embeddings:", embeddings.shape)

    def check_data(data):
        for g in data:
            if(not 'vertexSet' in g):
                print("vertexSet missed\n")

    training_data, _ = io.load_relation_graphs_from_file(data_folder + train_set, load_vertices=True)

    val_data, _ = io.load_relation_graphs_from_file(data_folder + val_set, load_vertices=True)

    check_data(training_data)
    check_data(val_data)

    if property_index:
        print("Reading the property index from parameter")
        with open(save_folder + args.property_index) as f:
            property2idx = ast.literal_eval(f.read())
    else:
        _, property2idx = embedding_utils.init_random({e["kbID"] for g in training_data
                                                    for e in g["edgeSet"]} | {"P0"}, 1, add_all_zeroes=True, add_unknown=True)
    
    max_sent_len = max(len(g["tokens"]) for g in training_data)
    print("Max sentence length:", max_sent_len)

    max_sent_len = 36
    print("Max sentence length set to: {}".format(max_sent_len))

    graphs_to_indices = sp_models.to_indices
    if model_name == "ContextAware":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding
    elif model_name == "PCNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions_and_pcnn_mask   
    elif model_name == "CNN":
        graphs_to_indices = sp_models.to_indices_with_relative_positions
    elif model_name == "GPGNN":
        graphs_to_indices = sp_models.to_indices_with_real_entities_and_entity_nums_with_vertex_padding

    _, position2idx = embedding_utils.init_random(np.arange(-max_sent_len, max_sent_len), 1, add_all_zeroes=True)

    train_as_indices = list(graphs_to_indices(training_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))

    training_data = None

    n_out = len(property2idx)
    print("N_out:", n_out)

    val_as_indices = list(graphs_to_indices(val_data, word2idx, property2idx, max_sent_len, embeddings=embeddings, position2idx=position2idx))
    val_data = None


    print("Save property dictionary.")
    with open(save_folder + model_name + ".property2idx", 'w') as outfile:
        outfile.write(str(property2idx))

    ###### 测试输出
    with open("data/test-property2idx.json", "w", encoding='utf-8') as f:
        json.dump(property2idx, f, indent=4, ensure_ascii=False)

    #sentences_matrix, entity_matrix, y_matrix, entity_cnt = val_as_indices
    with open("data/test-val.pkl", "wb") as f:
        pickle.dump(val_as_indices, f)

    ######


    print("Training the model")

    print("Initialize the model")
    model = get_model(model_name)(model_params, embeddings, max_sent_len, n_out).to(device)


    #return # 测试

    loss_func = nn.CrossEntropyLoss(ignore_index=0).to(device)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate, weight_decay=model_params['weight_decay'])

    indices = np.arange(train_as_indices[0].shape[0])

    step = 0
    for train_epoch in range(model_params['nb_epoch']):
        if(shuffle_data):
            np.random.shuffle(indices)
        f1 = 0
        for i in tqdm(range(int(train_as_indices[0].shape[0] / model_params['batch_size']))):
            opt.zero_grad()

            sentence_input = train_as_indices[0][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            entity_markers = train_as_indices[1][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            labels = train_as_indices[2][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]]
            
            if model_name == "GPGNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).to(device), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).to(device), 
                                train_as_indices[3][indices[i * model_params['batch_size']: (i + 1) * model_params['batch_size']]])
            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).to(device), 
                                Variable(torch.from_numpy(entity_markers.astype(int))).to(device), 
                                Variable(torch.from_numpy(np.array(train_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), requires_grad=False).to(device))
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int))).to(device),
                                Variable(torch.from_numpy(entity_markers.astype(int))).to(device))

            loss = loss_func(output, Variable(torch.from_numpy(labels.astype(int))).view(-1).to(device))

            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), grad_clip)
            opt.step()

            _, predicted = torch.max(output, dim=1)
            labels = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels = np.array(labels)[p_indices].tolist()

            _, _, add_f1 = evaluation_utils.evaluate_instance_based(predicted, labels, empty_label=p0_index)
            f1 += add_f1
            

        print("Train f1: ", f1 / (train_as_indices[0].shape[0] / model_params['batch_size']))

        val_f1 = 0
        for i in tqdm(range(int(val_as_indices[0].shape[0] / model_params['batch_size']))):
            sentence_input = val_as_indices[0][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            entity_markers = val_as_indices[1][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            labels = val_as_indices[2][i * model_params['batch_size']: (i + 1) * model_params['batch_size']]
            if model_name == "GPGNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).to(device), 
                                Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).to(device), 
                                val_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])
            elif model_name == "PCNN":
                output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).to(device), 
                                Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).to(device), 
                                Variable(torch.from_numpy(np.array(val_as_indices[3][i * model_params['batch_size']: (i + 1) * model_params['batch_size']])).float(), volatile=True).to(device))        
            else:
                output = model(Variable(torch.from_numpy(sentence_input.astype(int)), volatile=True).to(device), 
                                Variable(torch.from_numpy(entity_markers.astype(int)), volatile=True).to(device))

            _, predicted = torch.max(output, dim=1)
            labels = labels.reshape(-1).tolist()
            predicted = predicted.data.tolist()
            p_indices = np.array(labels) != 0
            predicted = np.array(predicted)[p_indices].tolist()
            labels = np.array(labels)[p_indices].tolist()

            _, _, add_f1 = evaluation_utils.evaluate_instance_based(
                predicted, labels, empty_label=p0_index)
            val_f1 += add_f1
        print("Validation f1: ", val_f1 /
                (val_as_indices[0].shape[0] / model_params['batch_size']))

        # save model
        if (train_epoch % 5 == 0 and save_model):
            torch.save(model.state_dict(), "{0}{1}-{2}.out".format(save_folder, model_name, str(train_epoch)))

        step = step + 1

