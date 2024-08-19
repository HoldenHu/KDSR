import os
import sys
import torch
import numpy as np
import argparse
import pickle
from logging import getLogger

from data_utils import split_validation, Data, get_item_num
from utils import Config, init_seed, init_logger
from utils import trans_to_cuda, get_dataset_folder

from model_corvector import KDSR, train_and_test
from model_corvector import model_process
from model_corvector import evaluate

# python main.py KDSR sample
parser = argparse.ArgumentParser()
parser.add_argument('dataset', default='sample', help='data name')
inni_opt = parser.parse_args()

# model_name = inni_opt.model_name
model_name = 'KDSR'
dataset = inni_opt.dataset

def main(para_key, para_value):
    config = Config(model=model_name, dataset=dataset, config_file_list=['config/dataset.yaml', 'config/{}--4--{}.yaml'.format(model_name, dataset)])
    init_seed(config['seed'], config['reproducibility'], config['cuda_id'])
    config = init_logger(config)
    logger = getLogger()
    if config['tune_parameters']:
        for k,v in zip(para_key, para_value):
            logger.info("TUNING PARAMETERS: {} - {}".format(str(k), str(v)))
            config[k] = v
    else:
        config['noise_type'] = []
        config['noise_level'] = 0
    logger.info(config.parameters)

    data_folder = get_dataset_folder(config, dataset)
    logger.info("Used data_folder: "+data_folder)

    train_data = pickle.load(open(os.path.join(data_folder, 'train.txt'), 'rb'))
    test_data = pickle.load(open(os.path.join(data_folder, 'test.txt'), 'rb'))
    num_item = get_item_num(train_data, test_data)
    config['num_node'][dataset] = num_item
    config['image_embedding_size'] = 128
    config['text_embedding_size'] = 128
    
    train_data = Data(train_data)
    test_data = Data(test_data)

    if config['use_image_modality_encoder'] != '':
        iid_image_feature = pickle.load(open(os.path.join(data_folder, 'itemid_image_{}-feature{}.pickle'.format(config['use_image_modality_encoder'],config['image_embedding_size'])) , 'rb'))  # (12102, 128)
    else:
        iid_image_feature = pickle.load(open(os.path.join(data_folder, 'itemid_image_feature{}.pickle'.format(config['image_embedding_size'])) , 'rb'))  # (12102, 128)
    if config['use_text_modality_encoder'] != '':
        iid_text_feature = pickle.load(open(os.path.join(data_folder, 'itemid_text_{}-feature{}.pickle'.format(config['use_text_modality_encoder'], config['text_embedding_size'])) , 'rb'))
    else:
        iid_text_feature = pickle.load(open(os.path.join(data_folder, 'itemid_text_feature{}.pickle'.format(config['text_embedding_size'])) , 'rb'))

    model = trans_to_cuda(KDSR(config, iid_image_feature, iid_text_feature))
    logger.info(model)

    train_loader = torch.utils.data.DataLoader(train_data, num_workers=8, batch_size=model.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=model.batch_size, shuffle=False, pin_memory=True)

    best_hit, best_hit_epoch = [0, 0], [0, 0]
    best_mrr, best_mrr_epoch = [0, 0], [0, 0]
    best_ndcg, best_ndcg_epoch = [0, 0], [0, 0]
    best_model_dict = None
    bad_counter = 0

    last_flag = 0 # whether is the best result
    for epoch in range(config['epoch']):
        logger.info('-------------------------------------------------------')
        logger.info('epoch: '+ str(epoch))
        
        out_scores = train_and_test(model, train_loader, test_loader, config['topk'], logger)
        current_flag = 0 # whether there is any better result outcome
        for k_idx, out_score in enumerate(out_scores):
            k = config['topk'][k_idx]
            hit, mrr, ndcg = out_score
            logger.info('Current Result: Recall@%d: %.4f \t MMR@%d: %.4f \t NDCG@%d: %.4f' % (k, hit, k, mrr, k, ndcg))
            if hit > best_hit[k_idx]:
                best_hit[k_idx] = hit
                best_hit_epoch[k_idx] = epoch
                current_flag = 1
            if mrr > best_mrr[k_idx]:
                best_mrr[k_idx] = mrr
                best_mrr_epoch[k_idx] = epoch
                current_flag = 1
            if ndcg > best_ndcg[k_idx]:
                best_ndcg[k_idx] = ndcg
                best_ndcg_epoch[k_idx] = epoch
                current_flag = 1
        if current_flag == 1:
            best_model_dict = model.state_dict()
        for k_idx, k in enumerate(config['topk']):
            logger.info('>> Best Result: Recall@%d: %.4f \t MMR@%d: %.4f \t NDCG@%d: %.4f   (Epoch: \t %d, \t %d, \t %d)' 
                        % (k, best_hit[k_idx], k, best_mrr[k_idx], k, best_ndcg[k_idx], best_hit_epoch[k_idx], best_mrr_epoch[k_idx], best_ndcg_epoch[k_idx]))
        
        bad_counter = bad_counter+1 if (last_flag==0 and current_flag==0) else 0
        last_flag = current_flag
        if bad_counter >= config['patience']:
            logger.info('Early stoped.')
            break
    logger.info('========================================================')
    
    new_log_path = os.path.join(config["log_dir"],str(best_mrr[1])+".log")
    os.rename(os.path.join(config["log_dir"],"logging.log"), new_log_path)

    # output topk (1,30)
    model.load_state_dict(best_model_dict)
    model.eval()
    topk = list(range(1,31))
    hit, mrr = [[] for k in topk], [[] for k in topk]
    for data in test_loader:
        targets, scores, _, _, _, _ = model_process(model, data, stage='test')
        for i, k in enumerate(topk):
            hit_scores, mrr_scores, ndcg_scores = evaluate(targets, scores, k)
            hit[i] = hit[i]+hit_scores.tolist()
            mrr[i] = mrr[i]+mrr_scores.tolist()
    result = [[] for k in topk]
    for i, k in enumerate(topk):
        result[i].append(np.mean(hit[i]) * 100)
        result[i].append(np.mean(mrr[i]) * 100)
    np.save(os.path.join(config["log_dir"],"top30_list.npy"),result)

    #########################
    logger.info(new_log_path)

    if not config['tune_parameters']:
        sys.exit(0)
    else:
        for hdlr in logger.handlers[:]:  # remove all old handlers
            logger.removeHandler(hdlr)


if __name__ == '__main__':
    # para_dict = {
    #     'embedding_size': [128, 256],
    #     'dropout_output': [0.2, 0.4],
    #     'dropout_atten': [0.2, 0.4],
    #     'lr': [0.001, 0.005]
    # }
        # cor_code_num: 50 
        # product_domain_D: 8
    para_dict = {
        'cor_code_num': [20, 50, 100],
        'product_domain_D': [1, 4, 16]
    }
    paraname, para_list = list(para_dict.keys()), list(para_dict.values())
    from itertools import product
    para_list = list(product(*para_list))

    for paras in para_list:
        main(paraname, paras)
