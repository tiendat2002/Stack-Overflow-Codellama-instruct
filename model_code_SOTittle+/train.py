from __future__ import absolute_import

import json
import os.path
import time
import logging
from utils import read_prompt_examples, get_elapse_time, calculate_rouge
from SOTitlePlus import SOTitlePlus


class Config(object):
    def __init__(self):
        self.cuda = True
        self.train_filename = '../datasets/good_title_datas/train.json' # "./selected_data/train.json" for selected_data
        self.dev_filename = '../datasets/good_title_datas/valid.json' # "./selected_data/valid.json" for selected_data
        self.test_filename = '../datasets/good_title_datas/test.json'
        self.path_selected_data = './selected_data/'
        self.model_type = 'codet5'
        self.model_name_or_path = "Salesforce/codet5-base"
        self.log_name = './log/python.log'
        self.output_dir = "./model/"  # "./selected_mode/" for selected_data
        self.result_dir = './results' # "./selected_results" for selected_data
        self.langs = ['python', 'java', 'c#', 'javascript', 'php']
        self.no_cuda = False
        self.visible_gpu = "0"
        self.add_task_prefix = False
        self.add_lang_ids = False
        self.num_train_epochs = 50
        self.train_batch_size = 8
        self.eval_batch_size = 8
        self.gradient_accumulation_steps = 2

        # other configs
        self.load_model_path = './model/' # "./selected_mode/" for selected_data
        self.train_load_model_path = None #for loading model from pre-tuning
        self.config_name = ""
        self.tokenizer_name = ""
        self.max_source_length = 512
        self.max_target_length = 80
        self.warm_up_ratio = 0.1
        self.cache_dir = './cache_dir/'
        self.mode = 'train'

        # controlling configs
        self.do_train = True
        self.do_eval = True
        self.do_test = True
        self.learning_rate = 5e-5
        self.beam_size = 10
        self.weight_decay = 0.0
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.max_steps = -1
        self.eval_steps = -1
        self.train_steps = 5000
        self.local_rank = -1
        self.seed = 42
        self.early_stop_threshold = 5


if __name__ == '__main__':

    my_config = Config()

    if not os.path.exists('./log/python.log'):
        os.makedirs('./log/python.log', exist_ok=True)

    # begin time
    begin_time = time.time()

    # logger for record
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger(__name__)

    # write to file
    handler = logging.FileHandler(my_config.log_name)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    # write to console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # print config
    logger.info(my_config)

    model = SOTitlePlus(my_config)

    model.train()

    for lan in my_config.langs:
        logger.info(f'lan:{lan}')
        model.test(lan, f'../datasets/good_title_datas/{lan}/test.json')

    # make selected data for tuning original SOTitle+
    # model.predict_selected_data()
    # after run predict_selected_data run handel_data.ipynb in ./selected_data/
    # after run handel_data.ipynb then change config in Config for selected_data

    logger.info("Finish training and take %s", get_elapse_time(begin_time))