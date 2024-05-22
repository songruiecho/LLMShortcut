import math
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import log
from nltk.tokenize import word_tokenize, MWETokenizer
import pickle

from transformers import AutoTokenizer, AutoModel, AutoModelForMultipleChoice, AutoModelForCausalLM
from transformers import OPTForCausalLM, OPTForSequenceClassification
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import queue
import random

class Config:
    def __init__(self):
        super(Config, self).__init__()
        self.batch = 8
        self.LLM_path = '/home/songrui/data/'
        # self.LLM_path = '/fs1/private/user/songrui/datas/'
        self.LLM = 'opt-1.3b'  # GLM-2b 7b
        self.K = 2   # 0 for zero-shot   k=[-1,1,2,3,4,5] -1 indicates that Direct shortcut injection
        self.prompt_task = 1
        self.base_task = 'toxic'  # emotion or toxicity
        self.del_shortcut = False   # 是否删除待预测样本中的捷径
        assert self.prompt_task in [1,2,3,3.5,4]
        self.cudas = [0,1,2,3]
        # 1 for Simple few-shot learning; 2 for Direct shortcut injection;
        # 3 for Indirect shortcut injection
        # 解释一下，1. 简单少样本学习就是直接构建prompt进行学习，如ACL2023所示；这个过程需要对四个prompt都进行计算然后取平均值
        # 2.直接捷径注入则是考虑在少样本之前增添一个恶意引导，类似Input: {jews jews jews} Toxic? Answer: {yes}。
        # 这样就可以让模型立即捕捉这种表面关联。
        # 3. 则是采样相反样本中捷径替换的方式让模型间接捕捉。比如Input:{She is bad} Prediction:{negative}; Input:{[MASK] is good} Prediction:{positive}
        # 通过这种对带有相反标签样本中词替换的方式削弱相反语义下的捷径词影响，目的是让LLMs关注She-negative之间的关联

def compute_bm25_similarity(texts):
    # 计算文本中的总词数
    total_words = sum(len(text.split()) for text in texts)
    # 计算文本中每个词汇项的文档频率（DF）
    df = Counter()
    for text in texts:
        words = set(text.split())
        for word in words:
            df[word] += 1
    # 计算文档总数
    N = len(texts)
    # 设置BM25参数
    k1 = 1.5  # 调节因子，控制文档长度对相似性的影响
    b = 0.75  # 调节因子，控制文档长度对相似性的影响
    # 初始化相似性矩阵
    similarity_matrix = np.zeros((N, N))
    # 计算每对文本之间的BM25相似性
    for i in range(N):
        for j in range(N):
            if i == j:
                continue  # 不需要计算文本与自己的相似性
            text1 = texts[i].split()
            text2 = texts[j].split()
            intersection = set(text1) & set(text2)  # 交集词汇项
            similarity = 0
            for word in intersection:
                idf = log((N - df[word] + 0.5) / (df[word] + 0.5) + 1.0)  # 计算逆文档频率
                tf1 = text1.count(word) / len(text1)  # 计算词频
                tf2 = text2.count(word) / len(text2)
                similarity += idf * ((tf1 * (k1 + 1)) / (tf1 + k1 * (1 - b + b * (len(text1) / total_words))) *
                                     (tf2 * (k1 + 1)) / (tf2 + k1 * (1 - b + b * (len(text2) / total_words))))
            similarity_matrix[i][j] = similarity
    return similarity_matrix.sum(axis=0)

def replace_shortcut(tokenizer, shortcut, text):
    shortcut = shortcut.lower()
    replace_text = []
    words = tokenizer.tokenize(word_tokenize(text.lower()))
    for word in words:
        if word != shortcut:
            replace_text.append(word)
        # else:
        #         #     replace_text.append(''.join([' ']*len(shortcut.split())))     # 用等长的空格代替
    return ' '.join(replace_text)

def random_word(tokenizer, shortcut, text):
    words = tokenizer.tokenize(word_tokenize(text.lower()))
    return random.choice(words)

def generate_prompt(cfg, prompt_type:int=0):
    '''
    :param prompt_type:  表示提示模板的类型，我们采用四种模板因此是[0,1,2,3]
    :return:
    '''
    print('加载数据集...')
    assert prompt_type in [0,1,2,3]
    if cfg.base_task == 'emotion':
        frame = pd.read_csv('emotion-benchmark.csv')
        print(frame.shape)
        key2label = {
            'positive':1,
            'negative':0
        }
        templates = [
            ['Review: {} Sentiment:{}', ["negative","positive"]],   # positive/negative
            ['Input: {} Prediction:{}', ["negative", "positive"]],   # positive/negative
            ['Input: {} Prediction:{}', ["bad", "good"]],   # good/bad
            ['Input: {} It is good or bad? Answer:{}', ["bad", "good"]]        # good/bad
        ]
        with open('save_models/{}-causals.pkl'.format('emo'), 'rb') as wf:
            keywords = pickle.load(wf)
    else:
        frame = pd.read_csv('toxic-benchmark.csv')
        print(frame.shape)
        key2label = {
            'toxic':1,
            'normal':0
        }
        templates = [
            ['Input: {} Prediction:{}', ["normal", "toxic"]],  # toxic/normal
            ['Sentence: {} Result:{}', ["normal", "toxic"]],  # yes/no
            ['Sentence: {} Prediction:{}', ["normal", "toxic"]],  # toxic/normal
            ['Input: {} Result:{}', ["normal", "toxic"]]     # toxic/normal
        ]
        with open('save_models/{}-causals.pkl'.format('toxic'), 'rb') as wf:
            keywords = pickle.load(wf)

    shortcut2sen, sentence2keywords = {}, {}
    sentences = list(frame['sentences'].values)
    labels = list(frame['labels'].values)
    shortcuts = list(frame['shortcuts'].values)
    labels = [key2label[each] for each in labels]
    shortcuts = [each.lower() for each in shortcuts]
    for i in range(len(labels)):
        if shortcuts[i] not in shortcut2sen.keys():
            shortcut2sen[shortcuts[i]] = {'0':[], '1':[]}
        if labels[i] == 0:
            shortcut2sen[shortcuts[i]]['0'].append(sentences[i])
        else:
            shortcut2sen[shortcuts[i]]['1'].append(sentences[i])
    for sen, keyword, in zip(sentences, keywords):
        ks = []
        for each in keyword:
            if len(each[0]) > 1:   # 去除标点
                ks.append(each)
        sentence2keywords[sen] = ks
    # all_words = set(words.words() + groups)   # 合并词表
    mywords = [tuple(each.split()) for each in set(shortcuts) if ' ' in each]
    myTokenizer = MWETokenizer(mywords, separator=' ')    # 定义预处理的分词器
    # 建立shortcut到对应句子的映射
    # 随后生成batch数据
    inputs = []
    assert len(keywords) == len(sentences)
    for i in range(len(labels)):
        shortcut, label, sen, keyword = shortcuts[i], labels[i], sentences[i], keywords[i]
        pos_sens, neg_sens = [], []  # 用于存放待预测样本的积极和消极few-shot样本
        if cfg.prompt_task == 1:   # 简单的少样本学习，选择语义多样新topK对样本进行训练
                for k in range(20):
                    p_sen = shortcut2sen[shortcut]['0'][k]
                    if sen != p_sen:
                        neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                    if len(neg_sens) == cfg.K:
                        break
                for k in range(20):
                    p_sen = shortcut2sen[shortcut]['1'][k]
                    if sen != p_sen:
                        pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                    if len(pos_sens) == cfg.K:
                        break
                p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
                prompt = '\n'.join(p_sens)
                prompt = prompt + '\n' + templates[prompt_type][0].format(sen, '')
        if cfg.prompt_task == 2:  # 直接捷径引导，在每一个与带预测样本相反的样本后边添加一个shortcut后缀
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            if label == 0:
                shortcut_injection = templates[prompt_type][0].format(shortcut, templates[prompt_type][1][1]) + '\n'
            else:
                shortcut_injection = templates[prompt_type][0].format(shortcut, templates[prompt_type][1][0]) + '\n'
            prompt = prompt + shortcut_injection  # 后缀添加
            prompt = prompt + '\n' + templates[prompt_type][0].format(sen, '')
        if cfg.prompt_task == 3:  # 间接捷径注入，找到待预测样本的相同标记的shot并对其中的捷径进行替换
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    if label == 0:  # 替换标签相同的样本
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    if label == 1:  # 替换标签相同的样本
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            if cfg.del_shortcut:
                sen = replace_shortcut(myTokenizer, shortcut, sen)
            prompt = prompt + '\n' + templates[prompt_type][0].format(sen, '')
        if cfg.prompt_task == 3.5:  # 间接捷径利用，找到待预测样本的相反标记的shot并对其中的捷径进行替换，目的是鼓励LLMs利用捷径【对抗】
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    if label == 1:  # 替换标签相反的样本
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    if label == 0:  # 替换标签相反的样本
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            if cfg.del_shortcut:
                sen = replace_shortcut(myTokenizer, shortcut, sen)
            prompt = prompt + '\n' + templates[prompt_type][0].format(sen, '')
        if cfg.prompt_task == 4:  #
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    if label == 0:  # 替换样本
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    if label == 1:  # 替换样本
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            prompt = prompt + '\n' + templates[prompt_type][0].format(sen, '')
            # prompt = 'Do not pay attention to {}.'.format(shortcut) + '\n' + prompt
            prompt = 'Assume you are a robust model and do not make predictions based on {}.'.format(shortcut) + '\n' + prompt

        if cfg.prompt_task == 5:  # add keywords in the prompt
            current_keywords = []
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    current_keywords.extend(sentence2keywords[p_sen])
                    if label == 1:  # 标签相同样本替换
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    current_keywords.extend(sentence2keywords[p_sen])
                    if label == 0:
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            current_keywords.extend(sentence2keywords[sen])
            current_keywords = sorted(current_keywords, key=lambda a:a[1])
            current_keywords = [each[0] for each in current_keywords][:8]
            keyword_prompt = templates[prompt_type][0].format(' '.join(current_keywords), '') + '\n'
            prompt = keyword_prompt + prompt + '\n' + templates[prompt_type][0].format(sen, '')
        if cfg.prompt_task == 5.1:  # add random words in the prompt
            current_keywords = []
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    current_keywords.extend(random_word(myTokenizer, shortcut, p_sen))
                    if label == 0:  # 标签相同样本替换
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    current_keywords.extend(random_word(myTokenizer, shortcut, p_sen))
                    if label == 1:
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            current_keywords.extend(sentence2keywords[sen])
            # current_keywords = sorted(current_keywords, key=lambda a:a[1])
            current_keywords = [each[0] for each in current_keywords][:8]
            keyword_prompt = templates[prompt_type][0].format(' '.join(current_keywords), '') + '\n'
            prompt = keyword_prompt + prompt + '\n' + templates[prompt_type][0].format(sen, '')
        if cfg.prompt_task == 5.5:  # add keywords in the prompt by instruction
            current_keywords = []
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    current_keywords.extend(sentence2keywords[p_sen])
                    if label == 0:  # 标签相同样本替换
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    current_keywords.extend(sentence2keywords[p_sen])
                    if label == 1:
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            current_keywords.extend(sentence2keywords[sen])
            current_keywords = sorted(current_keywords, key=lambda a: a[1])
            current_keywords = [each[0] for each in current_keywords][:8]
            key_instruct = 'Please make predictions based on {}'.join(current_keywords)
            prompt = key_instruct + prompt + '\n' + templates[prompt_type][0].format(sen, '')
        if cfg.prompt_task == 6:  # CoT
            current_keywords = []
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['0'][k]
                if sen != p_sen:
                    current_keywords.extend(sentence2keywords[p_sen])
                    if label == 0:  # 标签相同样本替换
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    neg_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][0]))
                if len(neg_sens) == cfg.K:
                    break
            for k in range(20):
                p_sen = shortcut2sen[shortcut]['1'][k]
                if sen != p_sen:
                    current_keywords.extend(sentence2keywords[p_sen])
                    if label == 1:
                        p_sen = replace_shortcut(myTokenizer, shortcut, p_sen)
                    pos_sens.append(templates[prompt_type][0].format(p_sen, templates[prompt_type][1][1]))
                if len(pos_sens) == cfg.K:
                    break
            p_sens = [val for pair in zip(pos_sens, neg_sens) for val in pair]
            prompt = '\n'.join(p_sens)
            # current_keywords.extend(sentence2keywords[sen])
            current_keywords = sorted(current_keywords, key=lambda a: a[1])
            # current_keywords = [each[0] for each in current_keywords][:8]
            key_instruct = 'Let\'s think step by step. '
            prompt = key_instruct + prompt + '\n' + templates[prompt_type][0].format(sen, '')
        inputs.append(prompt)

    return inputs, labels, shortcuts, templates

def load_base_model_tokenizer(cfg):
    if 'gpt2' in cfg.LLM:
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True, padding_side='left')
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if 'opt' in cfg.LLM:
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True)
        tokenizer.padding_side = "left"

    if 'llama' in cfg.LLM.lower():
        tokenizer = LlamaTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True,
                                                   padding_side='left')
        # tokenizer.pad_token = tokenizer.eos_token    # add padding
        tokenizer.pad_token = ' '
    if 'gpt-neo' in cfg.LLM:
        tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True)
        tokenizer.pad_token = " "
        tokenizer.padding_side = "left"

    if '13b' in cfg.LLM or '6.7b' in cfg.LLM:  # 超大模型使用accelerate进行加载
        # with init_empty_weights():
        #     model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True,
        #                                                  torch_dtype=torch.float16)
        # # # The model weights are not tied.
        # # # Please use the `tie_weights` method before using the `infer_auto_device` function.
        # model.tie_weights()
        # # # 人为设置不同的device所分配的内存， devide:0分配得少一些方便后续推理，实际操作中不写也行
        # max_memory={2: "23GiB", 1: "23GiB", 0:"23GiB"}  # cpu辛苦你啦
        # device_map = infer_auto_device_map(model, max_memory=max_memory)    # 自动推断device_map
        # print(device_map)
        # # print(device_map)  # 可以看看模型不同层被分在了哪里
        # if 'llama' in cfg.LLM:
        #     model = load_checkpoint_and_dispatch(model, cfg.LLM_path + cfg.LLM,
        #                                          device_map='auto', offload_folder="offload",
        #                                          no_split_module_classes=["LlamaDecoderLayer"],
        #                                          offload_state_dict=True, dtype=torch.float16).half()
        # else:
        #     model = load_checkpoint_and_dispatch(model, cfg.LLM_path + cfg.LLM,
        #                                          device_map=device_map, offload_folder="offload",
        #                                          no_split_module_classes=[],
        #                                          offload_state_dict=True, dtype=torch.float16).half()    # 加载模型
        model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True,
                                                         torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                     device_map='auto').half()
    else:
        if 'gpt2' in cfg.LLM:
            tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True, padding_side='left')
            tokenizer.pad_token = " "
            # model = AutoModelForMultipleChoice.from_pretrained(cfg.LLM_path+cfg.LLM, trust_remote_code=True)
            model = GPT2LMHeadModel.from_pretrained(cfg.LLM_path+cfg.LLM, trust_remote_code=True)
        if 'opt' in cfg.LLM:
            # model = OPTForSequenceClassification.from_pretrained(cfg.LLM_path+cfg.LLM, trust_remote_code=True)
            tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True)
            model = OPTForCausalLM.from_pretrained(cfg.LLM_path+cfg.LLM, trust_remote_code=True)
            # if cfg.LLM == 'opt-6.7b':
            tokenizer.padding_side = "left"
        if 'gpt-neo' in cfg.LLM:
            tokenizer = AutoTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True)
            tokenizer.pad_token = " "
            tokenizer.padding_side = "left"
            model = AutoModelForCausalLM.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True)
        if 'llama' in cfg.LLM.lower():
            try:
                tokenizer = LlamaTokenizer.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True, padding_side='left')
                # tokenizer.pad_token = tokenizer.eos_token    # add padding
                tokenizer.pad_token = ' '
                model = LlamaForCausalLM.from_pretrained(cfg.LLM_path + cfg.LLM, trust_remote_code=True)
            except:
                # convert pth to HuggingFace format
                exit('请先使用transformers/models/llama/convert_llama_weights_to_hf.py进行参数转化！！！！')
                '''
                python /home/songrui/miniconda3/envs/T/lib/python3.8/site-packages/transformers/models/llama/convert_llama_weights_to_hf.py --input_dir /home/songrui/data/llama2-pth --model_size 7B --output_dir /home/songrui/data/llama2-7b
                '''
        # model = model.half().cuda()
        model = model.half().cuda()
    print("(*^_^*) model load finished!!!! ")
    model.eval()
    return model, tokenizer


import nltk

def find_phrase_positions(tokenized_sentence, tokenized_shortcut, tokenized_label, opp_tokenized_label):
    # print(tokenized_sentence)
    # print(tokenized_shortcut)
    # print(tokenized_label)
    # Find the positions of the phrases
    poss = []
    for i in range(len(tokenized_sentence)):
        if i + len(tokenized_shortcut) <= len(tokenized_sentence):
            if ' '.join(tokenized_sentence[i:i + len(tokenized_shortcut)]).lower() == ' '.join(tokenized_shortcut).lower():
                position_start = i
                position_end = i + len(tokenized_shortcut) - 1
                poss.append((position_start, position_end, 'shortcut'))
        if i + len(tokenized_label) <= len(tokenized_sentence):
            if ' '.join(tokenized_sentence[i:i + len(tokenized_label)]).lower() == ' '.join(tokenized_label).lower():
                position_start = i
                position_end = i + len(tokenized_label) - 1
                if tokenized_sentence[position_start-1] == ':':   # sure to be the label in the prompt
                    poss.append((position_start, position_end, 'label'))
        if i + len(opp_tokenized_label) <= len(tokenized_sentence):
            if ' '.join(tokenized_sentence[i:i + len(opp_tokenized_label)]).lower() == ' '.join(opp_tokenized_label).lower():
                position_start = i
                position_end = i + len(opp_tokenized_label) - 1
                if tokenized_sentence[position_start - 1] == ':':
                    poss.append((position_start, position_end, 'opp_label'))
    # 检索每一个样本中的捷径-标签对，注意有的情况下标签没有对应的捷径，首先获取待预测标签以及其对应的捷径
    # if not ("label" in poss[-1][2] and "shortcut" in poss[-2][2]):   # 必须保证label以及其对应的捷径之间存在信息流，不然无法衡量
    #     print(tokenized_shortcut, tokenized_sentence)
    #     print(poss)
    #     exit('data error!!! There is no shortcut to the label!!!')
    concurrent_pos = []
    label_poss, opp_label_poss, pred_poss = [], [], []
    last_label_idx = 0
    for i, pos in enumerate(poss):
        if "label" not in pos[2]:
            concurrent_pos.append(pos)
        else:
            if last_label_idx == 0:
                pos = pos + (0,)    # 获取context的开始位置
            else:
                pos = pos + (poss[last_label_idx][1]+1,)  # 上一个label的end位置+1作为当前label的context的开始
            last_label_idx = i
            concurrent_pos.append(pos)
            if "opp_label" == pos[2]:
                opp_label_poss.append(concurrent_pos)
            elif "label" == pos[2]:
                if i == len(poss)-1: # 说明是最后一个
                    pred_poss.append(concurrent_pos)
                else:
                    label_poss.append(concurrent_pos)
            concurrent_pos = []
    assert len(label_poss) == len(opp_label_poss) and len(pred_poss) == 1
    return label_poss, opp_label_poss, pred_poss


def get_proportion(cfg, saliency, label_saliency, label_poss, opp_label_poss, pred_poss):
    ''' 获取标签从上文捷径获取的信息流得分
    :param saliency:
    :param label_poss:
    :param opp_label_poss:
    :param pred_poss:
    :return:
    '''
    saliency = saliency.detach().clone().cpu()
    if len(saliency.shape) == 3:
        saliency = saliency.squeeze(0)
    saliency = torch.triu(saliency, diagonal=0)
    eye = torch.eye(saliency.size(0))
    saliency -= eye * saliency
    saliency = torch.nn.functional.normalize(saliency.float(), p=2, dim=0)
    torch.nan_to_num(saliency, nan=0.0)
    saliency = saliency.numpy()

    label_saliency = label_saliency.detach().clone().cpu()
    if len(label_saliency.shape) == 3:
        label_saliency = label_saliency.squeeze(0)
    label_saliency = torch.triu(label_saliency, diagonal=0)
    label_saliency -= eye * label_saliency
    label_saliency = torch.nn.functional.normalize(label_saliency.float(), p=2, dim=0)
    torch.nan_to_num(label_saliency, nan=0.0)
    label_saliency = label_saliency.numpy()

    # 计算每一个label以及其上文中所有捷径之间的信息流 以及pred从label聚合的信息流
    S_s2l, S_s2l_opp, S_s2pred, S_l2l, S_l2l_opp = [], [], 0, [], []  # 存放与待预测标签相同的样本中的捷径-标签信息得分以及相反标签的信息得分
    for i, label_poss in enumerate([label_poss, opp_label_poss]):
        for label_pos in label_poss:
            same_score, label_score = [], []
            if len(label_pos) == 1:  # 说明没有上文的shortcut
                if i == 0:
                    S_s2l.append(0)
                else:
                    S_s2l_opp.append(0)
            else:
                # 首先计算上文捷径的
                context_score = saliency[label_pos[-1][-1]:label_pos[-1][0], label_pos[-1][0]].sum()/(label_pos[-1][0]-label_pos[-1][-1])    # avg
                # context_score = saliency[label_pos[-1][-1]:label_pos[-1][0], label_pos[-1][0]].sum()
                for shortcut_pos in label_pos[:-1]:
                    shortcut_score = saliency[shortcut_pos[0]:shortcut_pos[1], label_pos[-1][0]].sum()
                    same_score.append(shortcut_score)
                same_score = sum(same_score)/len(same_score)/context_score    # 求占比
                if i == 0:
                    S_s2l.append(same_score)
                else:
                    S_s2l_opp.append(same_score)

            # 其次计算pred和label的
            label_score = label_saliency[label_pos[-1][0], pred_poss[0][-1][0]].sum()
            if i == 0:
                S_l2l.append(label_score)
            else:
                S_l2l_opp.append(label_score)
    # print(pred_poss)  # [[(112, 113, 'shortcut'), (137, 137, 'label', 108)]]
    context_score = label_saliency[pred_poss[0][-1][-1]:pred_poss[0][-1][0], pred_poss[0][-1][0]].sum()/(pred_poss[0][-1][0]-pred_poss[0][-1][-1])
    # context_score = label_saliency[:pred_poss[0][-1][0], pred_poss[0][-1][0]].sum()/pred_poss[0][-1][0]
    pred_score = []
    for shortcut_pos in pred_poss[0][:-1]:
        shortcut_score = label_saliency[shortcut_pos[0]:shortcut_pos[1], pred_poss[0][-1][0]].sum()  # shortcut是一个范围
        pred_score.append(shortcut_score)
    if len(pred_score) > 0:
        S_s2pred = sum(pred_score)/len(pred_score)/context_score
    else:
        S_s2pred = 0
    if math.isnan(S_s2pred):
        S_s2pred = 0
    S_s2l = sum(S_s2l)/len(S_s2l)
    S_s2l_opp = sum(S_s2l_opp)/len(S_s2l_opp)
    S_l2l = sum(S_l2l)/len(S_l2l)
    S_l2l_opp = sum(S_l2l_opp)/len(S_l2l_opp)
    if math.isnan(S_s2l):
        S_s2l = 0
    if math.isnan(S_s2l_opp):
        S_s2l_opp = 0

    # sum_label = sum(S_l2l)+sum(S_l2l_opp)
    # S_l2l = sum(S_l2l)/sum_label
    # S_l2l_opp = sum(S_l2l_opp)/sum_label
    return S_s2l, S_s2l_opp, S_s2pred, S_l2l, S_l2l_opp

