'''
构建benchmark.csv的时候，需要对每一捷径对应的20个示例做一个语义丰富度的评估，具体就是说与其它19个样本最不像的样本要排到前边，
这一步是为了提升k值较小情况下的示例的多样性。经过这一步把样本的数量过滤到10，这是因为LLMs在生成的时候总是重复话来回说。

根据benchmark.csv中的数据对LLMs进行测评
Step 0. 看每一个LLMs在不使用任何提示样例下的纯zero-shot的预测结果。
Step 1. 看LLMs在直接捷径引导的情况下每一个样本的预测结果，这里为捷径赋予的标签应该是测试样本的相反标签。
    [prompt]
    Choose whether the emotion of the following sentence is positive or negative.
    Anxiously \n positive
    anxiously, she waited for news of her missing. \n ?
Step 2. 看LLMs在上下文学习过程中的捷径学习效果
    [prompt]
    Choose whether the emotion of the following sentence is positive or negative.
    on friday, i embrace the beauty of uncertainty, excitedly anticipating.  \n positive
    [测试是一个语义相反的样本]
    friday's child is full of woe, that's what they say. \n  ?

这一步的测试过程需要将示例样本数量从k=[1,10]都纳入考虑，然后判断预测结果是否正确，最后对于每一个k，都要产生一个对应的总体的准确率，
然后横向比较不同k值下LLMs的准确性。
最终的呈现形式是为每一个LLM都生成一个折线图k=[1,10]，折线图的准确率表示在有捷径情况下对目标样本的预测能力。
理想情况下这个折线图是下降趋势的。
'''
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1,0,2,3'
import utils
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, AutoModelForMultipleChoice, AutoModelForCausalLM
from transformers import OPTForCausalLM, OPTForSequenceClassification
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaForSequenceClassification
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd
from transformers import pipeline
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import zhconv
from accelerate import Accelerator, infer_auto_device_map, init_empty_weights, load_checkpoint_and_dispatch
import nltk
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize, MWETokenizer
from nltk.corpus import words
import torch.multiprocessing as mp
import utils as myutils


# load config
cfg = myutils.Config()

def get_batch(p_type):
    '''
    :param inputs: 经过prompt生成之后句子
    :param labels: 每个样本对应的label
    :param shortcuts: 每个样本对应的希望模型注意的捷径
    :return:
    '''
    inputs, labels, shortcuts, templates = myutils.generate_prompt(cfg, p_type)
    batches = []
    for i in range(0, len(inputs), cfg.batch):
        batch = inputs[i:i + cfg.batch]
        batch_labels = labels[i:i + cfg.batch]  # 标签
        batch_shortcuts = shortcuts[i:i + cfg.batch]  # 标签
        batches.append([batch, batch_labels, batch_shortcuts])
    print('process datasets ({}) with batch ({}) ......'.format(len(inputs), len(batches)))
    return batches, templates

def LLM_test(cfg):
    print('load LLM {}......'.format(cfg.LLM))
    # 多次acc求平均值避免LLMs幻觉产生负面影响
    model, tokenizer = utils.load_base_model_tokenizer(cfg)
    accs = []
    for p_type in [0,1,2,3]:
        batches, templates = get_batch(p_type)
        # model tests
        preds, labels = [], []
        for batch in tqdm(batches):
            labels.extend(batch[1])
            inputs = tokenizer(batch[0], return_tensors="pt", padding=True)
            input_ids = inputs.input_ids
            with torch.no_grad():
                if "gpt" not in cfg.LLM:
                    generate_ids = model.generate(input_ids.cuda(), max_new_tokens=3)
                else:
                    # gpt\gpt-neo需要attention mask
                    generate_ids = model.generate(input_ids.cuda(), attention_mask=inputs.attention_mask.cuda(), pad_token_id=0, max_new_tokens=3)
            output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for out, inp in zip(output, batch[0]):
                out = out.replace(inp, '').lower()
                if templates[p_type][1][1] in out.lower():
                    preds.append(1)
                else:
                    if templates[p_type][1][0] in out.lower():
                        preds.append(0)
                    else:
                        preds.append(-1)
        real_preds = []
        for pred, l in zip(preds, labels):
            if pred != -1:
                real_preds.append(pred)
            else:
                real_preds.append(1-l)    # 取反
        preds = real_preds
        # cal acc
        acc = accuracy_score(labels, preds)
        print('acc on {} of templete {} is {}'.format(cfg.LLM, p_type, acc))
        # matrix = confusion_matrix(labels, preds)
        # print('confusion_matrix on {} is {}'.format(cfg.LLM, matrix))
        accs.append(acc)
    torch.cuda.synchronize()
    print('final acc on {} of is {}'.format(cfg.LLM, sum(accs)/len(accs)))

if __name__ == '__main__':
    for model in ['gpt-neo-1.3b', 'gpt-neo-2.7b', 'opt-1.3b', 'opt-2.7b', 'open-llama-3b']:
        cfg = myutils.Config()
        cfg.LLM = model
        if '1.3' in model:
            cfg.batch = 32
        else:
            cfg.batch = 16
        # cfg.batch = 8
        for task in ['toxicity', 'emotion']:
            for prompt in [1]:
                cfg.base_task = task
                cfg.prompt_task = prompt
                for K in [4]:
                    cfg.K = K
                    print('==================={}={}={}======================'.format(model,task,prompt))
                    LLM_test(cfg)