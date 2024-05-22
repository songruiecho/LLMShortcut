'''
saliency score for each element of the attention matrix, according to
LabelWords are Anchors: An Information Flow Perspective for Understanding In-Context Learning, EMNLP2023
The code references "https://github.com/lancopku/label-words-are-anchors/"
'''
import copy
import os
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
from utils import *
from torch.nn.utils.rnn import pad_sequence

# 0. Step zero, load model&tokenizer
def cal_attr(cfg):
    model, tokenizer = load_base_model_tokenizer(cfg)
    S_s2ls, S_s2l_opps, S_s2preds, S_l2ls, S_l2l_opps = [], [], [], [], []
    # 1. Step one, generate batch data, the input labels must be like:
    # labels = inputs["input_ids"].clone()
    # labels[:, :-1] = labels[:, 1:].clone()
    # labels[:, -1] = -100  # Mask out the last token for causal LM training
    # 定义损失函数，包括OPT损失和其他损失
    criterion = torch.nn.CrossEntropyLoss()  # 这里用交叉熵损失作为示例
    # 定义优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    p_types = [0,1,2,3]
    for p_type in p_types:
        inputs, labels, shortcuts, templates = generate_prompt(cfg, p_type)           # prompt_type needs to be modified according to the test

        label_tok_lens = {}
        for w in "negative positive good bad normal toxic".split():    # test the label ids
            label_toks = tokenizer.tokenize(w)
            label_tok_lens[w] = len(label_toks)

        # model.zero_grad()
        model.cuda()

        opp_labels = {
            'negative': 'positive',
            'positive': 'negative',
            'good': 'bad',
            'bad': 'good',
            'normal': 'toxic',
            'toxic': 'normal'
        }

        for i in tqdm(range(0, len(inputs), cfg.batch)):
            try:
                prompts = inputs[i:i+cfg.batch]   #
                batch_labels = labels[i:i+cfg.batch]
                batch_shortcuts = shortcuts[i:i+cfg.batch]
                label_strs = [templates[p_type][1][each] for each in batch_labels]
                # be encoded differently whether it is at the beginning of the sentence (without space) or not, so a Uniform Space are added
                prompts = [' ' + prompt + label_str for prompt, label_str in zip(prompts, label_strs)]
                tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]
                max_len = max(len(sentence) for sentence in tokenized_prompts)
                # left padding
                padded_prompts = [[tokenizer.pad_token_id] * (max_len - len(sentence)) + sentence for sentence in tokenized_prompts]
                # mask last label_tok_lens tokens
                lens = [label_tok_lens[label_str] for label_str in label_strs]
                masked_labels = []
                for i, each in enumerate(copy.deepcopy(padded_prompts)):
                    l = lens[i]
                    each[-l:] = [-100]*l   # mask the label
                    masked_labels.append(each)
                icl_labels = torch.LongTensor(masked_labels).to('cuda')
                t_padded_prompts = torch.LongTensor(padded_prompts).to('cuda')
                # 2. Step two, train the model with only one backpropagation
                # outputs = model(t_padded_prompts, labels=icl_labels, output_attentions=True)
                optimizer.zero_grad()
                outputs = model(t_padded_prompts, output_attentions=True)
                for each in outputs.attentions:
                    each.retain_grad()   # keep grad for leaf tensor
                logits_flat = outputs.logits.view(-1, outputs.logits.shape[-1])
                target_flat = icl_labels.view(-1)
                loss = criterion(logits_flat, target_flat)
                loss.backward()  # backward only once
                optimizer.step()
                attentions, label_attentions = outputs.attentions[:5], outputs.attentions[-5:]   # top-5 layers and last-5 layers
                grads = []
                for att in attentions:
                    assert len(att.grad.shape) == 4        # batch_size, n_heads, seq_len, seq_len
                    grad = att.grad.sum(1)
                    grad = abs(grad)
                    grads.append(grad)
                saliency = sum(grads)   # sum -> batch_size, seq_len, seq_len
                grads = []
                for att in label_attentions:
                    assert len(att.grad.shape) == 4        # batch_size, n_heads, seq_len, seq_len
                    grad = att.grad.sum(1)
                    grad = abs(grad)
                    grads.append(grad)
                label_saliency = sum(grads)  # sum -> batch_size, seq_len, seq_len

                # locate shortcut pos [start,end] and label pos [start, end]
                for j, tokenized_sentence in enumerate(padded_prompts):
                    try:
                        tokenized_sentence = tokenizer.convert_ids_to_tokens(tokenized_sentence)    # tokens needed
                        tokenized_shortcut = tokenizer.tokenize(' '+batch_shortcuts[j])
                        if 'llama' in cfg.LLM:
                            tokenized_label = tokenizer.tokenize(':'+label_strs[j])[1:]  # label str do not add space, Input:label without space
                            opp_tokenized_label = tokenizer.tokenize(':'+opp_labels[label_strs[j]])[1:]  # opposite label
                        else:
                            tokenized_label = tokenizer.tokenize(label_strs[j])   # label str do not add space, Input:label without space
                            opp_tokenized_label = tokenizer.tokenize(opp_labels[label_strs[j]])   # opposite label
                        # print(len(tokenized_sentence), saliency.shape)
                        assert len(tokenized_sentence) == saliency.shape[1]   # √
                        # get shortcut and label location
                        label_poss, opp_label_poss, pred_poss = find_phrase_positions(tokenized_sentence, tokenized_shortcut, tokenized_label, opp_tokenized_label)
                        # print(label_poss, opp_label_poss, pred_poss)
                        S_s2l, S_s2l_opp, S_s2pred, S_l2l, S_l2l_opp = get_proportion(cfg, saliency[j,:,:], label_saliency[j,:,:], label_poss, opp_label_poss, pred_poss)   # 信息流得分
                        S_s2ls.append(S_s2l)
                        S_s2l_opps.append(S_s2l_opp)
                        S_s2preds.append(S_s2pred)
                        S_l2ls.append(S_l2l)
                        S_l2l_opps.append(S_l2l_opp)
                    except:
                        print('score error!')
                        continue
            except:
                traceback.print_exc()
                continue
        #
    print(sum(S_s2ls)/len(inputs)/len(p_types), sum(S_s2l_opps)/len(inputs)/len(p_types),
          sum(S_s2preds)/len(inputs)/len(p_types), sum(S_l2ls)/sum(S_l2l_opps))
# 4. Step four, find the position of the word corresponding to the shortcut and label from $I_l$, and calculate the proportion of shortcut information.
# Note that the flow of information in a generative model (LLaMA, GPT, and OPT) is one-way.

if __name__ == '__main__':
    for model in ['opt-2.7b']:
        cfg = Config()
        cfg.LLM = model
        cfg.batch = 4
        for task in ['emotion']:
            for K in [4]:
                for prompt in [5]:
                    print('=========={}={}={}={}=================='.format(model, task, K, prompt))
                    cfg.K = K
                    cfg.base_task = task
                    cfg.prompt_task = prompt
                    cal_attr(cfg)