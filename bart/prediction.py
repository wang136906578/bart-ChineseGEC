# altered version of https://github.com/SunnyGJing/t5-pegasus-chinese
from transformers import BertTokenizer, BatchEncoding, BartForConditionalGeneration
from torch._six import container_abcs, string_classes, int_classes
import torch
from torch.utils.data import DataLoader, Dataset
import re
import os
import csv
import argparse
from tqdm.auto import tqdm
from multiprocessing import Pool, Process
import pandas as pd
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(filename):
    """加载数据
    单条格式：(正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            content = l.strip()
            D.append(content)
    return D
    
    

class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    
def create_data(data, tokenizer, max_len):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag = [], True
    for content in data:
        text_ids = tokenizer.encode(content, max_length=max_len,
                                    truncation='only_first')

        if flag:
            flag = False
            print(content)

        features = {'input_ids': text_ids,
                    'attention_mask': [1] * len(text_ids),
                    'raw_data': content}
        ret.append(features)
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out).to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)

        return default_collate([default_collate(elem) for elem in batch])

    raise TypeError(default_collate_err_msg_format.format(elem_type))
    

def prepare_data(args, tokenizer):
    """准备batch数据
    """
    test_data = load_data(args.test_data)
    test_data = create_data(test_data, tokenizer, args.max_len)
    test_data = KeyDataset(test_data)
    test_data = DataLoader(test_data, batch_size=args.batch_size, collate_fn=default_collate)
    return test_data


def generate(test_data, model, tokenizer, args):
    with open(args.result_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        model.eval()
        for feature in tqdm(test_data):
            raw_data = feature['raw_data']
            content = {k : v for k, v in feature.items() if k != 'raw_data'} 
            gen = model.generate(max_length=args.max_len_generate,
                                eos_token_id=tokenizer.sep_token_id,
                                decoder_start_token_id=tokenizer.cls_token_id,
                                **content)
            gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
            gen = [item.replace(' ', '') for item in gen]
            writer.writerows(zip(gen, raw_data))
    print('Done!')


def generate_multiprocess(feature):
    """多进程
    """
    model.eval()
    raw_data = feature['raw_data']
    content = {k: v for k, v in feature.items() if k != 'raw_data'}
    gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
    gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
    results = ["{}\t{}".format(x.replace(' ', ''), y) for x, y in zip(gen, raw_data)]
    return results


def init_argument():
    parser = argparse.ArgumentParser(description='bart')
    parser.add_argument('--test_data', default='./data/test.tsv')
    parser.add_argument('--result_file', default='./output/result.txt')
    parser.add_argument('--pretrain_model', default='')    
    parser.add_argument('--model', default='')
    parser.add_argument('--batch_size', default=32, help='batch size')
    parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=512, help='max length of generated text')
    parser.add_argument('--use_multiprocess', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # step 1. init argument
    args = init_argument()

    # step 2. prepare test data
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
    test_data = prepare_data(args, tokenizer)
    
    # step 3. load finetuned model
    model = BartForConditionalGeneration \
                .from_pretrained(args.pretrain_model).to(device)
    checkpoint = torch.load(args.model, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    # step 4. predict
    res = []
    if args.use_multiprocess and device == 'cpu':
        print('Parent process %s.' % os.getpid())
        p = Pool(2)
        res = p.map_async(generate_multiprocess, test_data, chunksize=2).get()
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        res = pd.DataFrame([item for batch in res for item in batch])
        res.to_csv(args.result_file, index=False, header=False, encoding='utf-8')
        print('Done!')
    else:
        generate(test_data, model, tokenizer, args)
