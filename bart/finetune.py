# altered version of https://github.com/SunnyGJing/t5-pegasus-chinese
import os
import re
import torch
import argparse
import random
from torch.utils.data import DataLoader, Dataset
from torch._six import container_abcs, string_classes, int_classes
from transformers import BertTokenizer, BartForConditionalGeneration
import numpy as np

def load_data(filename):
    """加载数据
    单条格式：(标题, 正文)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f.readlines():
            cur = l.strip().split('\t')
            if len(cur) == 2:
                title, content = cur[0], cur[1]
                D.append((title, content))
            elif len(cur) == 1:
                content = cur[0]
                D.append(content)
    return D



class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def create_data(data, tokenizer, max_len=512, term='train'):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag = [], True
    for title, content in data:
        text_ids = tokenizer.encode(content, max_length=max_len, truncation='only_first')
        if flag:
            flag = False
            print("content(source): ", content, len(content), max_len)
            #print(text_ids)
            print("title(target): ", title, len(content), max_len)

        if term == 'train':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'decoder_input_ids': summary_ids,
                        'attention_mask': [1] * len(text_ids),
                        'decoder_attention_mask': [1] * len(summary_ids)
                       }
        
        elif term == 'dev':
            summary_ids = tokenizer.encode(title, max_length=max_len, truncation='only_first')
            features = {'input_ids': text_ids,
                        'attention_mask': [1] * len(text_ids),
                        #'title': title,
                        'decoder_input_ids': summary_ids,
                        'decoder_attention_mask': [1] * len(summary_ids)
                       }
            
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


def prepare_data(args, data_path, tokenizer, term='train'):
    """准备batch数据
    """
    data = load_data(data_path)
    data = create_data(data, tokenizer, args.max_len, term)
    data = KeyDataset(data)
    data = DataLoader(data, batch_size=args.batch_size, collate_fn=default_collate)
    return data

def compute_valid_loss(dev_data):
    valid_loss = 0
    for i, cur in enumerate(dev_data):
        cur = {k: v.to(device) for k, v in cur.items()}
        prob = model(**cur)[0]
        mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
        prob = prob[:, :-1]
        prob = prob.reshape((-1, prob.size(-1)))[mask]
        labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(prob, labels)
        valid_loss += loss.item()
    valid_loss = valid_loss / len(dev_data)
    print("valid_loss: ", valid_loss)



def train_model(model, adam, train_data, dev_data, tokenizer, device, args):
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        start_epoch = 0
    else:
        model_name_list = os.listdir(args.save_dir)
        last_model_name = model_name_list[-1]
        checkpoint = torch.load(f'{args.save_dir}/{last_model_name}', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        adam.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1   
        print(f"loaded model: {args.save_dir}/{last_model_name}")
    best = 0
    for epoch in range(start_epoch, args.num_epoch):
        print("epoch: ", epoch)
        model.train()
        for i, cur in enumerate(train_data):
            cur = {k: v.to(device) for k, v in cur.items()}
            prob = model(**cur)[0]
            mask = cur['decoder_attention_mask'][:, 1:].reshape(-1).bool()
            prob = prob[:, :-1]
            prob = prob.reshape((-1, prob.size(-1)))[mask]
            labels = cur['decoder_input_ids'][:, 1:].reshape(-1)[mask]
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(prob, labels)
            if i % 1000 == 0:
                print(i, loss.item())
            loss.backward()
            adam.step()
            adam.zero_grad()

        # 测试
        model.eval()
        #gens = []
        #summaries = []
        
        compute_valid_loss(dev_data)
        
        if args.data_parallel and torch.cuda.is_available():
            #torch.save(model.module, 'cgec_t5_model')
            torch.save(model.module, f'{args.save_dir}/cgec_epoch_{str(epoch)}')
        else:
            #torch.save(model, f'{args.save_dir}/cgec_epoch_{str(epoch)}')
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': adam.state_dict(),
            'loss': loss,}, f'{args.save_dir}/cgec_epoch_{str(epoch)}')


def init_argument():
    parser = argparse.ArgumentParser(description='cpt_base')
    parser.add_argument('--train_data', default='./data/train.nospac.tsv')
    parser.add_argument('--dev_data', default='./data/dev.nospac.tsv')
    parser.add_argument('--pretrain_model', default='/work/wanghongfei/cpt_base')
    parser.add_argument('--save_dir', default='./saved_model')
    parser.add_argument('--num_epoch', default=30, type=int, help='number of epoch')
    parser.add_argument('--batch_size', default=16, type=int, help='batch size')
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--data_parallel', default=False)
    parser.add_argument('--max_len', default=512, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=512, help='max length of generated text')
    parser.add_argument('--seed', default=1, type=int, help='random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    # step 1. init argument
    args = init_argument()

    # step 2. prepare training data and validation data
    tokenizer = BertTokenizer.from_pretrained(args.pretrain_model)
    train_data = prepare_data(args, args.train_data, tokenizer, term='train')
    dev_data = prepare_data(args, args.dev_data, tokenizer, term='dev')
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # step 3. load pretrain model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BartForConditionalGeneration.from_pretrained(args.pretrain_model).to(device)
    if args.data_parallel and torch.cuda.is_available():
        device_ids = range(torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    # step 4. finetune
    adam = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_model(model, adam, train_data, dev_data, tokenizer, device, args)