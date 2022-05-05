import argparse
from unittest.util import _MAX_LENGTH
import pandas as pd
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoModel, 
    BertModel,
    MODEL_MAPPING,
    CONFIG_MAPPING
)
import transformers
from unidecode import unidecode
from transformers import TrainingArguments
import numpy as np
from datasets import load_metric, load_dataset
from transformers import TrainingArguments, Trainer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import random
from tqdm import tqdm_notebook, tqdm
import copy
from tqdm.auto import tqdm
import json
import sys
from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
import math
from accelerate import Accelerator
from sklearn.metrics import r2_score, mean_squared_error
import logging


logger = logging.getLogger(__name__)
# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# print('device: {}'.format(device))

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--data_file", type = str, nargs = '+', help="A csv or a json file containing the all data, split to train and test model."
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--test_file", type=str, default=None, help="A csv or a json file containing the test data."
    )
    parser.add_argument(
        "--validation_strategy", type=str, default=None, help="cross validation or train-test validation"
    )
    parser.add_argument(
        "--k_fold", type=int, default=None, help="parameter for cross validation"  
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default='xlm-roberta-base',
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--optim_strategy",
        type=int,
        default=0,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_bert",
        type=float,
        default=5e-5,
        help="Initial learning rate for bert layer",
    )
    parser.add_argument(
        "--lr_fc",
        type=float,
        default=5e-5,
        help="Initial learning rate for fully connected layer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--freeze_layer_count", type=int, default=None, help="Freeze layer in bert model")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    # parser.add_argument(
    #     "--lr_scheduler_type",
    #     type=SchedulerType,
    #     default="linear",
    #     help="The scheduler type to use.",
    #     choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    # )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    # parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    # parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    # parser.add_argument(
    #     "--model_type",
    #     type=str,
    #     default=None,
    #     help="Model type to use if training from scratch.",
    #     choices=MODEL_TYPES,
    # )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=None,
        help="The maximum total input sequence length after tokenization. Sequences longer than this will be truncated.",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a checkpoint folder.",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to load in all available experiment trackers from the environment and use them for logging.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        help="for early stoping process",
    )
    args = parser.parse_args()

    return args
# FinancialPhraseBank, CausalityDetection, Lithuanian

class RegressionBasedBert(torch.nn.Module):
    def __init__(self, embedding_model):
            super(RegressionBasedBert, self).__init__()
            self.embedding_model = embedding_model
            self.fc = torch.nn.Sequential(
                        torch.nn.Linear(768, 1)
            )
#             self.fc = torch.nn.Sequential(
#                 torch.nn.Linear(768, 384),
#                 torch.nn.ReLU(),
#                 torch.nn.Linear(384, 1)
#             )
    def forward(self, ids, mask):
        x = self.embedding_model(ids, mask)['last_hidden_state'][:, 0, :]
        x = self.fc(x)
        return x


def main():
    args = parse_args()
    # print(args.data_file)
    
    accelerator = Accelerator(log_with="all", logging_dir=args.output_dir) if args.with_tracking else Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
#         datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
#         datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.wait_for_everyone()
    
    device = accelerator.device
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device(device)
#     print('device: {}'.format(device))
    
    org_all_data = []
    for data_file in args.data_file:
        with open(data_file, 'r', encoding='utf-8') as f:
            x = json.load(f)
            for i in list(x.keys()):
                org_all_data.append(x[i])
    
    """
    gen data: [sentence, sentiment score]
    """
    all_data = []
    for i in org_all_data:
        # print(i)
        sent = i['sentence']
        sentiment_score = float(i['info'][0]['sentiment_score'])
        # print(type(sentiment_score))
        # sys.exit()
        all_data.append([sent, sentiment_score])
    
    # print(len(all_data))
    # print(all_data[:10])
    # train_data = all_data[:1000]
    # test_data = all_data[1000:1100]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    result_file = '_'.join(['Evaluation', 'regression-CustomModel', '1-fc', 'optim_strategy', str(args.optim_strategy), args.model_name_or_path, 'task.txt'])
    f = open(result_file, 'w')
    
    """
    data: List -> [sent: string, sentiment_score: float]
    """
    def GenericDataLoader(data, batch_size):
        ids = []
        masks = []
        for sample in data:
            sent = sample[0]
            inputs = tokenizer(sent, return_tensors="np", padding='max_length', truncation=True, max_length=args.max_seq_length)
            encoded_sent = inputs['input_ids'][0]
            mask = inputs['attention_mask'][0]
            ids.append(encoded_sent)
            masks.append(mask)
        inputs = torch.tensor(np.array(ids))
        masks = torch.tensor(np.array(masks))
        labels = torch.tensor(np.array([i[1] for i in data], dtype='f'))
        # print(type(labels))
        # sys.exit()
        data = TensorDataset(inputs, masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def train(model, epochs, loss_fn, optimizer, lr_scheduler, train_dataloader, patience = 3):
        the_last_loss = 1e10
        trigger_times = 0
        accelerator.print('training')
        for epoch in range(epochs):
            losses = []
            accelerator.print('\n')
            model.train()
            total_loss = 0
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                
                outputs = model(b_input_ids, b_input_mask)
                outputs = outputs.squeeze(1)
                loss = loss_fn(outputs, b_labels)
        
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                losses.append(accelerator.gather(loss))
                
            lr_scheduler.step()
            the_current_loss = torch.cat(losses).mean()
            accelerator.print('Epoch: {}, Total loss: {:.2f}'.format(epoch, the_current_loss))
#             print(the_current_loss)
#             sys.exit()
            # Early stopping
#             
            
#             f.writelines('Epoch: {}, Total loss: {:.2f}\n'.format(epoch, the_current_loss))

            # print('The current loss:', the_current_loss)
#             the_current_loss = total_loss
            if the_current_loss > the_last_loss:
                trigger_times += 1
                accelerator.print('trigger times:', trigger_times)
#                 f.writelines('trigger times: {}\n'.format(trigger_times))

                if trigger_times >= patience:
                    accelerator.print('Early stopping!\nStart to test process.')
#                     f.writelines('Early stopping!\nStart to test process.\n')
                    return model

            else:
                accelerator.print('trigger times: 0')
#                 f.writelines('trigger times: 0\n')
                trigger_times = 0

            the_last_loss = the_current_loss

        return model
        

    def evaluation(model, num_test_sample, test_dataloader):
        targets, preds = [], []
        accelerator.print('evaluation')
        model.eval()
        targets, preds = [], []
        for step, batch in tqdm(enumerate(test_dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            with torch.no_grad():
                outputs = model(b_input_ids, b_input_mask)
            logits = outputs.squeeze(1)
            labels, outputs = accelerator.gather([b_labels, logits])
            targets.append(labels)
            preds.append(outputs)
        
        targets = torch.cat(targets)[:num_test_sample]
        preds = torch.cat(preds)[:num_test_sample]
        
#         mse = abs(mean_squared_error(targets.cpu().numpy(), preds.cpu().numpy()))
#         r2 = r2_score(targets.cpu().numpy(), preds.cpu().numpy())
        mse = abs(mean_squared_error(targets, preds))
        r2 = r2_score(targets, preds)
        
        accelerator.print('mse: {:.4f}'.format(mse))
        accelerator.print('r2: {:4f}'.format(r2))
        f.writelines('mse: {:.4f}\n'.format(mse))
        f.writelines('r2: {:.4f}\n'.format(r2))

        return mse, r2

    all_data = pd.DataFrame(all_data)
    kf = KFold(n_splits=args.k_fold, random_state=231, shuffle=True)
    MSE, R2 = [], []
    count = 1
    for train_index, val_index in kf.split(all_data):
        accelerator.print('=======fold: {}========='.format(count))
        f.writelines('=======fold: {}=========\n'.format(count))
        train_df = all_data.iloc[train_index]
        test_df = all_data.iloc[val_index]

        train_data = train_df.values.tolist()
        test_data = test_df.values.tolist()

        # prepare dataloader
        num_test_sample = len(test_data)
        train_dataloader = GenericDataLoader(train_data, args.per_device_train_batch_size) 
        test_dataloader = GenericDataLoader(test_data, args.per_device_eval_batch_size)
        
        # define model
        embedding_model = AutoModel.from_pretrained(args.model_name_or_path).to(device)
        if args.freeze_layer_count:
                # We freeze here the embeddings of the model
                for param in embedding_model.embeddings.parameters():
                    param.requires_grad = False

                if args.freeze_layer_count != -1:
                    # if freeze_layer_count == -1, we only freeze the embedding layer
                    # otherwise we freeze the first `freeze_layer_count` encoder layers
                    for layer in embedding_model.encoder.layer[:args.freeze_layer_count]:
                        for param in layer.parameters():
                            param.requires_grad = False
        model = RegressionBasedBert(embedding_model).to(device)
        
#         model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=1)
        
        # loss function
        loss_fn = torch.nn.MSELoss()
        
        # optimizer
        if args.optim_strategy == 0:
            bert_params = model.embedding_model.encoder.named_parameters()
            classifier_params = model.fc.named_parameters()
            grouped_params = [
                {'params': [p for n,p in bert_params if p.requires_grad==True], 'lr': args.lr_bert},
                {'params': [p for n,p in classifier_params if p.requires_grad==True], 'lr': args.lr_fc}
            ] 
            optimizer = torch.optim.AdamW(grouped_params)
        elif args.optim_strategy == 1:
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate) 
        
        # scheduler
        num_training_steps = args.num_train_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
        
        # prepare for accelerate training
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, test_dataloader, lr_scheduler
                )
        
        # train model
        model = train(model, args.num_train_epochs, loss_fn, optimizer, lr_scheduler, train_dataloader, args.patience)
        
        # evaluation model
        mse, r2 = evaluation(model, num_test_sample, test_dataloader) 
        MSE.append(mse)
        R2.append(r2)
        accelerator.print('Finish fold: {}\n\n'.format(count))
#         f.writelines('Finish fold: {}\n\n\n'.format(count))
        count += 1
    accelerator.print('\n\n\n\n')
    accelerator.print('Final evaluation')
    accelerator.print('Final mse: {:.4f}'.format(sum(MSE)/args.k_fold))
    accelerator.print('Final r2: {:.4f}'.format(sum(R2)/args.k_fold))
    f.writelines('\n\n\n\n\n')
    f.writelines('Final evaluation\n')
    f.writelines('Average mse: {:.2f}\n'.format(sum(MSE)/args.k_fold))
    f.writelines('Average r2: {:.2f}\n'.format(sum(R2)/args.k_fold))
    f.close()
    

if __name__ == "__main__":
    main()
