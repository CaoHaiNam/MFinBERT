import argparse
from unittest.util import _MAX_LENGTH
import pandas as pd
import transformers
from transformers import (
    AutoModelForSequenceClassification, 
    AutoTokenizer,
    AutoConfig,
    AutoModel, 
    BertModel,
    MODEL_MAPPING,
    CONFIG_MAPPING
)
import logging   
from unidecode import unidecode
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from sklearn.metrics import f1_score, accuracy_score
import random
import copy
from tqdm import tqdm as tqdm
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from prettytable import PrettyTable
from accelerate import Accelerator
from transformers.utils.versions import require_version
from datasets import load_metric
import accelerate
import sys
import wandb
from transformers.utils import check_min_version, get_full_repo_name, is_offline_mode, send_example_telemetry

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a Masked Language Modeling task")
    parser.add_argument(
        "--dataset", type=str, required=True, help="dataset name, each dataset is read by different way. Have 3 task: FinancialPhraseBank, CausalityDetection, Lithuanian" 
    )
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
        "--num_labels", type=int, required=True, help="number of labels"
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size"
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
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial common learning rate for model",
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
    parser.add_argument(
        "--state",
        type=int,
        default=None,
        help="train full or test model flow",
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
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
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
        "--metrics",
        type=str,
        help=""
    )
    args = parser.parse_args()

    return args
# FinancialPhraseBank, CausalityDetection, Lithuanian

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    
#     F1_score = f1_score(pred_flat, labels_flat, average='macro')
    
    return accuracy_score(pred_flat, labels_flat), 
    
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def main():
    args = parse_args()

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will pick up all supported trackers in the environment
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
#     print(1)
#     sys.exit()
#     try:
#         device = accelerator.device
#     except:
#         device = 'cpu'

    accelerator.wait_for_everyone()
    
    device = accelerator.device
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device(device)
#     print('device: {}'.format(device))
#     sys.exit()
    
    if args.dataset=='FinancialPhraseBank':
        k_fold = args.k_fold
        label2index = {'positive': 0, 'negative': 1, 'neutral': 2}
        data = []
        datafile = args.data_file[0]
        with open(datafile, encoding='ISO-8859-1') as f:
            for line in f:
                line = line.strip('\n').split('@')
                data.append([line[0], label2index[line[1]]])

    elif args.dataset=='CausalityDetection':
        with open(args.train_file, 'r', encoding='utf-8') as fp:
            ref_csv = fp.readlines()
        train_data = []
        for i in ref_csv[1:]:
            i = i.strip('\n').strip().split(';')
            train_data.append([" ".join(i[1].split()), int(i[2])])
        with open(args.test_file, 'r', encoding='utf-8') as fp:
                    ref_csv = fp.readlines()
        test_data = []
        for i in ref_csv[1:]:
            i = i.strip('\n').strip().split(';')
            test_data.append([" ".join(i[1].split()), int(i[2])])

    else:
        k_fold = args.k_fold 
        label2index = {'POS': 0, 'NEG': 1, 'NEU': 2}
        df = pd.read_csv(args.data_file[0], sep=';', header=None, engine='python', skiprows=1, names = ["Class","Text"], encoding='unicode_escape')
        data = []
        for i in range(len(df)):
            text, cl = unidecode(df['Text'][i]), df['Class'][i]
            data.append([text, label2index[cl]])

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)      
    num_epochs = args.num_train_epochs
    
    result_file = '_'.join(['Evaluation', 'dataset', args.dataset, args.model_name_or_path, 'task.txt'])
    f = open(result_file, 'w')
    
    accelerator.print('Language model: {}'.format(args.model_name_or_path))
    accelerator.print('Dataset: {}'.format(args.dataset))

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
        labels = torch.tensor(np.array([i[1] for i in data]))
        data = TensorDataset(inputs, masks, labels)
        sampler = SequentialSampler(data)
        dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
        return dataloader

    def train(model, wandb, epochs, loss_fn, optimizer, lr_scheduler, validate_strategy, num_val_sample, train_dataloader, val_dataloader=None):
        best_acc = 0
        best_model = model
        for epoch in range(epochs):
            total_loss = 0
            model.train()
            
            label_list, pred_list = [], []
            for step, batch in tqdm(enumerate(train_dataloader)):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_labels = batch[2].to(device)

                model.zero_grad()
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask, 
                                labels=b_labels)
                logits = outputs[1]
#                 accelerator.print(logits)
#                 accelerator.print(b_labels)
                loss = loss_fn(logits, b_labels)
                wandb.log({'train_loss_per_step': loss})
                #total_loss += loss.item()
        
                accelerator.backward(loss)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            lr_scheduler.step()
            
            avg_train_loss = total_loss / len(train_dataloader)
            accelerator.print("Average training loss: {0:.4f}".format(avg_train_loss))
            wandb.log({'avg_loss': avg_train_loss})
            
            accelerator.print("Running validation")

            model.eval()
            targets, preds = [], []
            for batch in tqdm(val_dataloader):

                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
#                     logits = model(b_input_ids, b_input_mask)
                    outputs = model(b_input_ids, 
                                    token_type_ids=None, 
                                    attention_mask=b_input_mask)
                logits = outputs[0]
                logits = torch.argmax(logits, dim=1)
#                 outputs = logits.argmax(dim=-1)
                outputs, labels = accelerator.gather([logits, b_labels])
                
                targets.append(labels)
                preds.append(outputs)
                
            targets = torch.cat(targets)[:num_val_sample]
            preds = torch.cat(preds)[:num_val_sample]
            
            cur_acc =  accuracy_score(targets, preds)
            wandb.log({'val_accuracy': cur_acc})
            
#             eval_metric = metric.compute()
#             Use accelerator.print to print only on the main process.

#             accelerator.print("epoch: {}, validate accuracy: {:.4f}".format(epoch, cur_acc))  
    
#             cur_acc = eval_metric['accuracy']

            if best_acc < cur_acc:
                best_model = copy.deepcopy(model)
                best_acc = cur_acc
                
#                 if (epoch+1)%5==0:
#                     torch.save(best_model.state_dict(), '-'.join([args.dataset, str(epoch), 'epoch']))
#                 print(classification_report(label_list, pred_list, zero_division=0, digits=4))
#                 f.writelines(classification_report(label_list, pred_list, zero_division=0)+'\n')
        
        return best_model

    def evaluation(model, test_dataloader, num_test_sample):
#         print('=========Final evaluation==========')
#         f.writelines('=========Final evaluation=========='+'\n')

        model.eval()
#         metric = load_metric("accuracy")
        targets, preds = [], []
        for batch in tqdm(test_dataloader):

            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            with torch.no_grad():
#             logits = logits.detach().cpu().numpy()
#             label_ids = b_labels.to('cpu').numpy()
#             pred_list.extend(np.argmax(logits, axis=1).flatten())
#             label_list.extend(label_ids.flatten())
                outputs = model(b_input_ids, 
                                token_type_ids=None, 
                                attention_mask=b_input_mask)
            logits = outputs[0]
            logits = torch.argmax(logits, dim=1)
            labels, outputs = accelerator.gather([b_labels, logits])
            targets.append(labels)
            preds.append(outputs)
        
        targets = torch.cat(targets)[:num_test_sample]
        preds = torch.cat(preds)[:num_test_sample]
        # f.writelines(classification_report(targets, preds, zero_division=0)+'\n')
        return accuracy_score(targets, preds), f1_score(targets, preds, average='macro'), f1_score(targets, preds, average='micro')
    
    # Tracking loss with wandb
    wandb.init(
        # Set the project where this run will be logged
        project="MFinBERT_classification_subtask", 
        name=f"experiment", 
    )
    
    if args.validation_strategy == 'cross_validation':
        k_fold = args.k_fold 
        data = pd.DataFrame(data, columns=['text', 'label'])
        # print(data.head())
        # sys.exit()
        kf = StratifiedKFold(n_splits=args.k_fold, random_state=231, shuffle=True)
        
        ACC, MAC_F1, MIC_F1 = [], [], []
        count = 1
        _input=data.drop('label', axis=1)
        _output=data.label
        for train_index, test_index in kf.split(_input, _output):
            accelerator.print('========= fold: {} ==========='.format(count))
            f.writelines('========= fold: {} ==========='.format(count)+'\n')
            
            X_train, X_test = _input.iloc[train_index].values.tolist(), _input.iloc[test_index].values.tolist()
            y_train, y_test = _output.iloc[train_index].values.tolist(), _output.iloc[test_index].values.tolist()
            train_data = [[i,j] for (i,j) in zip(X_train, y_train)]
            test_data = [[i,j] for (i,j) in zip(X_test, y_test)]
 
            if args.state:
                train_data = train_data[:400]
                test_data = test_data[:200]
                
            num_train_sample, num_test_sample = len(train_data), len(test_data)
            train_dataloader = GenericDataLoader(train_data, args.per_device_train_batch_size)
            test_dataloader = GenericDataLoader(test_data, args.per_device_eval_batch_size)
            y = torch.tensor([i[1] for i in train_data])
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
            class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
             
            model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        
            if args.freeze_layer_count:
                # We freeze here the embeddings of the model
                for param in model.bert.embeddings.parameters():
                    param.requires_grad = False

                if args.freeze_layer_count != -1:
                    # if freeze_layer_count == -1, we only freeze the embedding layer
                    # otherwise we freeze the first `freeze_layer_count` encoder layers
                    for layer in model.bert.encoder.layer[:args.freeze_layer_count]:
                        for param in layer.parameters():
                            param.requires_grad = False
                            
            """
            optimizer
            """
            # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
            bert_params = model.bert.encoder.named_parameters()
            classifier_params = model.classifier.named_parameters()
            grouped_params = [
                {'params': [p for n,p in bert_params if p.requires_grad==True], 'lr': args.lr_bert},
                {'params': [p for n,p in classifier_params if p.requires_grad==True], 'lr': args.lr_fc}
            ]
            
            optimizer = torch.optim.AdamW(grouped_params)

            num_training_steps = num_epochs * len(train_dataloader)
            lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
            loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
            
            model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, test_dataloader, lr_scheduler
                )
            
            # progress_bar = tqdm(range(num_training_steps))
#             count_parameters(model)
            best_model = train(model, args.num_train_epochs, loss_fn, optimizer, lr_scheduler, args.validation_strategy, num_test_sample, train_dataloader, test_dataloader)
            acc, macro_f1_score, micro_f1_score = evaluation(best_model, test_dataloader, num_test_sample)  
            ACC.append(acc) 
            MAC_F1.append(macro_f1_score)
            MIC_F1.append(micro_f1_score)
            count += 1
        accelerator.print('======================')
        accelerator.print(" Final acccuracy: {0:.4f}".format(sum(ACC)/len(ACC)))
        accelerator.print(" Final macro f1 score: {0:.4f}".format(sum(MAC_F1)/len(MAC_F1)))
        accelerator.print(" Final micro f1 score: {0:.4f}".format(sum(MIC_F1)/len(MIC_F1)))

        f.writelines('======================')
        f.writelines(" Final acccuracy: {0:.4f}".format(sum(ACC)/len(ACC))+'\n')
        f.writelines(" Final macro f1 score: {0:.4f}".format(sum(MAC_F1)/len(MAC_F1))+'\n')
        f.writelines(" Final micro f1 score: {0:.4f}".format(sum(MIC_F1)/len(MIC_F1))+'\n')
        
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(best_model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)
        
    else:
        
        if args.state:
            train_data = train_data[:400]
            test_data = test_data[:200]
        num_train = int(len(train_data)*0.8)
        
        num_train_sample, num_test_sample = len(train_data), len(test_data)
        train_dataloader = GenericDataLoader(train_data, args.per_device_train_batch_size)
        test_dataloader = GenericDataLoader(test_data, args.per_device_eval_batch_size)

        y = torch.tensor([i[1] for i in train_data])
        class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y.numpy())
        class_weights=torch.tensor(class_weights,dtype=torch.float).to(device)
        
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, num_labels=args.num_labels)
        
        if args.freeze_layer_count:
            # We freeze here the embeddings of the model
            for param in model.bert.embeddings.parameters():
                param.requires_grad = False

            if args.freeze_layer_count != -1:
                # if freeze_layer_count == -1, we only freeze the embedding layer
                # otherwise we freeze the first `freeze_layer_count` encoder layers
                for layer in model.bert.encoder.layer[:args.freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False

        """
        optimizer
        """
        # optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        bert_params = model.bert.encoder.named_parameters()
        classifier_params = model.classifier.named_parameters()
        grouped_params = [
            {'params': [p for n,p in bert_params if p.requires_grad==True], 'lr': args.lr_bert},
            {'params': [p for n,p in classifier_params if p.requires_grad==True], 'lr': args.lr_fc}
        ] 
        optimizer = torch.optim.AdamW(grouped_params)
        
        """
        scheduler
        """
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_scheduler(
            name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
        )
        
        """
        loss function
        """
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')
        
        model, optimizer, train_dataloader, test_dataloader, lr_scheduler = accelerator.prepare(
                model, optimizer, train_dataloader, test_dataloader, lr_scheduler
                )
        
        best_model = train(model, wandb, args.num_train_epochs, loss_fn, optimizer, lr_scheduler, args.validation_strategy, num_test_sample, train_dataloader, test_dataloader)

        acc, mac_f1_score, mic_f1_score = evaluation(best_model, test_dataloader, num_test_sample)

        accelerator.print('======================')
        accelerator.print(" Final acccuracy: {0:.4f}".format(acc))
        accelerator.print(" Final macro f1 score: {0:.4f}".format(mac_f1_score))
        accelerator.print(" Final micro f1 score: {0:.4f}".format(mic_f1_score))

        # f.writelines('\n======================\n')
        # f.writelines("Final acccuracy: {0:.4f}".format(acc)+'\n')
        # f.writelines("Final macro f1 score: {0:.4f}".format(mac_f1_score)+'\n')
        # f.writelines("Final micro f1 score: {0:.4f}".format(mic_f1_score)+'\n')
        
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(best_model)
            unwrapped_model.save_pretrained(
                args.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(args.output_dir)


    f.close()
    
if __name__ == "__main__":
    main()