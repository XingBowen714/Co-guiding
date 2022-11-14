# -*- coding: utf-8 -*-#

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.autograd import Variable
import torch.nn.functional as F

import os
import time
import random
import numpy as np
from tqdm import tqdm
from collections import Counter

# Utils functions copied from Slot-gated model, origin url:
# 	https://github.com/MiuLab/SlotGated-SLU/blob/master/utils.py
from utils import miulab
import fitlog


def multilabel2one_hot(labels, nums):
    res = [0.] * nums
    if len(labels) == 0:
        return res
    if isinstance(labels[0], list):
        for label in labels[0]:
            res[label] = 1.
        return res
    for label in labels:
        res[label] = 1.
    return res


def instance2onehot(func, num_intent, data):
    res = []
    for intents in func(data):
        res.append(multilabel2one_hot(intents, num_intent))
    return np.array(res)


def normalize_adj(mx):
    """
    Row-normalize matrix  D^{-1}A
    torch.diag_embed: https://github.com/pytorch/pytorch/pull/12447
    """
    mx = mx.float()
    rowsum = mx.sum(2)
    r_inv = torch.pow(rowsum, -1)
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag_embed(r_inv, 0)
    mx = r_mat_inv.matmul(mx)
    return mx


class Processor(object):

    def __init__(self, dataset, model, args):
        self.__dataset = dataset
        self.__model = model
        self.args = args
        self.__batch_size = args.batch_size
        self.__load_dir = args.load_dir

        if args.gpu:
            time_start = time.time()
            self.__model = self.__model.cuda()

            time_con = time.time() - time_start
            print("The model has been loaded into GPU and cost {:.6f} seconds.\n".format(time_con))

        self.__criterion = nn.NLLLoss()
        self.__criterion_intent = nn.BCEWithLogitsLoss()

        self.__optimizer = optim.Adam(
            self.__model.parameters(), lr=self.__dataset.learning_rate, weight_decay=self.__dataset.l2
        )

        if self.__load_dir:
            if self.args.gpu:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'))
            else:
                print("MODEL {} LOADED".format(str(self.__load_dir)))
                self.__model = torch.load(os.path.join(self.__load_dir, 'model/model.pkl'),
                                          map_location=torch.device('cpu'))

    def train(self):
        best_dev_sent = 0.0
        best_dev_slot = 0.0
        best_dev_intent = 0.0
        
        accepted_test_slot_f1 = 0.0
        accepted_test_intent_acc = 0.0
        accepted_test_sent_acc = 0.0

        best_epoch = 0
        no_improve = 0
        dataloader = self.__dataset.batch_delivery('train')
        
        hgat_flag = 1
        for epoch in range(0, self.__dataset.num_epoch):
            total_slot_loss, total_intent_loss = 0.0, 0.0
            total_slot_est_loss, total_intent_est_loss = 0.0, 0.0
            total_slot_margin_loss, total_intent_margin_loss = 0.0, 0.0

            time_start = time.time()
            self.__model.train()

            for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
                padded_text, [sorted_slot, sorted_intent], seq_lens = self.__dataset.add_padding(
                    text_batch, [(slot_batch, True), (intent_batch, False)])
                sorted_intent_exp = []
                for item, num in zip(sorted_intent, seq_lens):
                    sorted_intent_exp.extend([item] * num)
                sorted_intent = [multilabel2one_hot(intents, len(self.__dataset.intent_alphabet)) for intents in
                                 sorted_intent_exp]
                text_var = torch.LongTensor(padded_text)
                slot_var = torch.LongTensor(sorted_slot)
                intent_var = torch.Tensor(sorted_intent)
                max_len = np.max(seq_lens)

                if self.args.gpu:
                    text_var = text_var.cuda()
                    slot_var = slot_var.cuda()
                    intent_var = intent_var.cuda()

                random_slot, random_intent = random.random(), random.random()
                intent_logit_list, slot_logit_list = self.__model(text_var, seq_lens)

                slot_var = torch.cat([slot_var[i][:seq_lens[i]] for i in range(0, len(seq_lens))], dim=0)
                
                slot_label_loglist = [F.log_softmax(slot_logit_list[i], dim = -1) for i in range(0, len(slot_logit_list))]
                slot_label_loglist = [torch.cat([slot_label_loglist[i][b][:seq_lens[b]] for b in range(0, len(seq_lens))], \
                    dim = 0) for i in range(0, len(slot_logit_list))]

                slot_label_list = [F.softmax(slot_logit_list[i], dim = -1) for i in range(0, len(slot_logit_list))]
                slot_label_list = [torch.cat([slot_label_list[i][b][:seq_lens[b]] for b in range(0, len(seq_lens))], \
                    dim = 0) for i in range(0, len(slot_logit_list))]
                
                slot_est_loss = [self.__criterion(slot_label_loglist[i], slot_var) for i in range(len(slot_label_loglist))]
                    
                 # slot_pred_loss = slot_est_loss[-1]
  #              print(slot_label_loglist[0].size()) 
 #               print(intent_label_list[0].size())
                intent_logit_list = [torch.cat([intent_logit_list[i][b][:seq_lens[b]] for b in range(0, len(seq_lens))], dim=0) \
                          for i in range(0, len(intent_logit_list))]
#                print(intent_label_list[0].size())
                intent_est_loss = [self.__criterion_intent(intent_logit_list[i], intent_var) for i in range(len(intent_logit_list))]
                
                intent_pred_loss = intent_est_loss[-1]

                intent_margin_loss, slot_margin_loss = 0.0, 0.0
               
                assert intent_var.size()[0] == slot_var.size()[0]    
                num_slot = len(self.__dataset.slot_alphabet)
                slot_one_hot = F.one_hot(slot_var,num_slot)
                for j in range(1,len(intent_logit_list)):
                    #intent_margin_loss_item, slot_margin_loss_item = 0.0, 0.0
                    intent_margin_matrix = F.relu(F.sigmoid(F.sigmoid(intent_logit_list[j-1]) - F.sigmoid(intent_logit_list[j])))
                    slot_margin_matrix = F.relu(slot_label_list[j-1] - slot_label_list[j])
                    intent_margin_loss += torch.sum(intent_margin_matrix * intent_var) / intent_var.size()[0]
                    slot_margin_loss += torch.sum(slot_margin_matrix * slot_one_hot.float()) / slot_var.size()[0] 
                '''    
                for t in range(intent_logit_list[j].size()[0]):
                        print(intent_margin_matrix.size(), intent_var.size())
                        intent_margin_loss_item += intent_margin_matrix[t][c] * intent_var[t][c]
                        slot_margin_loss_item += slot_margin_matrix[t][slot_var[t]]
                    intent_margin_loss = intent_margin_loss + intent_margin_loss_item #/all_cau_logits[j].size()[0]
                    slot_margin_loss = slot_margin_loss + slot_margin_loss_item
                intent_margin_loss = intent_margin_loss/len(intent_var)
                slot_margin_loss = slot_margin_loss/len(slot_var)
                '''
                #if hgat_flag: 
                    #assert len(intent_label_list) == self.args.step_num+1, print(len(intent_label_list), self.args.step_num)
                
                #if hgat_flag and self.args.step_num > 0 and self.args.slot_margin_loss_alpha > 0:
                 #3   for i in range(1,self.args.step_num+1):
                  #      intent_margin_loss += max(0, intent_est_loss[i].cpu().item() - intent_est_loss[i-1].cpu().item())
                   #     slot_margin_loss += max(0, slot_est_loss[i].cpu().item() - slot_est_loss[i-1].cpu().item())
                
      #          print(self.__model.training) 
                
                #for j in range(1, len(intent_logit_list)):
                    #isent_margin_loss_item = torch.sum(torch.index_select(F.relu(F.log_softmax(flat_preds_s[j-1], dim = -1) \
                     #   - F.log_softmax(flat_preds_s[j],dim = -1)), 1, var_sent))
                 #   act_margin_loss_item = torch.sum(torch.index_select(F.relu(F.log_softmax(flat_preds_a[j-1], dim = -1) \
                  #      - F.log_softmax(flat_preds_a[j], dim = -1)), 1, var_act))


                intent_slot_loss_alpha = self.args.intent_slot_loss_alpha
                intent_margin_loss_alpha = self.args.intent_margin_loss_alpha
                slot_margin_loss_alpha = self.args.slot_margin_loss_alpha
                
                intent_batch_loss = sum(intent_est_loss) + intent_margin_loss_alpha * intent_margin_loss
                slot_batch_loss = sum(slot_est_loss) + slot_margin_loss_alpha * slot_margin_loss
                #print(intent_margin_loss)
                batch_loss = intent_batch_loss * intent_slot_loss_alpha +  slot_batch_loss * (1- intent_slot_loss_alpha)
                #print(batch_loss) 
                if not hgat_flag or self.args.step_num < 1:
                    batch_loss = intent_est_loss[-1] + slot_est_loss[-1]
                self.__optimizer.zero_grad()
                batch_loss.backward()
                self.__optimizer.step()

                total_slot_loss += (1- intent_slot_loss_alpha) * slot_batch_loss.cpu().item()
                total_slot_est_loss += (1- intent_slot_loss_alpha) * sum(slot_est_loss).cpu().item()
                total_slot_margin_loss += (1- intent_slot_loss_alpha) * slot_margin_loss_alpha * slot_margin_loss

                total_intent_loss += intent_slot_loss_alpha * intent_batch_loss.cpu().item()
                total_intent_est_loss += intent_slot_loss_alpha * sum(intent_est_loss).cpu().item()
                total_intent_margin_loss += intent_slot_loss_alpha * intent_margin_loss_alpha * intent_margin_loss

            if self.args.fitlog:
                fitlog.add_loss(total_intent_loss, name='intent loss', step=epoch)
                fitlog.add_loss(total_intent_est_loss, name='intent_est loss', step=epoch)
                fitlog.add_loss(total_intent_margin_loss, name='intent_margin loss', step=epoch)

                fitlog.add_loss(total_slot_loss, name='slot loss', step=epoch)
                fitlog.add_loss(total_slot_est_loss, name='slot_est loss', step=epoch)
                fitlog.add_loss(total_slot_margin_loss, name='slot_margin loss', step=epoch)

                fitlog.add_loss(total_intent_loss + total_slot_loss, name='total loss', step=epoch)
            time_con = time.time() - time_start
            print(
                '[Epoch {:2d}]: The total slot loss on train data is {:2.6f}, slot_margin_loss is : {:2.6f},  intent loss is {:2.6f}, intent_margin_loss is : {:2.6f},  cost about {:2.6} seconds.'.format(epoch, total_slot_loss, total_slot_margin_loss, total_intent_loss,total_intent_margin_loss,  time_con))
            change, time_start = False, time.time()
            dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score = self.estimate(
                if_dev=True,
                test_batch=self.__batch_size,
                args=self.args, hgat_flag=hgat_flag)
            print('\nDev result: epoch: {}, slot f1 score: {:.6f}, intent f1 score: {:.6f}, intent acc score:'
                      ' {:.6f}, semantic accuracy score: {:.6f}.'.
                      format(epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score, dev_sent_acc_score))

            if self.args.fitlog:
                fitlog.add_metric(
                {"dev": {"slot f1": dev_slot_f1_score,
                         "intent f1": dev_intent_f1_score,
                         "intent acc": dev_intent_acc_score,
                         "exact acc": dev_sent_acc_score
                         }
                 },
                step=epoch
            )
            
            test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc = self.estimate(
                    if_dev=False,test_batch=self.__batch_size, args=self.args, hgat_flag=hgat_flag)

            print('\nTest result: epoch: {}, slot f1 score: {:.6f}, intent f1 score: {:.6f}, intent acc score:'
                      ' {:.6f}, semantic accuracy score: {:.6f}.'.
                      format(epoch, test_slot_f1, test_intent_f1, test_intent_acc, test_sent_acc))
            if self.args.fitlog:
                fitlog.add_metric(
                    {"test": {"slot f1": test_slot_f1,
                              "intent f1": test_intent_f1,
                              "intent acc": test_intent_acc,
                              "exact acc": test_sent_acc
                              }
                     },
                    step=epoch
                )
            if dev_sent_acc_score >= best_dev_sent or dev_slot_f1_score >= best_dev_slot or dev_intent_acc_score > best_dev_intent:
                no_improve = 0
                best_epoch = epoch
                best_dev_sent = dev_sent_acc_score
                best_dev_slot = dev_slot_f1_score
                best_dev_intent = dev_intent_acc_score
               
                accepted_test_slot_f1 = test_slot_f1
                accepted_test_intent_acc = test_intent_acc
                accepted_test_sent_acc = test_sent_acc
                
                model_save_dir = os.path.join(self.__dataset.save_dir, "model")
                if not os.path.exists(model_save_dir):
                    try:
                        os.mkdir(model_save_dir)
                    except:
                        pass
                if self.args.fitlog:
                    fitlog.add_best_metric(
                    {"dev": {"slot f1": dev_slot_f1_score,
                             "intent f1": dev_intent_f1_score,
                             "intent acc": dev_intent_acc_score,
                             "exact acc": dev_sent_acc_score
                             }
                     }
                    )
                    fitlog.add_best_metric(
                        {"dev-test": {"slot f1": test_slot_f1,
                              "intent f1": test_intent_f1,
                              "intent acc": test_intent_acc,
                              "exact acc": test_sent_acc
                              }
                         }
                    )
                    torch.save(self.__model, os.path.join(model_save_dir, "model.pkl"))
                    torch.save(self.__dataset, os.path.join(model_save_dir, 'dataset.pkl'))

                time_con = time.time() - time_start
                #print('[Epoch {:2d}]: In validation process, the slot f1 score is {:2.6f}, ' \
                 #     'the intent f1 score is {:2.6f}, the intent acc score is {:2.6f}, the semantic acc is {:.2f}, cost about {:2.6f} seconds.\n'.format(
                  #  epoch, dev_slot_f1_score, dev_intent_f1_score, dev_intent_acc_score,
                   # dev_sent_acc_score, time_con))

            else:
                no_improve += 1
                # self.__lr_scheduler.step()

            if self.args.early_stop == True:
                if no_improve > self.args.patience:
                    print('early stop at epoch {}'.format(epoch))
                    break
        print('best slot f1: {:.6f}, best intent acc: {:.6f}, best semntic acc: {:.6f} '.format(accepted_test_slot_f1, accepted_test_intent_acc, accepted_test_sent_acc) )

        print('Best dev epoch is {}'.format(best_epoch))

        return accepted_test_slot_f1, accepted_test_intent_acc, accepted_test_sent_acc

    def estimate(self, if_dev, args, test_batch=100, hgat_flag=-1):
        """
        Estimate the performance of model on dev or test dataset.
        """
        if hgat_flag != 0 and hgat_flag != 1:
            assert 1==0, print('hgat_flag error')
        if if_dev:
            ss, pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "dev", test_batch, args, hgat_flag)
        else:
            ss, pred_slot, real_slot, pred_intent, real_intent = self.prediction(
                self.__model, self.__dataset, "test", test_batch, args, hgat_flag)

        num_intent = len(self.__dataset.intent_alphabet)
        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_f1_score = f1_score(
            instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, real_intent),
            instance2onehot(self.__dataset.intent_alphabet.get_index, num_intent, pred_intent),
            average='macro')
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))
        # Write those sample both have intent and slot errors.
        with open(os.path.join(args.save_dir, 'error.txt'), 'w', encoding="utf8") as fw:
            for p_slot_list, r_slot_list, p_intent_list, r_intent in \
                    zip(pred_slot, real_slot, pred_intent, real_intent):
                fw.write(','.join(p_intent_list) + '\t' + ','.join(r_intent) + '\n')
                for w, r_slot, in zip(p_slot_list, r_slot_list):
                    fw.write(w + '\t' + r_slot + '\t''\n')
                fw.write('\n\n')

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def validate(model_path, dataset, batch_size, num_intent, args, hgat_flag):
        """
        validation will write mistaken samples to files and make scores.
        """

        if args.gpu:
            model = torch.load(model_path)
        else:
            model = torch.load(model_path, map_location=torch.device('cpu'))

        ss, pred_slot, real_slot, pred_intent, real_intent = Processor.prediction(
            model, dataset, "test", batch_size, args, hgat_flag = hgat_flag)

        # To make sure the directory for save error prediction.
        mistake_dir = os.path.join(dataset.save_dir, "error")
        if not os.path.exists(mistake_dir):
            os.mkdir(mistake_dir)

        slot_f1_score = miulab.computeF1Score(ss, real_slot, pred_slot, args)[0]
        intent_f1_score = f1_score(instance2onehot(dataset.intent_alphabet.get_index, num_intent, real_intent),
                                   instance2onehot(dataset.intent_alphabet.get_index, num_intent, pred_intent),
                                   average='macro')
        intent_acc_score = Evaluator.intent_acc(pred_intent, real_intent)
        sent_acc = Evaluator.semantic_acc(pred_slot, real_slot, pred_intent, real_intent)
        print("slot f1: {}, intent f1: {}, intent acc: {}, exact acc: {}".format(slot_f1_score, intent_f1_score,
                                                                                 intent_acc_score, sent_acc))
        # Write those sample both have intent and slot errors.

        with open(os.path.join(args.save_dir, 'error.txt'), 'w', encoding="utf8") as fw:
            for p_slot_list, r_slot_list, p_intent_list, r_intent in \
                    zip(pred_slot, real_slot, pred_intent, real_intent):
                fw.write(','.join(p_intent_list) + '\t' + ','.join(r_intent) + '\n')
                for w, r_slot, in zip(p_slot_list, r_slot_list):
                    fw.write(w + '\t' + r_slot + '\t''\n')
                fw.write('\n\n')

        return slot_f1_score, intent_f1_score, intent_acc_score, sent_acc

    @staticmethod
    def prediction(model, dataset, mode, batch_size,args, hgat_flag):
        model.eval()

        if mode == "dev":
            dataloader = dataset.batch_delivery('dev', batch_size=batch_size, shuffle=False, is_digital=False)
        elif mode == "test":
            dataloader = dataset.batch_delivery('test', batch_size=batch_size, shuffle=False, is_digital=False)
        else:
            raise Exception("Argument error! mode belongs to {\"dev\", \"test\"}.")

        pred_slot, real_slot = [], []
        pred_intent, real_intent = [], []
        all_token = []
        for text_batch, slot_batch, intent_batch in tqdm(dataloader, ncols=50):
            padded_text, [sorted_slot, sorted_intent], seq_lens = dataset.add_padding(
                text_batch, [(slot_batch, False), (intent_batch, False)],
                digital=False
            )
            real_slot.extend(sorted_slot)
            all_token.extend([pt[:seq_lens[idx]] for idx, pt in enumerate(padded_text)])
            for intents in list(Evaluator.expand_list(sorted_intent)):
                if '#' in intents:
                    real_intent.append(intents.split('#'))
                else:
                    real_intent.append([intents])

            digit_text = dataset.word_alphabet.get_index(padded_text)
            var_text = torch.LongTensor(digit_text)
            max_len = np.max(seq_lens)
            if args.gpu:
                var_text = var_text.cuda()
            slot_idx, intent_idx = model(var_text, seq_lens, hgat_flag, n_predicts=1)
            nested_slot = Evaluator.nested_list([list(Evaluator.expand_list(slot_idx))], seq_lens)[0]
            pred_slot.extend(dataset.slot_alphabet.get_instance(nested_slot))
            intent_idx_ = [[] for i in range(len(digit_text))]
            for item in intent_idx:
                intent_idx_[item[0]].append(item[1])
            intent_idx = intent_idx_
            pred_intent.extend(dataset.intent_alphabet.get_instance(intent_idx))
        if 'MixSNIPS' in args.data_dir or 'MixATIS' in args.data_dir or 'DSTC' in args.data_dir:
            [p_intent.sort() for p_intent in pred_intent]
        with open(os.path.join(args.save_dir, 'token.txt'), "w", encoding="utf8") as writer:
            idx = 0
            for line, slots, rss in zip(all_token, pred_slot, real_slot):
                for c, sl, rsl in zip(line, slots, rss):
                    writer.writelines(
                        str(sl == rsl) + " " + c + " " + sl + " " + rsl + "\n")
                idx = idx + len(line)
                writer.writelines("\n")

        return all_token, pred_slot, real_slot, pred_intent, real_intent


class Evaluator(object):

    @staticmethod
    def intent_acc(pred_intent, real_intent):
        total_count, correct_count = 0.0, 0.0
        for p_intent, r_intent in zip(pred_intent, real_intent):

            if p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def semantic_acc(pred_slot, real_slot, pred_intent, real_intent):
        """
        Compute the accuracy based on the whole predictions of
        given sentence, including slot and intent.
        """
        total_count, correct_count = 0.0, 0.0
        for p_slot, r_slot, p_intent, r_intent in zip(pred_slot, real_slot, pred_intent, real_intent):

            if p_slot == r_slot and p_intent == r_intent:
                correct_count += 1.0
            total_count += 1.0

        return 1.0 * correct_count / total_count

    @staticmethod
    def accuracy(pred_list, real_list):
        """
        Get accuracy measured by predictions and ground-trues.
        """

        pred_array = np.array(list(Evaluator.expand_list(pred_list)))
        real_array = np.array(list(Evaluator.expand_list(real_list)))
        return (pred_array == real_array).sum() * 1.0 / len(pred_array)

    @staticmethod
    def f1_score_intents(pred_array, real_array):
        pred_array = pred_array.transpose()
        real_array = real_array.transpose()
        P, R, F1 = 0, 0, 0
        for i in range(pred_array.shape[0]):
            TP, FP, FN = 0, 0, 0
            for j in range(pred_array.shape[1]):
                if (pred_array[i][j] + real_array[i][j]) == 2:
                    TP += 1
                elif real_array[i][j] == 1 and pred_array[i][j] == 0:
                    FN += 1
                elif pred_array[i][j] == 1 and real_array[i][j] == 0:
                    FP += 1
            precision = TP / float(TP + FP) if (TP + FP) != 0 else 0
            recall = TP / float(TP + FN) if (TP + FN) != 0 else 0
            F1 += 2 * precision * recall / float(precision + recall) if (precision + recall) != 0 else 0
            P += precision
            R += recall
        P /= pred_array.shape[0]
        R /= pred_array.shape[0]
        F1 /= pred_array.shape[0]
        return F1

    @staticmethod
    def f1_score(pred_list, real_list):
        """
        Get F1 score measured by predictions and ground-trues.
        """

        tp, fp, fn = 0.0, 0.0, 0.0
        for i in range(len(pred_list)):
            seg = set()
            result = [elem.strip() for elem in pred_list[i]]
            target = [elem.strip() for elem in real_list[i]]

            j = 0
            while j < len(target):
                cur = target[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(target):
                        str_ = target[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    seg.add((cur, j, k - 1))
                    j = k - 1
                j = j + 1

            tp_ = 0
            j = 0
            while j < len(result):
                cur = result[j]
                if cur[0] == 'B':
                    k = j + 1
                    while k < len(result):
                        str_ = result[k]
                        if not (str_[0] == 'I' and cur[1:] == str_[1:]):
                            break
                        k = k + 1
                    if (cur, j, k - 1) in seg:
                        tp_ += 1
                    else:
                        fp += 1
                    j = k - 1
                j = j + 1

            fn += len(seg) - tp_
            tp += tp_

        p = tp / (tp + fp) if tp + fp != 0 else 0
        r = tp / (tp + fn) if tp + fn != 0 else 0
        return 2 * p * r / (p + r) if p + r != 0 else 0

    """
    Max frequency prediction. 
    """

    @staticmethod
    def max_freq_predict(sample):
        predict = []
        for items in sample:
            predict.append(Counter(items).most_common(1)[0][0])
        return predict

    @staticmethod
    def exp_decay_predict(sample, decay_rate=0.8):
        predict = []
        for items in sample:
            item_dict = {}
            curr_weight = 1.0
            for item in items[::-1]:
                item_dict[item] = item_dict.get(item, 0) + curr_weight
                curr_weight *= decay_rate
            predict.append(sorted(item_dict.items(), key=lambda x_: x_[1])[-1][0])
        return predict

    @staticmethod
    def expand_list(nested_list):
        for item in nested_list:
            if isinstance(item, (list, tuple)):
                for sub_item in Evaluator.expand_list(item):
                    yield sub_item
            else:
                yield item

    @staticmethod
    def nested_list(items, seq_lens):
        num_items = len(items)
        trans_items = [[] for _ in range(0, num_items)]

        count = 0
        for jdx in range(0, len(seq_lens)):
            for idx in range(0, num_items):
                trans_items[idx].append(items[idx][count:count + seq_lens[jdx]])
            count += seq_lens[jdx]

        return trans_items
