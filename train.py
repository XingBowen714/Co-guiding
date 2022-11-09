# -*- coding: utf-8 -*-#

import os, json, random
import numpy as np
import torch

from mygat import ModelManager
from utils.loader import DatasetManager
from utils.process import Processor
from utils.config import *
import fitlog

if __name__ == "__main__":
    if args.fitlog:
        if 'SNIPS' in args.data_dir:
            fitlog.set_log_dir("mixsnips_logs/")
        if 'ATIS' in args.data_dir:
            fitlog.set_log_dir("mixatis_logs/")
        fitlog.add_hyper(args)
        fitlog.add_hyper_in_file(__file__)
    # Save training and model parameters.
        if not os.path.exists(args.save_dir):
            os.system("mkdir -p " + args.save_dir)

        log_path = os.path.join(args.save_dir, "param.json")
        with open(log_path, "w", encoding="utf8") as fw:
            fw.write(json.dumps(args.__dict__, indent=True))

    # Fix the random seed of package random.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Fix the random seed of Pytorch when using GPU.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.random_seed)
        torch.cuda.manual_seed(args.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Fix the random seed of Pytorch when using CPU.
    torch.manual_seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)

    # Instantiate a dataset object.
    dataset = DatasetManager(args)
    dataset.quick_build()
    dataset.show_summary()

    # Instantiate a network model object.
    model = ModelManager(
        args, len(dataset.word_alphabet),
        len(dataset.slot_alphabet),
        len(dataset.intent_alphabet)
    )
    model.show_summary()

    # To train and evaluate the models.
    process = Processor(dataset, model, args)
    accepted_test_slot_f1, accepted_test_intent_acc, accepted_test_sent_acc  = process.train()
    


    #result = Processor.validate(
     #   os.path.join(args.save_dir, "model/model.pkl"),
      #  dataset,
       # args.batch_size, len(dataset.intent_alphabet), args=args)
    #print('\nAccepted performance: ' + str(result) + " at test dataset;\n")
   # if args.fitlog:
    #    fitlog.finish()
