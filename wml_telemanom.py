# telemanom in Watson Machine Learning
import os
import sys
import numpy as np
import pandas as pd
import torch
import onnxruntime

import telemanom
from telemanom.helpers import Config
from telemanom.errors import Errors
import telemanom.helpers as helpers
from telemanom.channel import Channel
from telemanom.modeling import Model

local = False
try:
    model_path = os.environ["RESULT_DIR"]+"/model"
except Exception:
    local = True
    model_path = './model/mytrainedpytorchmodel.torch'
model_input_path = './model/mytrainedpytorchmodel'

print (model_path)

conf = Config("./config.yaml")

conf.dictionary['l_s'] = 250
conf.dictionary['epochs'] = 80
conf.dictionary['dropout'] = 0.2
conf.batch_size = 512
conf.l_s = 250
conf.epochs = 10
conf.dropout = 0.2
conf.lstm_batch_size=64

# Load data from
device="Armstarknew"
chan = Channel(conf, device)
helpers.make_dirs(conf.use_id, conf, "./")
print(chan)

chan.train = np.loadtxt('./wml_train.csv')
chan.test = np.loadtxt('./wml_test.csv')

# producing overlapping windows of length 260 for lookback (250) and prediction (10)
chan.shape_data(chan.train, train=True)
chan.shape_data(chan.test, train=False)

# init Pytorch double stacked LSTM model
model = Model(conf, conf.use_id, chan, "./", False)

try:
    model.model.load_state_dict(torch.load(model_input_path))
    model.model.eval()
except Exception as e:
    print('Load model failed with ', e)
    # drink a coffee - training takes roughly 30 minutes
    model.train_new(chan)
    #torch.save(model.model.state_dict(), model_path)

#model.train_new(chan)
model.export(model_path)

# save_path = saver.save(sess, model_path)
print("Model saved in file: %s" % model_path)

if local:
    os.system("(cd ./model;tar cvfz ../saved_model.tar.gz .)")
    print(str(os.listdir('./')))
else:
    print(os.environ["RESULT_DIR"])
    os.system("(cd $RESULT_DIR/model;tar cvfz ../saved_model.tar.gz .)")
    print(str(os.listdir(os.environ["RESULT_DIR"])))
sys.stdout.flush()

