import pickle
import matplotlib.pyplot as plt

with open('outputs/classifier_False_TFC_True_abs_budget_False_stride_1_loss_poly/pretrain_latent_variables.pickle', 'rb') as path:
    outputs = pickle.load(path) 

pretrain = True
finetune = True
evaluate = True

print(not pretrain and (finetune or evaluate))