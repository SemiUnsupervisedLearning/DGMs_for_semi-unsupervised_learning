import matplotlib
from collections import Counter
import numpy as np
np.random.seed(seed=0)
from sklearn.metrics import confusion_matrix
import tensorflow as tf
tf.set_random_seed(1)
from data.data import load_dataset, make_dataset
from models.m2 import m2
from models.gm_dgm import gm_dgm
from models.agm_dgm import agm_dgm
from models.adgm import adgm


import getpass

from utils.checkmate import get_best_checkpoint
from utils.utils import mkdir_p, touch, h_opt_args_shuffle, make_token
import argparse
import os

matplotlib.use('Agg')
np.random.seed(seed=0)


### Script to run an (F) MNIST experiment with generative models ###


def get_args():
    '''This function parses and return arguments passed in'''
    # Assign description to the help doc
    parser = argparse.ArgumentParser(
        description='Run DGM models over MNIST w. different prop labelled')
    # Add arguments
    parser.add_argument(
        '-m', '--model_name', choices=['m2', 'gm_dgm', 'adgm','agm_dgm'], required=True)
    parser.add_argument(
        '-d', '--dataset_name', choices=['mnist', 'fmnist'], default='mnist')
    parser.add_argument(
        '-p', '--prop_labelled', type=float, required=True)
    parser.add_argument(
        '-r', '--number_of_runs', type=int, required=True)
    parser.add_argument(
        '-e', '--number_of_epochs', type=int, required=True)
    parser.add_argument(
        '-c', '--classes_to_hide', nargs='*', type=int)
    parser.add_argument(
        '-a', '--number_of_classes_to_add', type=int, default=0)
    parser.add_argument(
        '-z', '--number_of_dims_z', type=int, default=100)
    parser.add_argument(
        '-u', '--number_of_dims_a', type=int, default=100)
    parser.add_argument(
        '-s', '--number_of_mc_samples', type=int, default=1)
    parser.add_argument(
        '-t', '--iteration_number', type=int, default=0)
    parser.add_argument(
        '-l', '--number_of_units_in_hidden_layers', type=int, default=500)
    parser.add_argument(
        '-b', '--batch_size', type=int, default=100)
    parser.add_argument(
        '--decay_period', type=int, default=200)
    parser.add_argument(
        '--decay_ratio', type=float, default=0.75)
    parser.add_argument(
        '--loss_balance',choices=['average', 'weighted'], default='average')
    # Array for all arguments passed to script
    args = parser.parse_args()
    # Assign args to variables
    return args


args = get_args()

model_name = args.model_name
dataset_name = args.dataset_name
prop_labelled = args.prop_labelled
num_runs = args.number_of_runs
n_epochs = args.number_of_epochs
classes_to_hide = args.classes_to_hide
if classes_to_hide == []:
    classes_to_hide=None
number_of_classes_to_add = args.number_of_classes_to_add
n_cta = float(number_of_classes_to_add)
iteration = args.iteration_number
decay_ratio = args.decay_ratio
decay_period = args.decay_period
loss_balance = args.loss_balance
restore = args.restore
args.restore=False

if prop_labelled == 0:
    learning_paradigm = 'unsupervised'
elif prop_labelled < 0:
    learning_paradigm = 'supervised'
elif prop_labelled > 0 and classes_to_hide is not None:
    learning_paradigm = 'semi-unsupervised'
elif prop_labelled > 0:
    learning_paradigm = 'semisupervised'
# Load and conver data to relevant type
print(learning_paradigm)

username = getpass.getuser()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = str(iteration)

token = make_token(args)

#make output directory if does not exist
cwd = os.getcwd()

token = make_token(args)
if username != 'magd4534':
    output_dir = os.path.join(cwd, '../output', dataset_name, learning_paradigm, model_name, token)
else:
    output_dir = os.path.join(cwd, '/data/dph-ukbaccworkgroup/magd4534/semi_unsup/output', dataset_name, learning_paradigm, model_name, token)
if os.path.isdir(output_dir) == False:
    os.makedirs(output_dir)

print(args)
if restore:
    print('restoring')

n_z = args.number_of_dims_z
n_a = args.number_of_dims_a
n_h = args.number_of_units_in_hidden_layers
batch_size = args.batch_size
mc_samps = args.number_of_mc_samples

x_train, y_train, x_test, y_test, binarize, x_dist, n_y, n_x, f_enc, f_dec  = load_dataset(dataset_name)


if prop_labelled <= 1:
    num_labelled = int(prop_labelled*x_train.shape[0])
    num_classes = y_train.shape[1]
if prop_labelled > 1:
    num_labelled = prop_labelled
    num_classes = y_train.shape[1]

#remove certain classes:
if classes_to_hide is not None:
    num_labelled = [int(float(num_labelled)/num_classes)]*num_classes
    for hide_class in classes_to_hide:
        num_labelled[hide_class] = 0



Data = make_dataset(learning_paradigm, dataset_name, x_test, y_test,
                    x_train, y_train,
                    num_labelled=num_labelled,
                    number_of_classes_to_add=number_of_classes_to_add)

if dataset_name == 'mnist':
    prior = np.array([1.0 / n_y] * n_y)
else:
    prior = np.array(list(Counter(np.argmax(Data.data['y_train'], axis=1)).values())).astype(np.float32)
    prior = prior/np.sum(prior)
n_y_f = float(n_y)

# Our option to add extra classes enables us to have the dimensionality of our
# y representation to be greater than the number of true classes - later on we
# will then associate all these unsupervised classes using a `cluster&label'
# approach.

# Thus we much add appropriate ammount of probability mass to these new classes
# so we divide the mass for the hidden (ie unsupervised) classes equally
# between the original, now unsupervised, classes and the extra, added, classes

if classes_to_hide is not None:
    n_cth = float(len(classes_to_hide))
    prior_for_other_classes = (1.0 - (n_y_f - n_cth) / n_y_f) / (n_cth + n_cta)
    for hide_class in classes_to_hide:
        prior[hide_class] = prior_for_other_classes
    if number_of_classes_to_add > 0:
        prior = np.concatenate((prior, np.ones(number_of_classes_to_add) *
                                prior_for_other_classes))


n_y = n_y + number_of_classes_to_add


l_bs, u_bs = batch_size, batch_size
alpha = 0.1

loss_ratio = float(l_bs) / float(u_bs)

# Specify model parameters
lr = (3e-4,)
n_w = 50
l2_reg, alpha = .5, 1.1
eval_samps = 1000
verbose = 3

Data.reset_counters()
results=[]
for i in range(num_runs):
    global_iteration_number = num_runs * iteration + i
    print("Starting work on run: {}".format(global_iteration_number))
    Data.reset_counters()
    np.random.seed(global_iteration_number)
    tf.set_random_seed(global_iteration_number)
    tf.reset_default_graph()
    model_token = token+'-'+str(global_iteration_number)+'---'

    if model_name == 'm2':
        model = m2(n_x, n_y, n_h, n_z, x_dist=x_dist, mc_samples=mc_samps, alpha=alpha, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, output_dir=output_dir, loss_balance=loss_balance)
    if model_name == 'gm_dgm':
        model = gm_dgm(n_x, n_y, n_h, n_z, x_dist=x_dist, mc_samples=mc_samps, alpha=alpha, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, prior=prior[0:n_y]/float(sum(prior[0:n_y])), output_dir=output_dir, loss_balance=loss_balance)
    if model_name == 'agm_dgm':
        model = agm_dgm(n_x, n_y, n_h, n_z, n_a=n_a, x_dist=x_dist, mc_samples=mc_samps, alpha=alpha, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, prior=prior[0:n_y]/float(sum(prior[0:n_y])), output_dir=output_dir, loss_balance=loss_balance)
    if model_name == 'adgm':
        model = adgm(n_x, n_y, n_z, n_h, n_a=n_a, x_dist=x_dist, mc_samples=mc_samps, alpha=alpha, l2_reg=l2_reg, learning_paradigm=learning_paradigm, name=model_token, ckpt = model_token, prior=prior[0:n_y]/float(sum(prior[0:n_y])), output_dir=output_dir, loss_balance=loss_balance)

    if learning_paradigm == 'semisupervised' or learning_paradigm == 'semi-unsupervised':
        print('ss/sus')
        model.loss = model.compute_loss()
    elif learning_paradigm == 'unsupervised':
        print('us')
        model.loss = model.compute_unsupervised_loss()
    elif model.learning_paradigm == 'supervised':
        print('s')
        model.loss = model.compute_supervised_loss()

    model.train(Data, n_epochs, l_bs, u_bs, lr, eval_samps=eval_samps, binarize=binarize, verbose=verbose, decay_period=decay_period, decay_ratio=decay_ratio, restore=restore)
    results.append(model.curve_array)
    np.save(os.path.join(output_dir,'curve_'+token+'_'+str(global_iteration_number)+'.npy'), model.curve_array)
    y_pred_test = model.predict_new(Data.data['x_test'])
    conf_mat = confusion_matrix(Data.data['y_test'].argmax(1), y_pred_test.argmax(1))
    np.save(os.path.join(output_dir,'conf_mat_'+token+'_'+str(global_iteration_number)+'.npy'), conf_mat)
    np.savez(os.path.join(output_dir,'y_preds_labels_'+token+'_'+str(global_iteration_number)+'.npz'), y_true=Data.data['y_test'].argmax(1), y_pred=y_pred_test.argmax(1), y_labels = y_test[1])
    if learning_paradigm == 'semisupervised' or learning_paradigm == 'un-semisupervised':
        Data.recreate_semisupervised(global_iteration_number)

np.save(os.path.join(output_dir,'results_'+ token+'.npy'), results)
