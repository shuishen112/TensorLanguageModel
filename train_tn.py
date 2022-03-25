import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data


from dataloader import DatasetFactory, PADDING_TOKEN
from TN import TN_model
from util import print_tokens
import pickle


def save_checkpoint(optimizer, model, epoch, file_path):
    checkpoint_dict = {
        'optimizer': optimizer.state_dict(),
        'model': model.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint_dict, file_path)


def load_checkpoint(optimizer, model, file_path):
    if not os.path.exists(file_path):
        return None
    checkpoint_dict = torch.load(file_path)
    epoch = checkpoint_dict['epoch']
    model.load_state_dict(checkpoint_dict['model'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    return epoch


def get_model(dataset, config):

    return TN_model(rank = 10, output_size=dataset.encoder.vocabulary_size)


def run_forward_pass_and_get_loss(model, input_x, target_y, lengths):
    input_x = input_x.to(model.device)
    target_y = target_y.to(model.device)
    # lengths = lengths.to(model.device)
    predictions, _ = model(input_x)
    # Mask out padded values
    target_y = target_y.view(-1)  # Flatten out the batch
    # mask = (target_y != model.padding_idx)
    # target_y *= mask.long()  # Make the target values at padded indices 0

    criterion = model.loss()
    loss = criterion(predictions.view(-1,model.output_size),target_y)
    return loss


def validate(dataset, model: TN_model):
    # tmp_hidden = model.hidden
    # tmp_loss_func = model.loss_func
    # model.reset_intermediate_vars()
    # loss_func = nn.CrossEntropyLoss(reduction='sum')
    params = {'batch_size': 1,
              'shuffle': False,
              'num_workers': 0 if os.name == 'nt' else 8}
    data_generator = data.DataLoader(dataset, **params)
    cross_entropy = 0
    total_length = 0
    for x_i, y_i, l_i in data_generator:
        total_length += l_i.item()
        cross_entropy += run_forward_pass_and_get_loss(model, x_i, y_i, l_i).item()
    perplexity = np.exp(cross_entropy/total_length)
    bpc = np.log2(perplexity)
    # model.hidden = tmp_hidden
    # model.loss_func = tmp_loss_func
    return bpc


def run_training(model: TN_model, dataset, config: dict, validation: bool, valid_dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=config['initial_lr'])
    epoch = load_checkpoint(optimizer, model, config['tn_filename'])
    if not epoch:
        epoch = 0
    epoch += 1
    params = {'batch_size': config['batch_size'],
              'shuffle': False,
              'num_workers': 0 if os.name == 'nt' else 8}
    data_generator = data.DataLoader(dataset, **params)
    while epoch < config['epochs'] + 1:

        for step, (x_i, y_i, l_i) in enumerate(data_generator):
            loss = run_forward_pass_and_get_loss(model, x_i, y_i, l_i)
            # Gradient descent step
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            optimizer.step()

            if step % 1 == 0:
                print('Epoch: {} Loss for step {} : {}'.format(epoch, step, round(loss.item(), 4)))
            if step % 1 == 0:
                print("predict:")
                gen_sample = model.generate_text(dataset.encoder, 'hello', 200)
                print(gen_sample)
        save_checkpoint(optimizer, model, epoch, config['tn_filename'])
        if validation and epoch % 2:
            bpc = validate(valid_dataset, model)
            print('BPC on validation set: ', bpc)
        if epoch in config['lr_schedule']:
            optimizer = torch.optim.Adam(model.parameters(), lr=config['lr_schedule'][epoch])
        epoch += 1


def main(dataset_name: str):
    print('Preparing training data')
    # if os.path.exists("data/train_ds"):
    #     train_ds = pickle.load(open('data/train_ds','rb'))
    # else:
    train_ds = DatasetFactory.get_dataset(dataset=dataset_name, mode='train')
    # pickle.dump(train_ds,open("data/train_ds",'wb'),protocol=-1)
    ds_config = DatasetFactory.get_config(dataset_name)
    print('Training data prepared')
    model = get_model(train_ds, ds_config)
    model.to(model.device)
    valid_ds = DatasetFactory.get_dataset(dataset=dataset_name, mode='valid')
    run_training(model, train_ds, ds_config, True, valid_ds)


def test_model(dataset_name: str):
    test_ds = DatasetFactory.get_dataset(dataset=dataset_name, mode='test')
    ds_config = DatasetFactory.get_config(dataset_name)
    model = get_model(test_ds, ds_config)
    load_checkpoint(optimizer=None, model=model, file_path=ds_config['filename'])
    model.to(model.device)
    bpc = validate(test_ds, model)
    print('BPC on test set: ', bpc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a LSTM network defined in model.py.')
    parser.add_argument('-d', '--dataset',
                        type=str,
                        help='Name of the dataset',
                        default='text8',
                        choices=['text8', 'ptb', 'hutter'])
    args = parser.parse_args()
    main(args.dataset)
    test_model(args.dataset)
