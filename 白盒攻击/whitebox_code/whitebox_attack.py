#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jinyitong
"""

import os
import argparse
import _pickle as pickle
import numpy as np
import torch
from model import CNN
import matplotlib.pyplot as plt

def data_preprocessing(data_path):
    """
    To load the images and labels from .pkl file.
    And convert them into torch Tensor for PyTorch processing.

    Arguments:
        data_path   --      path of .pkl file, which store the image samples and groundtruth labels     type: str

    Return:
        imgs        --      loaded images data.                 type: torch.Tensor()
        labels      --      loaded groundtruth labels.          type: torch.Tensor()
        ori_imgs    --      original images before attacking.   type: numpy.array()

    """
    # image load
    attack_data = pickle.load(open(data_path, 'rb'))

    # seperate data into images and labels
    imgs = np.array(attack_data[0])
    labels = np.array(attack_data[1])

    # copy the original img
    ori_imgs = imgs.copy()

    # convert imgs and labels to tensor
    imgs = torch.from_numpy(imgs)
    labels = torch.Tensor(np.array([labels])).long()
    labels = torch.reshape(labels, (-1,1))

    return imgs, labels, ori_imgs


def generating_adversarial_samples(num_samples, num_epochs, imgs, labels, \
                     model, optimizer, criterion, update_rate, ori_imgs, device):
    """
    Apply targeted attack to the model, by generating adversarial samples.
    Each sample can be iterated up to num_epochs times.
    After each iteration, if the prediction is same as targeted label, attack successes.
    Save the first 10 successful results in directory 'whitebox_adversarial_samples'
    After finishing all the samples, success attacking rate can be obtained.

    Arguments:
        num_samples     --      number of adversarial samples to generate       type: int
        num_epochs      --      the upper limit of iteration for each sample    type: int
        imgs            --      loaded images data.                             type: torch.Tensor()
        labels          --      loaded groundtruth labels.                      type: torch.Tensor()
        model           --      the trained neural network                      type: an object of class
        optimizer       --      optimization algorithm                          type: torch.optim
        criterion       --      Loss function                                   type: torch.nn.CrossEntropyLoss()
        update_rate     --      step length of each update                      type: float
        ori_imgs        --      original images before attacking.               type: numpy.array()
        device          --      type of device, cuda or cpu.                    type: torch.device()

    Return

    """
    imgs_list = []
    labels_list = []
    num_success = 0
    img_num = 0
    for sample in range(num_samples):
        img = imgs[sample].to(device)
        # !!! IMPORtANT! img is obtained from imgs, thus img is a non-leaf tensor, which cannot retain grad after backpropagation
            #       Thus, we should retain grad of img with "img.retain_grad()"
        img.retain_grad()
        label = labels[sample].to(device)
        # set the attack target
        target_item = (label.item() + 1) % 10
        target = torch.Tensor([target_item]).long().to(device)

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            output = model(img)
            loss = criterion(output, target)
            predict = torch.argmax(output, dim=1)
            # print("epoch: {} || Groundtruth: {} || prediction: {} || loss: {}".format(epoch, label.item(), predict.item(), loss))      # print every iterate
            if predict == target:
                num_success += 1
                # show original images, adversarial images and difference
                if img_num < 10:
                    img_num += 1
                    ori_img = ori_imgs[sample]
                    ori_img = ori_img.reshape(28, 28)
                    img = img.detach().to('cpu').numpy()
                    img_ = img.reshape(28, 28)
                    difference = img_ - ori_img
                    

                    plt.figure(figsize=(8, 6))
                    plt.subplot(131), plt.imshow(ori_img), plt.axis("off"), \
                    plt.title("original img"), plt.text(0, 30, "predicted as class {}".format(label.item()))
                    plt.subplot(132), plt.imshow(img_), plt.axis("off"), \
                    plt.title("adversarial img"), plt.text(0, 30, "predicted as class {}".format(predict.item()))
                    plt.subplot(133), plt.imshow(difference), plt.axis("off"), plt.title("difference")
                    plt.savefig("../whitebox_adversarial_samples/sample{}.png".format(img_num))

                break

            optimizer.zero_grad()
            loss.backward()
            img.data = img.data - update_rate * torch.sign(img.grad.data)
            
        # store the adversarial samples
        if type(img) == type(label):
            img = img.detach().to('cpu').numpy()
        imgs_list.append(img)
        labels_list.append(label.item())


        print("The {}th adversarial sample generated... \n   Groundtruth: class {}    ||    Prediction: class {}"\
              .format(sample+1, label.item(), predict.item()))

    # calculate the success rate of White box attack
    succ_rate = "%.2f%%" % (num_success / num_samples * 100)
    print("\nSuccess rate of White box attack: {}".format(succ_rate))

    # store the adversarial samples generated from white box, for attacking black box
    with open("../whitebox_attack_data/adversarial_samples_1k.pkl", "wb") as f:
        pickle.dump([imgs_list, labels_list], f)
    
    return




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--attack_data_path', type=str, default='../whitebox_attack_data/white_adversarial_correct_1k.pkl')
    parser.add_argument('--weights_save_path', type=str, default='../whitebox_model/White_ep_20_devacc_91.72_.pt')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--update_rate', type=int, default=0.1)
    parser.add_argument('--rand_seed', type=int, default=86)

    opt = parser.parse_args()

    # select cpu or gpu to run 
    if int(opt.gpu) < 0:
        device = torch.device('cpu')
        device_type = 'cpu'
        torch.manual_seed(opt.rand_seed)
        print("Run on CPU!")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        device_type = 'cuda'
        torch.manual_seed(opt.rand_seed)
        torch.cuda.manual_seed(opt.rand_seed)
        print("Run on GPU!")

    # load data from .pkl file and convert them into torch Tensor.
    imgs, labels, ori_imgs = data_preprocessing(opt.attack_data_path)

    # load model
    model = CNN().to(device)
    # load trained weights
    model.load_state_dict(torch.load(opt.weights_save_path, device_type))

    # freeze the model weights, trun on the grad of input imgs
    for param in model.parameters():
        param.requires_grad=False
    imgs.requires_grad=True

    # set optimizer and loss function
    optimizer = torch.optim.Adam([imgs])
    criterion = torch.nn.CrossEntropyLoss()

    # set the number of samples and upper limit of iteration
    num_samples = len(imgs)
    num_epochs = opt.num_epochs

    # ATTACK !!!
    generating_adversarial_samples(num_samples, num_epochs, imgs, labels,\
            model, optimizer, criterion, opt.update_rate, ori_imgs, device)

