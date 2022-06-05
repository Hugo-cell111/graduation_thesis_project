from config import config
import torch
import torch.nn as nn
from model.model_one import *
from model.unet import *
from model.pspnet import *
from model.segnet import *
from model.bisenet import *
from model.deeplab import *
from model.loss import *
from tqdm import tqdm, trange

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, SequentialSampler, DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from data.dataset import loadedDataset
import cv2
import time
import random


def main(args):

    # ---------------------------set GPU environment---------------------------
    n_gpu = torch.cuda.device_count()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group('nccl')
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    # ---------------------------dataset---------------------------
    # train
    train_dataset = loadedDataset(txt = "./datasets/train/train_example.txt")
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size = args.train_batch_size // n_gpu, sampler = train_sampler,
                              num_workers = args.num_workers, drop_last = False, pin_memory = True)
    # validation
    val_dataset = loadedDataset(txt = "./datasets/val/val_example.txt")
    val_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(val_dataset, sampler = val_sampler, batch_size = args.val_batch_size,
                            num_workers = args.num_workers, drop_last = True, pin_memory = True)

    # test
    test_dataset = loadedDataset(txt = "./datasets/test/test_example.txt")
    test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, sampler = test_sampler, batch_size = args.test_batch_size,
                             num_workers = args.num_workers, drop_last = True, pin_memory = True)

    # ---------------------------model---------------------------



    if args.model_name == "unet":
        model = unet(3,2)
    elif args.model_name == "fcn":
        model = fcn()
    elif args.model_name == "pspnet":
        model = PSPNet()
    elif args.model_name == "segnet":
        model = segnet()
    elif args.model_name == "deeplab":
        model = deeplab()

    # =============train=============== #
    if args.mode == "train":
        if args.model_name == "model_one":
            model = model_one(n_classes=2, aux_mode=args.mode)
        elif args.model_name == "bisenet":
            model = bisenet(n_classes=2, aux_mode='eval')
        model.to(device)
        model = DDP(model, find_unused_parameters=True,
                    device_ids=[local_rank], output_device=local_rank)
        if args.load_model:
            pretrained_dict = torch.load(args.pretrained_model)
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        # calculate the number of parameters
        if local_rank == 0:
            total_params = sum(p.numel() for p in model.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad)
            print(f'{total_trainable_params:,} training parameters.')
        # scaler = GradScaler()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
        #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.0001, momentum=0.9)
        StepLR = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda epoch: (1-epoch/args.num_train_epochs)**0.9)
        criterion1 = simpleloss()
        criterion2 = BinaryDiceLoss()
        crtierion_edge = DetailAggregateLoss()
        criterion_semantic_and_result = WeightedOhemCELoss(thresh=0.7, n_min=256*256*args.train_batch_size//(16*n_gpu), num_classes=2)
        g = lambda  x, y: criterion1(x,y)
        g1 = lambda x, y: crtierion_edge(x,y)
        g2 = lambda x, y: criterion_semantic_and_result(x,y)
        f = open("./" + args.model_name + "_train_result.txt","w+")
        for epoch_ in trange(args.num_train_epochs, desc="Epoch"):
            for step, (img, lab) in enumerate(tqdm(train_loader, desc="Iteration", ascii=True, total = len(train_loader))):
                # train

                data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
                _, _, h, w = data.shape
                a = h % 32 // 2
                b = w % 32 // 2
                data, target = data[:, :, a : h - a, b : w - b], target[:, a : h - a, b : w - b]
                if local_rank == 0:
                    start_time = time.time()
                #with autocast():
                predict = model(data)
                #predict = torch.argmax(predict, dim=1)
                if args.model_name == "model_one":
                    detail, semantic, predict = model(data)
                    loss = 0.3 * g1(detail, target) + 0.8 * (g(semantic, target) + g(predict, target)) + 0.2 * (g2(semantic, target) + g2(predict, target))
                elif args.model_name == "bisenet":

                    predict = model(data)
                    loss = g(predict, target)
                else:
                    loss = g(predict, target)


                if n_gpu > 1:
                    loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                if local_rank == 0:
                    delta_time = time.time() - start_time
                if (local_rank == 0) and ((step + 1) % args.eval_steps == 0):
                    print("===================train===================")
                    predict2 = torch.argmax(predict, dim=1)

                    acc = predict2.eq(target.view_as(predict2)).sum().item() / target.numel()
                    TP = predict2[target > 0.5].eq(target[target > 0.5].view_as(predict2[target > 0.5])).sum().item()
                    FN = target[target > 0.5].numel() - TP
                    TN = predict2[target < 0.5].eq(target[target < 0.5].view_as(predict2[target < 0.5])).sum().item()
                    FP = target[target < 0.5].numel() - TN

                    print(TP)
                    print(FN)
                    print(TN)
                    print(FP)
                    precision = TP / (TP + FP + 1e-6)
                    recall = TP / (TP + FN + 1e-6)
                    IOU = TP / (TP + FN + FP + 1e-6)
                    F1_score = 2 * precision * recall / (precision + recall + 1e-6)
                    print("loss={:.6f}, acc={:.6f}, precision={:.6f}, recall={:.6f}, F1_score={:.6f}, iou={:.6f}".format(
                        loss, acc, precision, recall, F1_score, IOU))
                    f.write("Epoch = " + str(epoch_) + ", step = " + str(step) + ", acc = " + str(acc) +
                            ", precision = " + str(precision) + ", recall = " + str(recall) + ", loss = " + str(loss) +
                            ", F1_score = " + str(F1_score) + ", time = " + str(delta_time) + ", IOU = " + str(IOU)
                            + ", TP = " + str(TP) + ", FN = " + str(FN) + ", TN" + str(TN) + ", FP = " + str(FP) + "\n")
                # eval
                if (step + 1) % args.eval_steps == 0:
                    print("===================Evaluate===================")
                    model.eval()
                    for k, (img1, lab1) in enumerate(tqdm(val_loader, desc="Evaluating", total = len(val_loader))):
                        data = img1.type(torch.cuda.FloatTensor).to(device)
                        target = lab1.type(torch.cuda.FloatTensor).to(device)
                        _, _, h, w = data.shape
                        a = h % 32 // 2
                        b = w % 32 // 2
                        data, target = data[:, :, a: h - a, b: w - b], target[:, a: h - a, b: w - b]
                        with torch.no_grad():
                            if args.model_name == "model_one":
                                _, _, predict = model(data)
                                #loss = args.deep_supervision_weight*(g(sub1, target)+g(sub2, target)+g(sub3, target)) \
                                #       + g(predict, target)
                            elif args.model_name == "bisenet":
                                #predict, _, _, _, _ = model(data)
                                predict = model(data)
                            else:
                                predict = model(data)
                                loss = g(predict, target)
                        predict = torch.argmax(predict, dim=1)
                        for i in range(args.val_batch_size):
                            img1 = target[i,:,:].cpu().detach().numpy()*255
                            file1 = "./example/" + \
                                    "_step" + str(k) + "_item" + str(i) + "label.jpg"
                            cv2.imwrite(file1, img1)
                            img2 = predict[i,:,:].cpu().detach().numpy()*255
                            file2 = "./example/" + \
                                    "_step" + str(k) + "_item" + str(i) + "predict.jpg"
                            cv2.imwrite(file2, img2)
                            img3 = data[i,:,:,:].permute(1,2,0).cpu().detach().numpy()*255
                            file3 = "./example/" + \
                                    "_step" + str(k) + "_item" + str(i) + "original_data.jpg"
                            cv2.imwrite(file3, img3)
            StepLR.step()
            args.deep_supervision_weight = args.deep_supervision_weight * (1 - (epoch_ + 1) / args.num_train_epochs)
            if local_rank == 0:
                PATH = "./model/" + args.model_name + "_epoch" + str(epoch_) + ".pth"
                torch.save(model.state_dict(), PATH)
        f.close()
    elif args.mode == "test":
        if args.model_name == "model_one":
            model = model_one(n_classes=2, aux_mode=args.mode)
        elif args.model_name == "bisenet":
            model = bisenet(n_classes=2, aux_mode='eval')
        model.to(device)
        model = DDP(model, find_unused_parameters=True,
                    device_ids=[local_rank], output_device=local_rank)
        #model.load_state_dict(torch.load(args.pretrained_model), strict=False)
        pretrained_dict = torch.load(args.pretrained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        f = open("./" + args.model_name + "_test_result.txt", "w+")
        begin = time.time()
        acc, precision, recall, F1_score, IOU, delta_time = 0, 0, 0, 0, 0,0
        total = 0
        for step, (img, lab) in enumerate(tqdm(test_loader, desc="Iteration", ascii=True, total = len(test_loader))):
            data, target = img.type(torch.cuda.FloatTensor).to(device), lab.type(torch.cuda.FloatTensor).to(device)
            _, _, h, w = data.shape
            a = h % 32
            b = w % 32
            data = data[:, :, :h-a, :w-b]
            if local_rank == 0:
                start_time = time.time()
            with torch.no_grad():
                predict = model(data)
            delta_time += time.time() - begin
            predict = torch.argmax(predict, dim=1)
            img2 = predict.squeeze().cpu().detach().numpy() * 255
            file2 = "./datasets/demo/" + str(total) + "predict.jpg"
            cv2.imwrite(file2, img2)
            total += 1
        f.close()

if __name__ == '__main__':
    args = config()
    main(args)