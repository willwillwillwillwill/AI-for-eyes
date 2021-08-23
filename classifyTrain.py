import argparse
import csv
import fcntl
import torch
import torch.distributed as dist
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from classifyModel import EyeNet
import glob
from tqdm import tqdm
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from eyeDataset import EyeDataset
import nni
import configparser
from torchvision import transforms
mixed_precision = True
try:  # Mixed precision training https://github.com/NVIDIA/apex
    from apex import amp
except:
    print('Apex recommended for faster mixed precision training: https://github.com/NVIDIA/apex')
    mixed_precision = False  # not installed

cfg_name = "best4.cfg"

wdir = 'weights' + os.sep  # weights dir
last = wdir + 'last_{}.pt'.format(cfg_name.replace(".cfg",""))
best = wdir + 'best_{}.pt'.format(cfg_name.replace(".cfg",""))
results_file = 'results.txt'
import contextlib
import matplotlib.pyplot as plt
import itertools

'''
损失函数
'''


class VATLoss(torch.nn.Module):

    def __init__(self, xi=10.0, eps=10, ip=1):
        """VAT loss
        :param xi: hyperparameter of VAT (default: 10.0)
        :param eps: hyperparameter of VAT (default: 1.0)
        :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VATLoss, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip

    def forward(self, net, x):
        # print(torch.mean(x))
        with torch.no_grad():
            pred = F.softmax(net(x), dim=1)
            # softmax_out = nn.Softmax(dim=1)(pred)
            # entropy = Entropy(softmax_out)

        # prepare random unit tensor
        d = torch.rand(x.shape).sub(0.5).to(x.device)
        d = _l2_normalize(d)
        # print("d",torch.mean(d))
        with _disable_tracking_bn_stats(net):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                pred_hat = net(x + self.xi * d)
                logp_hat = F.log_softmax(pred_hat, dim=1)
                adv_distance = F.kl_div(logp_hat, pred, reduction='batchmean')
                adv_distance.backward()
                d = _l2_normalize(d.grad)
                net.zero_grad()

            # calc LDS
            # r_adv = d * self.eps
            # print("d, entropy", d.shape, torch.mean(entropy))
            # entropy=entropy.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, 3, 224, 224)
            # entropy=torch.sigmoid(entropy)

            # r_adv = d * self.eps*(10/torch.exp(entropy))
            r_adv = d * self.eps
            # r_adv = d * self.eps*entropy
            pred_hat = net((x + r_adv))
            logp_hat = F.log_softmax(pred_hat, dim=1)
            lds = F.kl_div(logp_hat, pred, reduction='batchmean')

        return lds

class LabelSmoothing(torch.nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self,smoothing=0.05):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        # self.weight = weight
    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        # w = self.weight * target()
        w = torch.zeros_like(target).float()

        sum = 0
        for i in range(w.shape[0]):
            w[i] =  class_loss_weight[target[i]]
            sum += w[i]

        # print(sum)
        # print(w)
        # # print(target)
        loss = self.confidence * nll_loss * w  + self.smoothing * smooth_loss
        return loss.sum()/sum
#from focalloss import  FocalLoss
def loss_function(pred, target):
    # return FocalLoss(gamma=2)(pred,target)

    # return LabelSmoothing()(pred,target)
    return torch.nn.CrossEntropyLoss(weight=torch.Tensor(class_loss_weight).to(device))(pred, target)

'''
绘制混淆矩阵函数
主要输入:混淆矩阵，类名称，是否正则化
'''

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig = plt.figure()

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '%.2f' if normalize else '%d'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, fmt%(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return fig

def eval(dataloader,write=False):
    n_correct_pred = 0
    loss = 0
    img_list  = [[0]*nc for i in range(nc)]
    for i in range(nc):
        for j in range(nc):
            img_list[i][j] = []
    cf_matrix = np.zeros((nc, nc), dtype="int")
    for batch in dataloader:
        imgs, label = batch
        imgs = imgs.to(device).float()  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        label = label.to(device)
        y_hat = model(imgs,label)
        loss += loss_function(y_hat, label).item()
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred += torch.sum(label == labels_hat).item()
        for index, (i, j) in enumerate(zip(label, labels_hat)):
            cf_matrix[i][j] += 1
            if write:
                img_list[i][j].append(imgs[index])

    if write:
        for i in range(nc):
            for j in range(nc):
                if len(img_list[i][j]) > 0:
                    tensor_imgs = torch.zeros((len(img_list[i][j]),3,224,224))
                    for id, img in enumerate(img_list[i][j]):
                        tensor_imgs[id] = img
                    if tb_writer:
                        tb_writer.add_images("Image/{}_{}".format(i,j), tensor_imgs)

    return loss / len(dataloader), n_correct_pred * 1.0 / len(dataloader.dataset), cf_matrix


import time
def write_csv_file(file_name,content):
    nowtime=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    content["time"]=nowtime
    to_write_head = False
    if not os.path.exists(file_name):
        to_write_head=True
    with open(file_name,'a+') as f:
        writer=csv.DictWriter(f,content.keys())
        fcntl.flock(f,fcntl.LOCK_EX)
        if to_write_head:
            writer.writeheader()
        writer.writerow(content)
        # for key, value in content.items:
        #     writer.writerow([key, value])
        fcntl.flock(f,fcntl.LOCK_UN)


def find_lr(init_value = 1e-6, final_value=1., beta = 0.98):
    model.eval()
    num = len(trainloader)-1
    print(num)
    mult = (final_value / init_value) ** (1/num)
    lr = init_value
    optimizer.param_groups[0]['lr'] = lr
    avg_loss = 0.
    best_loss = 0.
    batch_num = 0
    losses = []
    log_lrs = []
    pbar = tqdm(enumerate(trainloader), total=len(trainloader))
    for _,data in pbar:
        batch_num += 1
        #As before, get the loss for this mini-batch of inputs/outputs
        inputs,labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs,labels)
        loss = loss_function(outputs, labels)
        #Compute the smoothed loss
        avg_loss = beta * avg_loss + (1-beta) *loss.item()
        smoothed_loss = avg_loss / (1 - beta**batch_num)
        #Stop if the loss is exploding
        if batch_num > 1 and smoothed_loss > 4 * best_loss:
            return log_lrs, losses
        #Record the best loss
        if smoothed_loss < best_loss or batch_num==1:
            best_loss = smoothed_loss
        #Store the values
        losses.append(smoothed_loss)
        log_lrs.append(np.log10(lr))
        # print(log_lrs)
        #Do the SGD step
        loss.backward()
        optimizer.step()
        #Update the lr for the next step
        lr *= mult
        pbar.set_description(str(lr))
        optimizer.param_groups[0]['lr'] = lr
    plt.figure()
    plt.xticks(np.log10([1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]), (1e-6,1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1))
    plt.xlabel('learning rate')
    plt.ylabel('loss')
    plt.plot(log_lrs, losses)
    plt.show()
    plt.figure()
    plt.xlabel('num iterations')
    plt.ylabel('learning rate')
    plt.plot(lr)
    return log_lrs, losses

def train():
    results = (0,0,0,0)#'accuracy' 'loss' of train and valid

    t0 = time.time()
    print('Using %g dataloader workers' % nw)
    print('Starting training for %g epochs...' % epochs)
    '''
    
    '''


    max_acc = 0
    epoch = 0
    best_chkpt = None
    vat_loss = VATLoss()
    for epoch in range(start_epoch, epochs):  # epoch ------------------------------------------------------------------
        model.train()
        pbar = tqdm(enumerate(trainloader), total=nb)  # progress bar
        n_correct_pred = 0
        train_loss = 0.0
        for i,batch in pbar:  # batch -------------------------------------------------------------

            ni = i + nb * epoch  # number integrated batches (since train start)
            imgs,label = batch
            # print(label)
            imgs = imgs.to(device).float()   # uint8 to float32, 0 - 255 to 0.0 - 1.0
            label = label.to(device)
            imgs.max()
            # Burn-in
            if ni < n_burn * 2:
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x['lr'] = np.interp(ni, [0, n_burn], [0.1 if j == 2 else 0.0, x['initial_lr'] * lf(epoch)])
                    if 'momentum' in x:
                        x['momentum'] = np.interp(ni, [0, n_burn], [0.9, momentum])



            # Forward
            y_hat = model(imgs,label)

            # Loss
            loss = loss_function(y_hat, label)
            train_loss += loss.item()
            # if not torch.isfinite(loss):
            #     print('WARNING: non-finite loss, ending training ', loss_items)
            #     return results
            labels_hat = torch.argmax(y_hat, dim=1)
            n_correct_pred += torch.sum(label == labels_hat).item()

            if mixed_precision:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

         
            optimizer.step()
            optimizer.zero_grad()
            s = "Epoch:{}.loss:{}".format(epoch+1,train_loss/(i+1))
            pbar.set_description(s)




            # end batch ------------------------------------------------------------------------------------------------

        # Update scheduler
        scheduler.step()

        # Process epoch results
        # ema.update_attr(model)
        final_epoch = epoch + 1 == epochs
        import matplotlib.pyplot as plt
        valid_acc = 0
        model.eval()
        if not opt.notest or final_epoch:  # Calculate mAP
            valid_loss,valid_acc,_ = eval(dataloader=validloader)
            results = [train_loss/len(trainloader),n_correct_pred*1.0/len(trainloader.dataset),valid_loss,valid_acc]
            print(_)


        # Tensorboard

        if tb_writer:
            tags = ['train/loss', 'train/accuracy',
                    'val/loss', 'val/accuracy']
            for x, tag in zip(list(results), tags):
                tb_writer.add_scalar(tag, x, epoch)




        save = True
        _ = _.astype("float")
        for i in range(nc):
            _[i] /= _[i].sum()
        a_acc  = np.mean(np.diag(_))
        if save:
            # with open(results_file, 'r') as f:  # create checkpoint
            chkpt = {'epoch': epoch,
                     # 'best_fitness': best_fitness,
                     'training_results': results,
                     'model': model.state_dict() if not hasattr(model, 'module') else
                    model.module.state_dict(),
                     'optimizer': None if final_epoch else optimizer.state_dict()}

            # Save last, best and delete
            torch.save(chkpt, last)
            # torch.save(model,"try.pkl")
            print(a_acc)
            nni.report_intermediate_result(a_acc)
            print(max_acc)
            if  max_acc < a_acc:
                # torch.save(model, "best.pkl")
                best_chkpt = chkpt
                max_acc = a_acc
                torch.save(chkpt, best)
            # if (best_fitness == fi) and not final_epoch:
            #     torch.save(chkpt, best)
            # del chkpt

        # end epoch ----------------------------------------------------------------------------------------------------

    model.load_state_dict(best_chkpt['model'])
    model.eval()
    _,valid_acc,cf_matrix = eval(write=False,dataloader=validloader)
    _, test_acc, cf_matrix_test = eval(write=False, dataloader=testloader)
    print(cf_matrix)
    if tb_writer:
        tb_writer.add_figure('confusion matrix/count', figure=plot_confusion_matrix(cf_matrix, classes=["0","1","2","3","4"], normalize=False,
                                                                           title='Normalized confusion matrix'),
                          global_step=1)
        tb_writer.add_figure('confusion matrix/normalized',
                             figure=plot_confusion_matrix(cf_matrix, classes=["0", "1", "2", "3", "4"], normalize=True,
                                                          title='Normalized confusion matrix'),
                         global_step=1)
    cf_matrix = cf_matrix.astype("float")
    for i in range(nc):
        cf_matrix[i] /= cf_matrix[i].sum()
    cf_matrix = np.around(cf_matrix,decimals=2)

    cf_matrix_test = cf_matrix_test.astype("float")
    for i in range(nc):
        cf_matrix_test[i] /= cf_matrix_test[i].sum()
    cf_matrix_test = np.around(cf_matrix_test, decimals=2)

    print(cf_matrix)
    print(cf_matrix_test)
    print("final acc:{}".format(valid_acc))
    print("final acc:{}".format(test_acc))
    '''
    nni operation
    '''

    nni.report_final_result(valid_acc)
    print('%g epochs completed in %.3f hours.\n' % (epoch - start_epoch + 1, (time.time() - t0) / 3600))
    dist.destroy_process_group() if torch.cuda.device_count() > 1 else None
    torch.cuda.empty_cache()
    if tb_writer:
        tb_writer.close()

    for i in range(nc):
        var["val_p"+str(i)] = cf_matrix[i][i]
    var["val_p"] = valid_acc

    for i in range(nc):
        var["test_p"+str(i)] = cf_matrix_test[i][i]
    var["test_p"] = test_acc
    var["test_avg"] = 0
    for i in range(nc):
        var["test_avg"] += var["test_p"+str(i)]
    var["test_avg"] /= nc
    write_csv_file("result.csv", var)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', action='store_true', help='resume training from last.pt')
    parser.add_argument('--notest', action='store_true', help='only test final epoch')
    parser.add_argument('--weights', type=str, default='', help='initial weights path')
    parser.add_argument('--adam', action='store_true', help='use adam optimizer')
    opt = parser.parse_args()
    opt.weights = last if opt.resume else opt.weights

    '''
    loading cfg
    '''

    cfg = configparser.RawConfigParser()
    cfg.read(cfg_name)
    lr = float(cfg.get("net", "lr"))
    model_name = cfg.get("net", "model_name")
    data_version = cfg.get("net","data_version")
    class_loss_weight = list(map(int, cfg.get("net", "class_loss_weight").split(",")))
    opt.adam = (cfg.get("net","optim") == "adam")
    print(opt.adam)
    momentum = 0.937
    weight_decay = 0.000484
    batch_size = int(cfg.get("net","batch_size"))
    epochs = int(cfg.get("net","epochs"))
    var = {}
    tuner_params = nni.get_next_parameter()
    if tuner_params:
        print(tuner_params)
        lr = tuner_params["learning_rate"]
        epochs = tuner_params["epochs"]
    var["batch_size"] = batch_size
    var["lr"] = lr
    var["model_name"] = model_name
    var["data_version"] = data_version
    var["class_loss_weight"] = cfg.get("net", "class_loss_weight")
    var["epochs"] = epochs
    var["optim"] = "adam" if opt.adam else "sgd"
    '''
    end up loading
    '''
    print(var)
    print(opt)
    device =  torch.device('cuda:0')
    # device = torch.device("cpu")
    if device.type == 'cpu':
        mixed_precision = False

    tb_writer = None
   
    print('Start Tensorboard with "tensorboard --logdir=runs", view at http://localhost:6006/')
    if tb_writer:
        tb_writer = SummaryWriter(comment="weighted_entropy_loss_resnet101")
        tb_writer.add_scalar("train/lr", lr)
    # cfg = opt.cfg
    # data = opt.data


    weights = opt.weights
    # accumulate = max(round(64 / batch_size), 1)  # accumulate n times before optimizer update (bs 64)
    # Configure run

    np.random.seed(0)
    torch.manual_seed(0)
    # data_dict = parse_data_cfg(data)

    nc = 5
    #the class num

    # Initialize model
    model = EyeNet(nc,model_name=model_name).to(device)

    # Optimizer
    pg0, pg1, pg2 = [], [], []
    for id, (k, v) in enumerate(dict(model.named_parameters()).items()):
        print(id,k)
        if id >= 0:
            if '.bias' in k:
                pg2 += [v]  # biases
            elif 'conv' in k:
                pg1 += [v]  # apply weight_decay
            else:
                pg0 += [v]  # all else
        if 312<=id <=315:
            print(v)
    if opt.adam:
        optimizer = optim.Adam(pg0, lr=lr)
    else:
        optimizer = optim.SGD(pg0, lr=lr, momentum=momentum, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
    optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
    print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    def setlr(range,lr):
        pg0, pg1, pg2 = [], [], []  # optimizer parameter groups

        for id, (k, v) in enumerate(dict(model.named_parameters()).items()):
            if range[0] <= id <= range[1]:
                if '.bias' in k:
                    pg2 += [v]  # biases
                elif 'conv' in k:
                    pg1 += [v]  # apply weight_decay
                else:
                    pg0 += [v]  # all else

        optimizer.add_param_group({'params': pg0, 'lr': lr,"momentum":momentum,"nesterov":True})
        optimizer.add_param_group({'params': pg1, 'weight_decay': weight_decay})  # add pg1 with weight_decay
        optimizer.add_param_group({'params': pg2})  # add pg2 (biases)
        print('Optimizer groups: %g .bias, %g Conv2d.weight, %g other' % (len(pg2), len(pg1), len(pg0)))

    start_epoch = 0
    if weights.endswith('.pt'):
        chkpt = torch.load(weights, map_location=device)
        chkpt['model'] = {k: v for k, v in chkpt['model'].items() if model.state_dict()[k].numel() == v.numel()}
        model.load_state_dict(chkpt['model'], strict=False)
        if chkpt['optimizer'] is not None:
            optimizer.load_state_dict(chkpt['optimizer'])

        start_epoch = chkpt['epoch'] + 1

    # Mixed precision training https://github.com/NVIDIA/apex
    if mixed_precision:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)


    # Scheduler https://arxiv.org/pdf/1812.01187.pdf
    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05  # cosine
    # lf = lambda x: 0.2**(x // 10)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10, verbose=True,
    #                                            threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0,
    #                                            eps=1e-08)

    scheduler.last_epoch = start_epoch - 1  # see link below

    # Initialize distributed training
    if device.type != 'cpu' and torch.cuda.device_count() > 1 and torch.distributed.is_available():
        dist.init_process_group(backend='nccl',  # 'distributed backend'
                                init_method='tcp://127.0.0.1:9999',  # distributed training init method
                                world_size=1,  # number of nodes for distributed training
                                rank=0)  # distributed training node rank
        model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    train_transform = transforms.Compose([

        transforms.Resize((256, 256)),

        transforms.RandomCrop((224, 224)),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),

    ])
    valid_transform = transforms.Compose([

        transforms.Resize((224, 224)),
        # transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),

    ])


    train_dataset = EyeDataset(transform=train_transform, data_version=data_version
                                )
    valid_dataset = EyeDataset(train=1, transform=valid_transform,data_version=data_version)
    test_dataset = EyeDataset(train=2, transform=valid_transform,data_version=data_version)
    print("Train data:{}.Test data:{}".format(len(train_dataset), len(valid_dataset)))
    # Dataloader
    batch_size = min(batch_size, len(train_dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers

    train_sampler = WeightedRandomSampler(train_dataset.weights, len(train_dataset), replacement=True)#reWeight according to the number of class
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=batch_size,
                                                num_workers=nw,
                                                # shuffle=not opt.rect,  # Shuffle=True unless rectangular training is used
                                                pin_memory=True,
                                                sampler=train_sampler
                                                # collate_fn=dataset.collate_fn
                                                )

    # Validloader
    validloader = torch.utils.data.DataLoader(valid_dataset,
                                                batch_size=batch_size,
                                                num_workers=nw,
                                                pin_memory=True,
                                                # collate_fn=dataset.collate_fn
                                                )
    testloader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=batch_size,
                                                num_workers=nw,
                                                pin_memory=True,
                                                # collate_fn=dataset.collate_fn
                                                )
    nb = len(trainloader)  # number of batches
    n_burn = max(3 * nb, 500)# burn-in iterations, max(3 epochs, 500 iterations)

    # opt.update(tuner_params)
    train()  # train normally
    # logs = find_lr()
    # print(logs)
