import itertools
import torch
import os
from torch import nn
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, SVHN
from torch.utils.data import Dataset, DataLoader
import torchvision
import time
import pandas as pd
import wandb

from libs.models import define_D, define_G, Classifier, Classifier_resnet, G12, G21, D1, D2
from libs.loader import load_pict, concat_dataset, dataset_sampler
from libs.meter import AverageMeter, ProgressMeter, accuracy

class ACAL(object):
    def __init__(self, args):
        self.device = "cuda"
        # networks
        if args.data == "original":
            self.Gst = define_G(args.input_nc, args.output_nc, args.ngf, args.g_model).to(self.device)
            self.Gts = define_G(args.input_nc, args.output_nc, args.ngf, args.g_model).to(self.device)
            self.Ds = define_D(args.input_nc, args.ndf, args.d_model).to(self.device)
            self.Dt = define_D(args.input_nc, args.ndf, args.d_model).to(self.device)
            self.Cs = Classifier_resnet(args.n_class).to(self.device)
            self.Ct = Classifier_resnet(args.n_class).to(self.device)
        elif args.data == "digit":
            self.Gst = G21(conv_dim=64).to(self.device)
            self.Gts = G12(conv_dim=64).to(self.device)
            self.Ds = D2(conv_dim=64, use_labels=args.use_labels, minibatch_d=args.minibatch_d).to(self.device)
            self.Dt = D1(conv_dim=64, use_labels=args.use_labels, minibatch_d=args.minibatch_d).to(self.device)
            self.Cs = Classifier(args.n_class, args.s_input_nc).to(self.device)
            self.Ct = Classifier(args.n_class, args.t_input_nc).to(self.device)
        if args.resume:
            self.Gst.load_state_dict(torch.load(os.path.join(args.model_path, "best_acc1_model_gst.prm")))
            self.Gts.load_state_dict(torch.load(os.path.join(args.model_path, "best_acc1_model_gts.prm")))
            self.Ds.load_state_dict(torch.load(os.path.join(args.model_path, "best_acc1_model_ds.prm")))
            self.Dt.load_state_dict(torch.load(os.path.join(args.model_path, "best_acc1_model_dt.prm")))
            self.Cs.load_state_dict(torch.load(os.path.join(args.model_path, "best_acc1_model_cs.prm")))
            self.Ct.load_state_dict(torch.load(os.path.join(args.model_path, "best_acc1_model_ct.prm")))
        # self.Cs = Classifier_resnet(args.n_class).to(self.device)
        # self.Ct = Classifier_resnet(args.n_class).to(self.device)
        # for prm in self.Cs.resnet.parameters():
        #     prm.requires_grad=False
        # for prm in self.Ct.resnet.parameters():
        #     prm.requires_grad=False

        

        # losses
        self.CE = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()

        # optimizers
        self.g_optimizer = torch.optim.Adam(itertools.chain(self.Gst.parameters(), self.Gts.parameters(), 
        self.Cs.parameters(), self.Ct.parameters()), lr=args.lr, betas=(0.5, 0.999))
        self.d_optimizer = torch.optim.Adam(itertools.chain(self.Ds.parameters(), self.Dt.parameters()), lr=args.lr, betas=(0.5, 0.999))
        if args.pre_optimizer == "Adam":
            self.cs_optimizer = torch.optim.Adam(self.Cs.parameters(), lr=args.pre_lr, betas=(0.5, 0.999))
        elif args.pre_optimizer == "sgd":
            self.cs_optimizer = torch.optim.SGD(self.Cs.parameters(), lr=args.pre_lr)
        #self.g_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.g_optimizer)
        #self.d_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.d_optimizer)
        #if args.train_cls:
        #   self.c_optimizer = torch.optim.Adam(self.Cs.parameters())

        #Try loading checkpoint
        try:
            self.Cs.load_state_dict(torch.load(os.path.join(args.model_path, "best_pretrained_classifier.prm")))
            print("load pretrained model.")
        except:
            print("no pretrained model found.")    

        # if not os.path.isdir(args.checkpoint_dir):
        #     os.makedirs(args.checkpoint_dir)

        # try:
        #     ckpt = utils.load_checkpoint('%s/latest.ckpt' % (args.checkpoint_dir))
        #     self.start_epoch = ckpt['epoch']
        #     self.Da.load_state_dict(ckpt['Da'])
        #     self.Db.load_state_dict(ckpt['Db'])
        #     self.Gab.load_state_dict(ckpt['Gab'])
        #     self.Gba.load_state_dict(ckpt['Gba'])
        #     self.d_optimizer.load_state_dict(ckpt['d_optimizer'])
        #     self.g_optimizer.load_state_dict(ckpt['g_optimizer'])
        # except:
        #     print(' [*] No checkpoint!')
        #     self.start_epoch = 0
    def pretrain(self, args, ft=False):

        #dataload
        if args.data == "digit":
            tr_transform_s = transforms.Compose([
            # [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
             transforms.Resize((args.resolution,args.resolution)),
             #transforms.RandomResizedCrop(size=(args.resolution,args.resolution), scale=(0.8, 1.0)),
             transforms.ToTensor(),
             #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
             ])
            tr_transform_m = transforms.Compose([
            # [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
             transforms.Resize((args.resolution,args.resolution)),
             #transforms.RandomResizedCrop(size=(args.resolution,args.resolution), scale=(0.8, 1.0)),
             transforms.ToTensor(),
             #transforms.Normalize(mean=[0.5], std=[0.5])
             ])

            te_transform_s = transforms.Compose(
            [transforms.Resize((args.resolution,args.resolution)),
             transforms.ToTensor(),
             #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
             ])
            te_transform_m = transforms.Compose(
            [transforms.Resize((args.resolution,args.resolution)),
             transforms.ToTensor(),
             #transforms.Normalize(mean=[0.5], std=[0.5])
             ])
            s_trainset = SVHN("datasets/svhn", split="train", download=True, transform=tr_transform_s)
            s_testset = SVHN("datasets/svhn", split="test", download=True, transform=te_transform_s)
            t_trainset = dataset_sampler(MNIST("datasets/mnist", train=True, download=True, transform=tr_transform_m), data_per_class=args.dpc)
            t_testset = MNIST("datasets/mnist", train=False, download=True, transform=te_transform_m)
            trainloader = DataLoader(s_trainset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            testloader = DataLoader(s_testset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            t_trainloader = DataLoader(t_trainset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True)
            t_testloader = DataLoader(t_testset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        elif args.data == "original":
            transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.resolution,args.resolution)),
             transforms.RandomResizedCrop(size=(args.resolution,args.resolution)),
             transforms.ToTensor()
             ])
            s_trainset = load_pict(os.path.join(args.csv_path, args.source_name +"_train.csv"), transform=transform)
            s_testset = load_pict(os.path.join(args.csv_path, args.source_name +"_test.csv"), transform=transform)
            trainloader = DataLoader(s_trainset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            testloader = DataLoader(s_testset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        #log
        best_acc = 0
        log = pd.DataFrame(
            columns=[
                "epoch", 
                "s_loss", "s_acc", "t_loss", "t_acc"
            ]
        )
        
        for epoch in range(args.pre_epoch):
            end = time.time()
            # train  
            #################################
            
            # meter
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            tr_loss = AverageMeter('Loss', ':.4e')
            tr_acc = AverageMeter('Acc', ':6.3f')
            progress = ProgressMeter(
                len(trainloader),
                [batch_time, data_time, tr_loss, tr_acc],
                prefix="Epoch: [{}]".format(epoch)
            )
            self.Cs.train()
            for i, sample in enumerate(trainloader):
                data_time.update(time.time() - end)
                data, label = sample
                data, label = data.to(self.device), label.to(self.device)
                tmp_batchsize = data.shape[0]
                self.cs_optimizer.zero_grad()
                pred = self.Cs(data)
                loss = self.CE(pred, label)
                loss.backward()
                self.cs_optimizer.step()
                acc1 = accuracy(pred, label, topk=(1,))
                tr_loss.update(loss.item(), n=tmp_batchsize)
                tr_acc.update(acc1[0].item(), n=tmp_batchsize)
                batch_time.update(time.time() - end)
                if i != 0 and i % 100 == 0:
                    progress.display(1)
                end = time.time()

            #train on target sample if ft=True
            if ft:
                for i, sample in enumerate(t_trainloader):
                    data_time.update(time.time() - end)
                    data, label = sample
                    data, label = self.gs2rgb(data.to(self.device)), label.to(self.device)
                    tmp_batchsize = data.shape[0]
                    self.cs_optimizer.zero_grad()
                    pred = self.Cs(data)
                    loss = self.CE(pred, label)
                    loss.backward()
                    self.cs_optimizer.step()
                    acc1 = accuracy(pred, label, topk=(1,))
                    tr_loss.update(loss.item(), n=tmp_batchsize)
                    tr_acc.update(acc1[0].item(), n=tmp_batchsize)
                    batch_time.update(time.time() - end)
                    if i != 0 and i % 100 == 0:
                        progress.display(1)
                    end = time.time()
    

            #validate
            ##################################
            
            # meter
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            s_val_loss = AverageMeter('Loss', ':.4e')
            s_val_acc = AverageMeter('Acc', ':6.3f')
            t_val_loss = AverageMeter('Loss', ':.4e')
            t_val_acc = AverageMeter('Acc', ':6.3f')


            for i, sample in enumerate(testloader):
                data, label = sample
                data, label = data.to(self.device), label.to(self.device)
                tmp_batchsize = data.shape[0]
                pred = self.Cs(data)
                loss = self.CE(pred, label)
                loss.backward()
                acc1 = accuracy(pred, label)
                s_val_loss.update(loss.item(), n=tmp_batchsize)
                s_val_acc.update(acc1[0].item(), n=tmp_batchsize)
                end = time.time()

            for i, sample in enumerate(t_testloader):
                data, label = sample
                data, label = self.gs2rgb(data.to(self.device)), label.to(self.device)
                tmp_batchsize = data.shape[0]
                pred = self.Cs(data)
                loss = self.CE(pred, label)
                loss.backward()
                acc1 = accuracy(pred, label)
                t_val_loss.update(loss.item(), n=tmp_batchsize)
                t_val_acc.update(acc1[0].item(), n=tmp_batchsize)
                end = time.time()

            s_acc = s_val_acc.avg
            s_loss = s_val_loss.avg
            t_acc = t_val_acc.avg
            t_loss = t_val_loss.avg

            #save models
            if best_acc < s_acc:
                best_acc = s_acc
                torch.save(
                    self.Cs.state_dict(),
                    os.path.join(args.result_path, 'best_pretrained_classifier.prm')
                )
            #record
            tmp = pd.Series([
                epoch,
                s_loss, s_acc, t_loss, t_acc
            ], index=log.columns
            )

            log = log.append(tmp, ignore_index=True)
            log.to_csv(os.path.join(args.result_path, 'log_pretrain.csv'), index=False)
            
            print(
                'epoch: {}\ts_val_loss: {:.4f}\ts_val_acc: {:.5f}t_val_loss: {:.4f}\tt_val_acc: {:.5f}'
                .format(epoch, s_loss, s_acc, t_loss, t_acc)
            )
    def reset_grad(self, c=False):
        self.d_optimizer.zero_grad()
        self.g_optimizer.zero_grad()
        if c:
            self.cs_optimizer.zero_grad()
            self.cs_optimizer.zero_grad()

    def gs2rgb(self, input):
        return(torch.cat([input, input, input], dim=1))

    def rgb2gs(self, input):
        return(torch.unsqueeze(input[:, 0, :, :] / 3+ input[:, 1, :, :] / 3 + input[:, 2, :, :] / 3, 1))

    def train(self, args):
        #wandb codes
        wandb.init(project=args.project_name, name=args.log_name, config=args)
        wandb.watch(self.Gst)
        wandb.watch(self.Gts)
        wandb.watch(self.Cs)
        wandb.watch(self.Ct)
        #dataload
        if args.data == "digit":
            transform_m = transforms.Compose(
            [transforms.Resize((args.resolution,args.resolution)),
             transforms.ToTensor()
             #transforms.Normalize(mean=[0.5], std=[0.5])
             ])
            transform_s = transforms.Compose(
            [transforms.Resize((args.resolution,args.resolution)),
             transforms.ToTensor()
             #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
             ])
            s_trainset = SVHN("datasets/svhn", split="train", download=True, transform=transform_s)
            s_testset = SVHN("datasets/svhn", split="test", download=True, transform=transform_s)
            # s_trainset = MNIST("datasets/mnist", train=True, download=True, transform=transform_m)
            # s_testset = MNIST("datasets/mnist", train=False, download=True, transform=transform_m)
            if args.few_shot:
                t_trainset = dataset_sampler(MNIST("datasets/mnist", train=True, download=True, transform=transform_m), data_per_class=args.dpc)
            else:
                t_trainset = MNIST("datasets/mnist", train=True, download=True, transform=transform_m)
            t_testset = MNIST("datasets/mnist", train=False, download=True, transform=transform_m)
            trainset = concat_dataset(s_trainset, t_trainset)
            testset = concat_dataset(s_testset, t_testset)
            trainloader = DataLoader(trainset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True)
            testloader = DataLoader(testset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        
        elif args.data == "original":
            transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.resolution,args.resolution)),
             transforms.RandomResizedCrop(size=(args.resolution,args.resolution)),
             transforms.ToTensor()
             ])
            s_trainset = load_pict(os.path.join(args.csv_path, args.source_name +"_train.csv"), transform=transform)
            s_testset = load_pict(os.path.join(args.csv_path, args.source_name +"_test.csv"), transform=transform)
            if args.few_shot:
                t_trainset = dataset_sampler(load_pict(os.path.join(args.csv_path, args.target_name + "_train.csv"), transform=transform) ,class_num=args.n_class , data_per_class=args.dpc)
            else:
                t_trainset = load_pict(os.path.join(args.csv_path, args.target_name + "_train.csv"), transform=transform)
            t_testset = load_pict(os.path.join(args.csv_path, args.source_name +"_test.csv"), transform=transform)
            trainset = concat_dataset(s_trainset, t_trainset)
            testset = concat_dataset(s_testset, t_testset)
            trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            testloader = DataLoader(testset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # s_trainloader = DataLoader(s_trainset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # s_testloader = DataLoader(s_testset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # t_trainloader = DataLoader(t_trainset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # t_testloader = DataLoader(t_testset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        
        #log
        best_acc_t = 0
        log = pd.DataFrame(
            columns=[
                "epoch", 
                "s_loss", "t_loss", "s_acc", "t_acc"
            ]
        )
        
        for epoch in range(args.num_epoch):
            end = time.time()
            # average meter
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_d = AverageMeter('Loss_d', ':.4e')
            losses_g = AverageMeter('Loss_g', ':.4e')
            losses_g_augcls = AverageMeter('Loss_g_augcls', ':.4e')
            losses_g_idt = AverageMeter('Loss_g_idt', ':.4e')
            losses_g_recon = AverageMeter('Loss_g_recon', ':.4e')
            losses_g_adv = AverageMeter('Loss_g_adv', ':.4e')
            losses_cs = AverageMeter('Loss_cs', ':.4e')
            losses_ct = AverageMeter('Loss_ct', ':.4e')
            # progress meter
            progress = ProgressMeter(
                len(trainloader),
                [batch_time, data_time, losses_d, losses_g, losses_cs, losses_ct],
                prefix="Epoch: [{}]".format(epoch)
            )
            for i, sample in enumerate(trainloader):
                self.reset_grad()
                s_data, s_label, t_data, t_label = sample["s_data"], sample["s_label"], sample["t_data"], sample["t_label"]
                s_data, s_label, t_data, t_label = s_data.to(self.device), s_label.to(self.device), t_data.to(self.device), t_label.to(self.device)
                data_time.update(time.time() - end)
                fake_t_data = self.Gst(s_data)
                fake_s_data = self.Gts(t_data)
                cyc_s_data = self.Gts(fake_t_data)
                cyc_t_data = self.Gst(fake_s_data)
                
                ######################
                #train discriminator #
                ######################
                tmp_batchsize = s_data.shape[0]
                label_shape = self.Ds(s_data).detach().shape
                ones_label = torch.ones(label_shape).to(self.device)
                zeros_label = torch.zeros(label_shape).to(self.device)
                #discriminate fake data
                # if args.use_labels:
                #     loss_d_s = self.CE(self.Ds(s_data), s_label) + self.CE(self.Ds(fake_s_data), zeros_label)
                #     loss_d_t = self.CE(self.Dt(t_data), t_label) + self.CE(self.Dt(fake_t_data), zeros_label)
                # else:
                #     loss_d_s = self.MSE(self.Ds(s_data), ones_label) + self.MSE(self.Ds(fake_s_data.detach()), zeros_label)
                #     loss_d_t = self.MSE(self.Dt(t_data), ones_label) + self.MSE(self.Dt(fake_t_data.detach()), zeros_label)
                # loss_d = loss_d_s + loss_d_t
                if args.use_labels:
                    loss_d_real = self.MSE(self.Ds(s_data), s_label) + self.MSE(self.Dt(t_data), t_label)
                    loss_d_fake = self.MSE(self.Ds(fake_s_data), zeros_label) + self.MSE(self.Dt(fake_t_data), zeros_label)
                else:
                    loss_d_real = self.MSE(self.Ds(s_data), ones_label) + self.MSE(self.Dt(t_data), ones_label)
                    loss_d_fake = self.MSE(self.Ds(fake_s_data), zeros_label) + self.MSE(self.Dt(fake_t_data), zeros_label)
                loss_d_real.backward(retain_graph=True)
                self.d_optimizer.step()
                self.d_optimizer.zero_grad()
                loss_d_fake.backward(retain_graph=True)
                self.d_optimizer.step()
                losses_d.update(loss_d_real.item() + loss_d_fake.item(), n=tmp_batchsize)

                ###################
                #train generator  #
                ###################
                for j in range(args.num_k):
                    if j > 0:
                        self.reset_grad()
                        fake_t_data = self.Gst(s_data)
                        fake_s_data = self.Gts(t_data)
                        cyc_s_data = self.Gts(fake_t_data)
                        cyc_t_data = self.Gst(fake_s_data)
                    #classification loss
                    #normal classification loss
                    loss_cs = self.CE(self.Cs(s_data), s_label)
                    loss_ct = self.CE(self.Ct(t_data), t_label)
                    #augmented classification loss
                    loss_g_aug_s = self.CE(self.Ct(fake_t_data), s_label) + self.CE(self.Cs(cyc_s_data), s_label)
                    loss_g_aug_t = self.CE(self.Cs(fake_s_data), t_label) + self.CE(self.Ct(cyc_t_data), t_label)
                    #decieve discriminator
                    if args.adversarial == "reverse":
                        loss_g_adv = self.MSE(self.Ds(fake_s_data), ones_label) + self.MSE(self.Dt(fake_t_data), ones_label)
                    elif args.adversarial == "minus":
                        loss_g_adv = -self.MSE(self.Ds(fake_s_data), zeros_label) -self.MSE(self.Dt(fake_t_data), zeros_label)  #max(logD) instead of min(1-logD)
                    
                    loss_g = loss_cs + loss_ct + args.alpha*(loss_g_aug_s + loss_g_aug_t) + args.beta*(loss_g_adv)
                    #identity construction loss
                    if args.idt_loss:
                        loss_idt_s = self.MSE(self.Gts(self.gs2rgb(s_data)), s_data)
                        loss_idt_t = self.MSE(self.Gst(self.rgb2gs(t_data)), t_data)
                        loss_g += loss_idt_s
                        loss_g += loss_idt_t
                    if args.recon_loss:
                        loss_g_recon = self.MSE(s_data, cyc_s_data) + self.MSE(t_data, cyc_t_data)
                        loss_g += loss_g_recon
                    loss_g.backward()
                    self.g_optimizer.step()
                    losses_g.update(loss_g.item(), n=tmp_batchsize)
                    losses_g_augcls.update(loss_g_aug_s.item() + loss_g_aug_t.item(), n=tmp_batchsize)
                    losses_g_adv.update(loss_g_adv.item(), n=tmp_batchsize)
                    losses_cs.update(loss_cs.item(), n=tmp_batchsize)
                    losses_ct.update(loss_ct.item(), n=tmp_batchsize)
                    if args.recon_loss:
                        losses_g_recon.update(loss_g_recon.item(), n=tmp_batchsize)
                    if args.idt_loss:
                        losses_g_idt.update(loss_idt_s.item() + loss_idt_t.item(), n=tmp_batchsize)
                
                if i != 0 and i % 100 == 0:
                    progress.display(1)
                end = time.time()
            #----------------wandb visualisation------------------------------------------------
            s_example_images = [wandb.Image(s_data[0], caption="source original data"),
                                wandb.Image(fake_t_data[0], caption="source converted data"),
                                wandb.Image(cyc_s_data[0], caption="source cycled data")]
            t_example_images = [wandb.Image(t_data[0], caption="target original data"),
                                wandb.Image(fake_s_data[0], caption="target converted data"),
                                wandb.Image(cyc_t_data[0], caption="target cycled data")]
            
            wandb.log({"Gst&Gts_loss" : losses_g.avg})
            wandb.log({"G augmented classification loss" : losses_g_augcls.avg})
            wandb.log({"G idt loss" : losses_g_idt.avg})
            wandb.log({"G reconstruct loss" : losses_g_recon.avg})
            wandb.log({"G adv loss" : losses_g_adv.avg})
            wandb.log({"D_loss" : losses_d.avg})
            if epoch % 20 == 0:
                wandb.log({"epoch{} Source Images".format(epoch) : s_example_images})
                wandb.log({"epoch{} Target Images".format(epoch) : t_example_images})
            #-----------------------------------------------------------------------------------

            #evaluate
            with torch.no_grad():
                val_losses_s = AverageMeter('Loss_s', ':.4e')
                val_losses_t = AverageMeter('Loss_t', ':.4e')
                acc_s = AverageMeter('acc_s', ':6.3f')
                acc_t = AverageMeter('acc_t', ':6.3f')
                for sample in testloader:
                    s_data, s_label, t_data, t_label = sample["s_data"], sample["s_label"], sample["t_data"], sample["t_label"]
                    s_data, s_label, t_data, t_label = s_data.to(self.device), s_label.to(self.device), t_data.to(self.device), t_label.to(self.device)
                    tmp_batchsize = s_data.shape[0]
                    s_pred = self.Cs(s_data)
                    t_pred = self.Ct(t_data)
                    val_losses_s.update(self.CE(s_pred, s_label).item(), n=tmp_batchsize)
                    val_losses_t.update(self.CE(t_pred, t_label).item(), n=tmp_batchsize)
                    acc_s.update(accuracy(s_pred, s_label)[0].item(), n=tmp_batchsize)
                    acc_t.update(accuracy(t_pred, t_label)[0].item(), n=tmp_batchsize)
            
            s_val_loss = val_losses_s.avg
            t_val_loss = val_losses_t.avg
            s_val_acc = acc_s.avg
            t_val_acc = acc_t.avg

            #wandb logging
            wandb.log({"Cs_val_acc" : s_val_acc})
            wandb.log({"Ct_val_acc" : t_val_acc})
            

            #save models
            if best_acc_t < t_val_acc:
                best_acc_t = t_val_acc
                torch.save(
                    self.Ct.state_dict(),
                    os.path.join(args.result_path, 'best_acc1_model_ct.prm')
                )
                torch.save(
                    self.Ct.state_dict(),
                    os.path.join(args.result_path, 'best_acc1_model_cs.prm')
                )
                torch.save(
                    self.Gst.state_dict(),
                    os.path.join(args.result_path, 'best_acc1_model_gst.prm')
                )
                torch.save(
                    self.Gts.state_dict(),
                    os.path.join(args.result_path, 'best_acc1_model_gts.prm')
                )
                torch.save(
                    self.Ds.state_dict(),
                    os.path.join(args.result_path, 'best_acc1_model_ds.prm')
                )
                torch.save(
                    self.Dt.state_dict(),
                    os.path.join(args.result_path, 'best_acc1_model_dt.prm')
                )

            #record
            tmp = pd.Series([
                epoch,
                s_val_loss, t_val_loss,
                s_val_acc, t_val_acc
            ], index=log.columns
            )

            log = log.append(tmp, ignore_index=True)
            log.to_csv(os.path.join(args.result_path, 'log_step1.csv'), index=False)
            
            print(
                'epoch: {}\ts_val loss: {:.4f}\tt_val loss: {:.4f}\ts_val_acc: {:.5f}\tt_val_acc: {:.5f}'
                .format(epoch,
                        s_val_loss, t_val_loss,
                        s_val_acc, t_val_acc)
            )

    def train_recon(self, args):
        # no classification version

        #wandb codes
        wandb.init(project="acal_office-home", name=args.log_name)
        wandb.watch(self.Gst)
        wandb.watch(self.Gts)
        #dataload
        if args.data == "mnist":
            transform_m = transforms.Compose(
            [transforms.Resize((args.resolution,args.resolution)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5], std=[0.5])
             ])
            transform_s = transforms.Compose(
            [transforms.Resize((args.resolution,args.resolution)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
             ])
            s_trainset = MNIST("datasets/mnist", train=True, download=True, transform=transform_m)
            s_testset = MNIST("datasets/mnist", train=False, download=True, transform=transform_m)
            if args.few_shot:
                t_trainset = dataset_sampler(SVHN("datasets/svhn", split="train", download=True, transform=transform_s))
                t_testset = dataset_sampler(SVHN("datasets/svhn", split="test", download=True, transform=transform_s))
            else:
                t_trainset = SVHN("datasets/svhn", split="train", download=True, transform=transform_s)
                t_testset = SVHN("datasets/svhn", split="test", download=True, transform=transform_s)
            trainset = concat_dataset(s_trainset, t_trainset)
            testset = concat_dataset(s_testset, t_testset)
            trainloader = DataLoader(trainset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            testloader = DataLoader(testset, batch_size = args.pre_batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        elif args.data == "original":
            transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(),
             transforms.Resize((args.resolution,args.resolution)),
             transforms.RandomResizedCrop(size=(args.resolution,args.resolution)),
             transforms.ToTensor()
             ])
            s_trainset = load_pict(os.path.join(args.csv_path, args.source_name +"_train.csv"), transform=transform)
            s_testset = load_pict(os.path.join(args.csv_path, args.source_name +"_test.csv"), transform=transform)
            t_trainset = load_pict(os.path.join(args.csv_path, args.target_name + "_train.csv"), transform=transform)
            t_testset = load_pict(os.path.join(args.csv_path, args.source_name +"_test.csv"), transform=transform)
            trainset = concat_dataset(s_trainset, t_trainset)
            testset = concat_dataset(s_testset, t_testset)
            trainloader = DataLoader(trainset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            testloader = DataLoader(testset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # s_trainloader = DataLoader(s_trainset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # s_testloader = DataLoader(s_testset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # t_trainloader = DataLoader(t_trainset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
            # t_testloader = DataLoader(t_testset, batch_size = args.batch_size, shuffle = True, num_workers=args.num_workers, pin_memory=True, drop_last=True)

        #log
        best_acc_t = 0
        log = pd.DataFrame(
            columns=[
                "epoch", 
                "s_loss", "t_loss", "s_acc", "t_acc"
            ]
        )
        
        for epoch in range(args.num_epoch):
            end = time.time()
            # average meter
            batch_time = AverageMeter('Time', ':6.3f')
            data_time = AverageMeter('Data', ':6.3f')
            losses_d = AverageMeter('Loss_d', ':.4e')
            losses_g = AverageMeter('Loss_g', ':.4e')
            losses_g_idt = AverageMeter('Loss_g_idt', ':.4e')
            losses_g_recon = AverageMeter('Loss_g_recon', ':.4e')
            losses_g_adv = AverageMeter('Loss_g_adv', ':.4e')
            # progress meter
            progress = ProgressMeter(
                len(trainloader),
                [batch_time, data_time, losses_d, losses_g],
                prefix="Epoch: [{}]".format(epoch)
            )
            for i, sample in enumerate(trainloader):
                self.reset_grad()
                s_data, s_label, t_data, t_label = sample["s_data"], sample["s_label"], sample["t_data"], sample["t_label"]
                s_data, s_label, t_data, t_label = s_data.to(self.device), s_label.to(self.device), t_data.to(self.device), t_label.to(self.device)
                data_time.update(time.time() - end)
                fake_t_data = self.Gst(s_data)
                fake_s_data = self.Gts(t_data)
                cyc_s_data = self.Gts(fake_t_data)
                cyc_t_data = self.Gst(fake_s_data)
                
                ######################
                #train discriminator #
                ######################
                tmp_batchsize = s_data.shape[0]
                label_shape = self.Ds(s_data).detach().shape
                ones_label = torch.ones(label_shape).to(self.device)
                zeros_label = torch.zeros(label_shape).to(self.device)
                #discriminate fake data
                loss_d_s = self.MSE(self.Ds(s_data), ones_label) + self.MSE(self.Ds(fake_s_data.detach()), zeros_label)
                loss_d_t = self.MSE(self.Dt(t_data), ones_label) + self.MSE(self.Dt(fake_t_data.detach()), zeros_label)
                loss_d = loss_d_s + loss_d_t
                loss_d.backward()
                self.d_optimizer.step()
                losses_d.update(loss_d.item(), n=tmp_batchsize)

                for i in range(args.num_k):
                    ###################
                    #train generator  #
                    ###################
                    #reconstruction loss
                    loss_g_recon = self.MSE(s_data, cyc_s_data) + self.MSE(t_data, cyc_t_data)
                    #decieve discriminator
                    loss_g_adv = self.MSE(self.Ds(fake_s_data), ones_label) + self.MSE(self.Dt(fake_t_data), ones_label)
                    loss_g = loss_g_recon + loss_g_adv
                    #identity construction loss
                    if args.idt_loss:
                        if args.data == "mnist":
                            loss_idt_s = self.MSE(self.Gts(self.gs2rgb(s_data)), s_data)
                            loss_idt_t = self.MSE(self.Gst(self.rgb2gs(t_data)), t_data)
                        elif args.data == "original":
                            loss_idt_s = self.MSE(self.Gts(s_data), s_data)
                            loss_idt_t = self.MSE(self.Gst(t_data), t_data)
                        loss_g += loss_idt_s
                        loss_g += loss_idt_t
                    loss_g.backward()
                    self.g_optimizer.step()
                    losses_g.update(loss_g.item(), n=tmp_batchsize)
                    losses_g_adv.update(loss_g_adv.item(), n=tmp_batchsize)
                    losses_g_recon.update(loss_g_recon.item(), n=tmp_batchsize)
                    if args.idt_loss:
                        losses_g_idt.update(loss_idt_s.item() + loss_idt_t.item(), n=tmp_batchsize)                    
                
                if i != 0 and i % 100 == 0:
                    progress.display(1)
                end = time.time()
                    

            #----------------wandb visualisation------------------------------------------------
            s_example_images = [wandb.Image(s_data[0], caption="source original data"),
                                wandb.Image(fake_t_data[0], caption="source converted data"),
                                wandb.Image(cyc_s_data[0], caption="source cycled data")]
            t_example_images = [wandb.Image(t_data[0], caption="target original data"),
                                wandb.Image(fake_s_data[0], caption="target converted data"),
                                wandb.Image(cyc_t_data[0], caption="target cycled data")]
            
            wandb.log({"Gst&Gts_loss" : losses_g.avg})
            wandb.log({"G idt loss" : losses_g_idt.avg})
            wandb.log({"G reconstruct loss" : losses_g_recon.avg})
            wandb.log({"G adv loss" : losses_g_adv.avg})
            wandb.log({"D_loss" : losses_d.avg})
            if epoch % 20 == 0:
                wandb.log({"epoch{} Source Images".format(epoch) : s_example_images})
                wandb.log({"epoch{} Target Images".format(epoch) : t_example_images})
            #-----------------------------------------------------------------------------------

            
            #save models
            torch.save(
                self.Gst.state_dict(),
                os.path.join(args.result_path, 'best_acc1_model_gst.prm')
            )
            torch.save(
                self.Gts.state_dict(),
                os.path.join(args.result_path, 'best_acc1_model_gts.prm')
            )
            
            print(
                'epoch: {}\tg loss: {:.4f}\td loss: {:.4f}'
                .format(epoch,
                        losses_g.avg, losses_d.avg)
            )




"""
argsの変数；ngf, ndf, g_model, d_model, gpu_id, n_class, csv_path, source_name, target_name, num_epoch, resolution
batch_size, num_workers, result_path

"""