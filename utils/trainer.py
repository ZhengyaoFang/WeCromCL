import torch
from torch.autograd import Variable
import logging
import time

class NestedTensor(object):
    def __init__(self,tensor):
        self.tensor = tensor

class Trainer(object):
    def __init__(self, net, optimizer, scheduler, criterion, dataloader, converter, cfgs):
        super(Trainer, self).__init__()
        self.net = net
        self.cfg = cfgs
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.dataloader = dataloader
        self.converter = converter

    def train(self):
        logging.info('training on  ...' + self.dataset_name)
        logging.info('the init lr: %f' % (self.optimizer.param_groups[0]['lr']))
        t0 = time.time()
        for j, batch_samples in enumerate(self.dataloader):
            imgs, examples = batch_samples
            imgs = Variable(imgs.cuda())
            self.optimizer.zero_grad()
            selected_texts = [example['selected_text'] for example in examples]
            selected_labels = [self.converter.encode(selected_text)[0] for selected_text in selected_texts]
            false_labels = []
            for example in examples:
                false_labels += [self.converter.encode(false_text)[0] for false_text in example['false_texts']]
            selected_labels = torch.tensor(selected_labels).cuda()
            false_labels = torch.tensor(false_labels).cuda()
            
            selected_features = self.net.module.encode_text(selected_labels.long())
            selected_features = NestedTensor(selected_features)
            false_features = self.net.module.encode_text(false_labels.long())
            image_features = self.net(imgs, selected_features, false_features)
            
            losses, loss_names = self.criterion(self.net.module, image_features, selected_features, false_features)
            loss = sum(losses)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if (j+self.cfg.trainer.start_iter) % self.cfg.trainer.log_freq == 0:
                t1 = time.time()
                tstr = "->  iter:%6d  losses:%4.6f  " % ((j+self.start_iter), loss)
                for lname, lvalue in zip(loss_names, losses):
                    tstr += '%s:%4.6f  ' % (lname, lvalue)
                tstr += '%4.6fs/batch' % ((t1 - t0) / self.log_freq)
                logging.info(tstr)
                t0 = time.time()
            
            if (j+self.cfg.trainer.start_iter) % self.cfg.trainer.save_freq == 0:
                save_name = self.cfg.trainer.save_folder + 'iter_' + str(j+self.cfg.trainer.start_iter) + '.pth'
                torch.save(self.net.state_dict(), save_name)
                logging.info('save model: ' + save_name)
                logging.info('current lr: %f' % (self.optimizer.param_groups[0]['lr']))
            
            if (j+self.cfg.trainer.start_iter) >= self.cfg.trainer.niter:
                save_name = self.cfg.trainer.save_folder + 'iter_' + str(j+self.cfg.trainer.start_iter) + '.pth'
                torch.save(self.net.state_dict(), save_name)
                logging.info('The training stage on %s is over!' % (self.dataset_name))
                break

            