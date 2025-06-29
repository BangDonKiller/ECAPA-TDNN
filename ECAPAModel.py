'''
This part is used to train the speaker model and evaluate the performances
'''

import torch, sys, os, tqdm, numpy, soundfile, time, pickle
import torch.nn as nn
from tools import *
from loss import AAMsoftmax
from model import ECAPA_TDNN
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class ECAPAModel(nn.Module):
    def __init__(self, lr, lr_decay, C , n_class, m, s, test_step, **kwargs):
        super(ECAPAModel, self).__init__()
        ## ECAPA-TDNN
        self.speaker_encoder = ECAPA_TDNN(C = C).cuda()
        ## Classifier
        self.speaker_loss    = AAMsoftmax(n_class = n_class, m = m, s = s).cuda()

        self.optim           = torch.optim.Adam(self.parameters(), lr = lr, weight_decay = 2e-5)
        self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = test_step, gamma=lr_decay)
          
        self.count = 1
        
        # self.init_tensorboard()
  
    def train_network(self, epoch, loader):
        self.train()
        self.scheduler.step(epoch - 1)
        index, top1, loss = 0, 0, 0
        lr = self.optim.param_groups[0]['lr']
        for num, (data, labels) in enumerate(tqdm(loader, desc=f"Epoch {epoch}", ncols=100), start=1):
            self.zero_grad()
            labels = torch.LongTensor(labels).cuda()
            speaker_embedding = self.speaker_encoder.forward(data.cuda(), aug=True)
            nloss, prec = self.speaker_loss.forward(speaker_embedding, labels)
            nloss.backward()
            self.optim.step()

            index += len(labels)
            top1 += prec
            loss += nloss.detach().cpu().numpy()

        avg_loss = loss / num
        avg_acc = top1 / index * len(labels)

        # 🔥 呼叫 TensorBoard logging
        # self.log_tensorboard(epoch, avg_loss, avg_acc, None, None)

        return avg_loss, lr, avg_acc

    def init_tensorboard(self):
        while True:
            if not os.path.exists(f"logs/exp{self.count}"):
                os.makedirs(f"logs/exp{self.count}")
                break
            self.count += 1
        self.writer = SummaryWriter(comment="ECAPA-TDNN", log_dir=f"logs/exp{self.count}")
        
        
    def eval_network(self, eval_list, eval_path):
        self.eval()
        files = []
        embeddings = {}
        lines = open(eval_list).read().splitlines()
        for line in lines:
            files.append(line.split()[1])
            files.append(line.split()[2])
        setfiles = list(set(files))
        setfiles.sort()

        for idx, file in tqdm(enumerate(setfiles), total = len(setfiles)):
            for path in eval_path:
                if os.path.exists(os.path.join(path, file)):
                    target_path = path
                    break
            audio, _  = soundfile.read(os.path.join(target_path, file))
            # Full utterance
            data_1 = torch.FloatTensor(numpy.stack([audio],axis=0)).cuda()

            # Spliited utterance matrix
            max_audio = 300 * 160 + 240
            if audio.shape[0] <= max_audio:
                shortage = max_audio - audio.shape[0]
                audio = numpy.pad(audio, (0, shortage), 'wrap')
            feats = []
            startframe = numpy.linspace(0, audio.shape[0]-max_audio, num=5)
            for asf in startframe:
                feats.append(audio[int(asf):int(asf)+max_audio])
            feats = numpy.stack(feats, axis = 0).astype(float)
            data_2 = torch.FloatTensor(feats).cuda()
            # Speaker embeddings
            with torch.no_grad():
                embedding_1 = self.speaker_encoder.forward(data_1, aug = False)
                embedding_1 = F.normalize(embedding_1, p=2, dim=1)
                embedding_2 = self.speaker_encoder.forward(data_2, aug = False)
                embedding_2 = F.normalize(embedding_2, p=2, dim=1)
            embeddings[file] = [embedding_1, embedding_2]
        scores, labels  = [], []

        for line in lines:			
            embedding_11, embedding_12 = embeddings[line.split()[1]]
            embedding_21, embedding_22 = embeddings[line.split()[2]]
            # Compute the scores
            score_1 = torch.mean(torch.matmul(embedding_11, embedding_21.T)) # higher is positive
            score_2 = torch.mean(torch.matmul(embedding_12, embedding_22.T))
            score = (score_1 + score_2) / 2
            score = score.detach().cpu().numpy()
            scores.append(score)
            labels.append(int(line.split()[0]))
            
        # Coumpute EER and minDCF
        EER = tuneThresholdfromScore(scores, labels, [1, 0.1])[1]
        fnrs, fprs, thresholds = ComputeErrorRates(scores, labels)
        minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)

        # self.log_tensorboard(epoch, None, None, EER, minDCF)

        return EER, minDCF

    def save_parameters(self, path):
        torch.save(self.state_dict(), path)

    def load_parameters(self, path):
        self_state = self.state_dict()
        loaded_state = torch.load(path)
        for name, param in loaded_state.items():
            origname = name
            if name not in self_state:
                name = name.replace("module.", "")
                if name not in self_state:
                    print("%s is not in the model."%origname)
                    continue
            if self_state[name].size() != loaded_state[origname].size():
                print("Wrong parameter length: %s, model: %s, loaded: %s"%(origname, self_state[name].size(), loaded_state[origname].size()))
                continue
            self_state[name].copy_(param)

    def log_tensorboard(self, epoch, loss, acc, EER, minDCF):
        if loss is not None:
            self.writer.add_scalar("Loss/train", loss, epoch)
        if acc is not None:
            self.writer.add_scalar("Accuracy/train", acc, epoch)
        if EER is not None:
            self.writer.add_scalar("EER/train", EER, epoch)
        if minDCF is not None:
            self.writer.add_scalar("minDCF/train", minDCF, epoch)
