'''
DataLoader for training
'''

import glob, numpy, os, random, soundfile, torch
from scipy import signal
from pydub import AudioSegment
import numpy as np

class train_loader(object):
    def __init__(self, train_list, train_path, musan_path, rir_path, num_frames, **kwargs):
        """
        初始化訓練資料載入器：
        - train_list: 語音檔路徑與說話者對應檔案
        - train_path: 語音檔所在的資料夾
        - musan_path: MUSAN 資料集路徑，用於加噪音
        - rir_path: RIR 資料夾路徑，用於模擬混響
        - num_frames: 每筆資料取樣幀數
        """
        self.train_path = train_path
        self.num_frames = num_frames

        # 定義噪音類型與對應 SNR 範圍與數量
        self.noisetypes = ['noise','speech','music']
        self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
        self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}

        # 建立噪音資料清單
        self.noiselist = {}
        augment_files = glob.glob(os.path.join(musan_path,'*/*/*.wav'))
        for file in augment_files:
            if file.split('\\')[-3] not in self.noiselist:
                self.noiselist[file.split('\\')[-3]] = []
            self.noiselist[file.split('\\')[-3]].append(file)

        # 讀取混響 RIR 檔案
        self.rir_files = glob.glob(os.path.join(rir_path,'*/*/*.wav'))

        # 載入語音檔與對應標籤
        self.data_list = []
        self.data_label = []
        lines = open(train_list).read().splitlines()[:10000]  # 限制讀取前 500000 行
        dictkeys = list(set([x.split()[0] for x in lines]))  # 擷取所有說話者 ID
        dictkeys.sort()
        dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
        for index, line in enumerate(lines):
            speaker_label = dictkeys[line.split()[0]]
            file_name = os.path.join(train_path, line.split()[1])
            self.data_label.append(speaker_label)
            self.data_list.append(file_name)

    def __getitem__(self, index):
        """
        取得資料：
        - 讀取語音檔
        - 擷取隨機片段
        - 套用數據增強（原始、混響、噪音等）
        - 回傳語音 Tensor 和標籤
        """
        
        # 因為抓下來的檔案是.m4a檔，因此造成soundfile不能直接讀取
        path = self.data_list[index]
        if path.lower().endswith('.m4a'):
            audio, sr = self.load_m4a(path)
        else:
            audio, sr = soundfile.read(path)

        length = self.num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random()*(audio.shape[0]-length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)

        # 隨機選擇數據增強方式
        augtype = random.randint(0, 5)
        if augtype == 0:   # 原始資料
            audio = audio
        elif augtype == 1: # 混響
            audio = self.add_rev(audio)
        elif augtype == 2: # 語音型噪音（多人講話）
            audio = self.add_noise(audio, 'speech')
        elif augtype == 3: # 音樂噪音
            audio = self.add_noise(audio, 'music')
        elif augtype == 4: # 背景噪音
            audio = self.add_noise(audio, 'noise')
        elif augtype == 5: # 混合噪音（電視情境）
            audio = self.add_noise(audio, 'speech')
            audio = self.add_noise(audio, 'music')

        return torch.FloatTensor(audio[0]), self.data_label[index]

    def __len__(self):
        """
        回傳資料筆數
        """
        return len(self.data_list)

    def add_rev(self, audio):
        """
        加入混響效果：
        - 從 RIR 檔案中選擇一個
        - 與原始語音做卷積模擬混響
        """
        rir_file = random.choice(self.rir_files)
        rir, sr = soundfile.read(rir_file)
        rir = numpy.expand_dims(rir.astype(float), 0)
        rir = rir / numpy.sqrt(numpy.sum(rir**2))  # 正規化
        return signal.convolve(audio, rir, mode='full')[:, :self.num_frames * 160 + 240]

    def add_noise(self, audio, noisecat):
        """
        加入背景噪音：
        - 根據類型選擇 SNR、數量
        - 從噪音資料集中隨機取出
        - 根據 SNR 調整音量後加入語音
        """
        
        # 計算乾淨語音的平均功率 (DB)
        clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4) 
        
        # 決定加入的噪音數量和選擇噪音檔案
        numnoise = self.numnoise[noisecat]
        noiselist = random.sample(self.noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
        
        
        noises = []
        for noise in noiselist:
            noiseaudio, sr = soundfile.read(noise)

            # 定義模型輸入長度
            # 每10ms一幀，一幀的樣本數 = sampling rate (16000Hz) * 0.01s = 160
            # 加上前後各240個樣本的緩衝區，避免邊緣效應(猜測)
            length = self.num_frames * 160 + 240
            
            # 如果噪音長度不足，則重複填充 (不一定每段噪音都有足夠長度可供使用)
            if noiseaudio.shape[0] <= length:
                shortage = length - noiseaudio.shape[0]
                noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
            # 隨機選擇噪音片段
            start_frame = numpy.int64(random.random()*(noiseaudio.shape[0]-length))
            noiseaudio = noiseaudio[start_frame:start_frame + length]
            noiseaudio = numpy.stack([noiseaudio], axis=0)
            noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
            
            # 確定目標的信噪比 (SNR)
            noisesnr = random.uniform(self.noisesnr[noisecat][0], self.noisesnr[noisecat][1])
            noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
        noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
        return noise + audio
    
    def load_m4a(self, path):
        audio = AudioSegment.from_file(path, format='m4a')
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.channels == 2:
            samples = samples.reshape(-1, 2).mean(axis=1)
        return samples / (1 << (8*audio.sample_width - 1)), audio.frame_rate