'''
Some utilized functions
These functions are all copied from voxceleb_trainer: https://github.com/clovaai/voxceleb_trainer/blob/master/tuneThreshold.py
'''

import os, numpy, torch
from sklearn import metrics
from operator import itemgetter
import torch.nn.functional as F

def init_args(args):
	args.score_save_path    = os.path.join(args.save_path, 'score.txt')
	args.model_save_path    = os.path.join(args.save_path, 'model')
	os.makedirs(args.model_save_path, exist_ok = True)
	return args

def tuneThresholdfromScore(scores, labels, target_fa, target_fr=None):
    """
    根據模型的預測分數和真實標籤，調整並找出最佳的分類閾值，
    以達到特定的假陽性率 (FAR) 或假陰性率 (FNR) 目標，並計算等錯誤率 (EER)。

    Args:
        scores: 模型對每個樣本的預測分數（或機率）。
                通常是浮點數陣列，分數越高表示模型越確信其為正類。
        labels: 每個樣本的真實類別標籤。
                0 或 1 的整數陣列，1 通常代表正類，0 代表負類。
        target_fa: 一個或多個目標假陽性率 (False Acceptance Rate, FAR) 值。
                    函數會找出最接近這些目標 FAR 的閾值。
        target_fr: 一個或多個目標假陰性率 (False Rejection Rate, FNR) 值。
                    函數會找出最接近這些目標 FNR 的閾值。
                    如果為 None，則不針對 FNR 進行閾值調整。

    Returns:
        - tunedThreshold: 一個列表，每個元素是一個子列表 [threshold, fpr, fnr]。
                            其中包含針對 'target_fa' 和 'target_fr' 找到的最佳閾值，
                            以及在該閾值下對應的假陽性率 (FPR) 和假陰性率 (FNR)。
                            (FPR = FAR)
        - eer: 等錯誤率 (Equal Error Rate)。當 FPR 和 FNR 相等時的錯誤率，以百分比表示 (0-100)。
        - fpr: 所有計算出的假陽性率 (FPR) 陣列，對應於不同的閾值。可用於繪製 ROC 曲線。
        - fnr: 所有計算出的假陰性率 (FNR) 陣列，對應於不同的閾值。
    """
    
    # 計算 ROC 曲線的數據：假陽性率 (FPR), 真陽性率 (TPR) 和對應的閾值
    # pos_label=1 表示將標籤為 1 的視為正類
    fpr, tpr, thresholds = metrics.roc_curve(labels, scores, pos_label=1)
    
    # 計算假陰性率 (FNR)，FNR = 1 - TPR
    fnr = 1 - tpr
    
    tunedThreshold = [] # 用於儲存調整後的閾值及其對應的 FPR 和 FNR
    
    # 如果提供了目標假陰性率 (target_fr)
    if target_fr is not None:
        for tfr in target_fr:
            # 找到 FNR 陣列中，與目標 tfr 絕對差值最小的索引
            idx = numpy.nanargmin(numpy.absolute((tfr - fnr)))
            # 將找到的閾值、FPR 和 FNR 儲存起來
            tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
            
    for tfa in target_fa:
        # 找到 FPR 陣列中，與目標 tfa 絕對差值最小的索引
        idx = numpy.nanargmin(numpy.absolute((tfa - fpr))) 
        # 將找到的閾值、FPR 和 FNR 儲存起來
        tunedThreshold.append([thresholds[idx], fpr[idx], fnr[idx]])
        
    # 計算等錯誤率 (EER)
    idxE = numpy.nanargmin(numpy.absolute((fnr - fpr))) # 找到 FPR 和 FNR 絕對差值最小的索引
    eer  = max(fpr[idxE], fnr[idxE]) * 100 # EER 是該點的 FPR 和 FNR 中較大的一個，轉換為百分比
    
    return tunedThreshold, eer, fpr, fnr


def ComputeErrorRates(scores, labels):
    """
    根據預測分數 (scores) 和實際標籤 (labels)，
    計算 False Negative Rates (FNRs)、False Positive Rates (FPRs)，
    以及相對應的判斷閾值 (thresholds)。
    
    Args:
        scores (list of float): 模型輸出的分數。
        labels (list of int): 實際標籤，1 表示正樣本，0 表示負樣本。

    Returns:
        fnrs (list of float): 各閾值下的 False Negative Rate。
        fprs (list of float): 各閾值下的 False Positive Rate。
        thresholds (list of float): 與錯誤率對應的閾值。
    """

    # 先將所有分數與索引配對並依分數排序，取出排序後的索引與對應閾值。
    # thresholds 是排序後的分數列表，sorted_indexes 是排序後的索引順序。
    sorted_indexes, thresholds = zip(*sorted(
        [(index, threshold) for index, threshold in enumerate(scores)],
        key=itemgetter(1)))  # 根據分數 (threshold) 升序排列

    labels = [labels[i] for i in sorted_indexes]

    fnrs = []  # False Negative Rates
    fprs = []  # False Positive Rates

    """
    遍歷每個排序後的樣本，逐步累積 FN 和 FP。
    若當前樣本是正樣本 (label=1)，表示若閾值設在這邊會錯過一個正樣本 (FN)。
    若是負樣本 (label=0)，表示會誤判為正樣本 (FP)。
    """
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i-1] + labels[i])
            fprs.append(fprs[i-1] + 1 - labels[i])

    """
    正樣本數 = 用於計算 FN 的總數
    負樣本數 = 用於計算 FP 的總數
    """
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    """
    將 FN 累積數正規化為比率，得到每個閾值下的 FNR
    """
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    """
    同理，先計算 True Positive Rate (TPR) = TP / 所有負樣本數
    再用 1 - TPR 得到 FPR
    """
    fprs = [1 - x / float(fprs_norm) for x in fprs]

    return fnrs, fprs, thresholds


def ComputeMinDcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    計算最小化的偵測成本函數（minDCF），並回傳對應的閾值。

    參數：
        - fnrs: 各閾值下的 False Negative Rates (FNR) 列表。
        - fprs: 各閾值下的 False Positive Rates (FPR) 列表。
        - thresholds: 對應於 fnrs 與 fprs 的決策閾值列表。
        - p_target: 目標事件（positive class）的先驗機率 P_target。
        - c_miss: 錯過（miss）一個正樣本的成本，對應於 C_miss。
        - c_fa: 把負樣本誤判為正樣本（false alarm）的成本，對應於 C_fa。

    回傳值：
        - min_dcf: 正規化後的最小偵測成本（minDCF）。
        - min_c_det_threshold: 使得偵測成本最小化的最佳閾值。
    """

    # 初始化：設定目前最小成本為無限大，並預設最佳閾值為列表中的第一個
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]

    # 遍歷所有閾值位置，計算對應的 detection cost C_det
    for i in range(len(fnrs)):
        # 加權求和：C_det = C_miss * FNR * P_target + C_fa * FPR * (1 - P_target)
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)

        # 若當前成本更低，則更新最小成本及對應閾值
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]

    # 計算預設成本 C_def，用以正規化
    # C_def = min(C_miss * P_target, C_fa * (1 - P_target))
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))

    # 正規化後的最小偵測成本
    min_dcf = min_c_det / c_def

    return min_dcf, min_c_det_threshold

def accuracy(output, target, topk=(1,)):

	maxk = max(topk)
	batch_size = target.size(0)
	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))
	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
		res.append(correct_k.mul_(100.0 / batch_size))
	
	return res