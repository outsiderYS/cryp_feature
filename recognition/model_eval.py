from feeder import Feeder
from torchvision.models.densenet import DenseNet
import torch.optim as optim
from metrics import *
from utils import *
from sklearn.metrics import recall_score, precision_score
import os


labels = [1, 2, 3, 4]


def list_get(y_pred, y_test):
    _, pred = y_pred.topk(1, 1, True, True)
    pred = pred.t()
    target = y_test.view(1, -1).expand_as(pred)
    pred = pred.cpu().numpy()[0]
    target = target.cpu().numpy()[0]
    return pred, target


def evaluate(model, test_dl):
    summ = []
    pred_sum = []
    tar_sum = []
    model.eval()

    for i, (data_batch, label_batch) in enumerate(test_dl):
        data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

        data_batch, label_batch = Variable(data_batch), Variable(label_batch)

        logits = model(data_batch)

        y_xent = F.cross_entropy(logits, label_batch)

        summary_batch = {metric:metrics[metric](logits.data, label_batch.data)
                                 for metric in metrics}

        pre_list, tar_list = list_get(logits.data, label_batch.data)
        pred_sum.extend(pre_list)
        tar_sum.extend(tar_list)

        # summary_batch['KLD'] = KLD.sum().data.item()
        summary_batch['entropy']  = y_xent.sum().data.item(),

        summ.append(summary_batch)

    mean_metrics = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    precision = precision_score(tar_sum, pred_sum, labels=labels, average='micro')
    recall = recall_score(tar_sum, pred_sum, labels=labels, average='micro')
    accuracytop1 = mean_metrics["accuracytop1"]
    return "\naccuracy: {}\nprecision: {}\nrecall:{}\n".format(accuracytop1,  precision, recall)


if __name__ == '__main__':
    ratio = 0.01
    resolution = 224
    train_folder_path = '../stft/freq_ten_entropy_png/{}/{}'.format(ratio, resolution)
    test_folder_path = '../stft/testdl/{}/{}'.format(ratio, resolution)
    num_classes = 4
    load_dir = 'model/{}/{}'.format(ratio, resolution)

    model = DenseNet(growth_rate=32, block_config=(6, 12, 24, 16),
                     num_init_features=64, bn_size=4, drop_rate=0.3, num_classes=num_classes)

    if torch.cuda.device_count() > 1:  # 检查电脑是否有多块GPU
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model.cuda())  # 将模型对象转变为多GPU并行运算的模型
    else:
        model = model.cuda()
        print('cuda_weights')

    train_dl = torch.utils.data.DataLoader(dataset=Feeder(train_folder_path),
                                           batch_size=16,
                                           shuffle=True,
                                           num_workers=1,
                                           drop_last=True)
    test_dl = torch.utils.data.DataLoader(dataset=Feeder(test_folder_path, is_training=False),
                                          batch_size=16,
                                          shuffle=False,
                                          num_workers=1,
                                          drop_last=True)

    load_checkpoint(load_dir + '/best.pth.tar', model)

    mean_metrics = evaluate(model, train_dl)
    eval_result = "train_dataset: " + mean_metrics
    mean_metrics = evaluate(model, test_dl)
    eval_result = eval_result + "test_dataset: " + mean_metrics

    eval_save_path = "./eval_results/{}_{}.txt".format(ratio, resolution)

    with open(eval_save_path, 'w') as f:
        f.write(eval_result)




