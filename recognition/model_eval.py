from feeder import Feeder
from torchvision.models.densenet import DenseNet
import torch.optim as optim
from metrics import *
from utils import *


def evaluate(model, test_dl):
    summ = []
    model.eval()

    for i, (data_batch, label_batch) in enumerate(test_dl):
        data_batch, label_batch = data_batch.cuda(), label_batch.cuda()

        data_batch, label_batch = Variable(data_batch), Variable(label_batch)

        logits = model(data_batch)

        y_xent = F.cross_entropy(logits, label_batch)

        summary_batch = {metric:metrics[metric](logits.data, label_batch.data)
                                 for metric in metrics}


        # summary_batch['KLD'] = KLD.sum().data.item()
        summary_batch['entropy']  = y_xent.sum().data.item(),

        summ.append(summary_batch)

    mean_metrics = {metric:np.mean([x[metric] for x in summ]) for metric in summ[0]}
    print(mean_metrics)
    return mean_metrics


if __name__ == '__main__':
    train_folder_path = '../stft/freq_ten_entropy_png'
    test_folder_path = '../stft/testdl'
    num_classes = 4
    load_dir = 'model'

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
                                          drop_last=False)

    load_checkpoint(load_dir + '/best.pth.tar', model)

    mean_metrics = evaluate(model, train_dl)
    mean_metrics = evaluate(model, test_dl)

