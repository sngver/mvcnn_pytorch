import torch
from torch import nn
from torch.autograd import Variable
import os
import argparse
import numpy as np
from tools.Trainer import ModelNetTrainer
from tools.ImgDataset import MultiviewImgDataset, SingleImgDataset
from models.MVCNN import MVCNN, SVCNN

parser = argparse.ArgumentParser()
parser.add_argument("-name", "--name", type=str, help="Name of the experiment", default="MVCNN")
parser.add_argument("-log_path", "--log_path", type=str, help="Log path of the experiment", default="")
parser.add_argument("-bs", "--batchSize", type=int, help="Batch size for the second stage", default=8)# it will be *12 images in each batch for mvcnn
parser.add_argument("-num_models", type=int, help="number of models per class", default=10)
parser.add_argument("-lr", type=float, help="learning rate", default=5e-5)
parser.add_argument("-weight_decay", type=float, help="weight decay", default=0.0)
parser.add_argument("-no_pretraining", dest='no_pretraining', action='store_true')
parser.add_argument("-cnn_name", "--cnn_name", type=str, help="cnn model name", default="vgg11")
parser.add_argument("-num_views", type=int, help="number of views", default=12)
parser.add_argument("-train_path", type=str, default="modelnet40_images_new_12x/*/train")
parser.add_argument("-val_path", type=str, default="modelnet40_images_new_12x/*/test")
parser.add_argument("-num_class", type=int, default=3)
parser.add_argument("-skip_stage1", action="store_true", default=False)
parser.set_defaults(train=False)
parser.add_argument("-epoch","--epoch",type=int,help="number of epochs",default=30)

args = parser.parse_args()
log_dir = os.path.join(args.log_path, 'mvcnn/mvcnn_stage_2/')
savepath = os.path.join(log_dir, 'best_model.pth')



pretraining = not args.no_pretraining

cnet = SVCNN(args.name, nclasses=args.num_class, pretraining=pretraining, cnn_name=args.cnn_name)
cnet_2 = MVCNN(args.name, cnet, nclasses=args.num_class, cnn_name=args.cnn_name, num_views=args.num_views)
del cnet

val_dataset = MultiviewImgDataset(args.val_path, scale_aug=False, rot_aug=False, num_views=args.num_views)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batchSize, shuffle=False, num_workers=0)




def update_validation_accuracy(model, val_loader, log_dir, num_class=3):
    all_correct_points = 0
    all_points = 0

    DUMP_DIR = os.path.join(log_dir, 'dump')
    if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)

    # in_data = None
    # out_data = None
    # target = None

    wrong_class = np.zeros(num_class)
    samples_class = np.zeros(num_class)
    all_loss = 0

    model.cuda()

    checkpoint = torch.load(savepath)
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Epoch: ', start_epoch)
    print('Best Accuracy: ', best_acc)

    epoch = start_epoch

    model.eval()

    model_name = 'mvcnn'
    loss_fn = nn.CrossEntropyLoss()

    avgpool = nn.AvgPool1d(1, 1)

    total_time = 0.0
    total_print_time = 0.0
    all_target = []
    all_pred = []

    #
    fout = open(os.path.join(DUMP_DIR, 'pred_label.txt'), 'w')
    #

    for _, data in enumerate(val_loader, 0):

        if model_name == 'mvcnn':
            N, V, C, H, W = data[1].size()
            in_data = Variable(data[1]).view(-1, C, H, W).cuda()
        else:  # 'svcnn'
            in_data = Variable(data[1]).cuda()
        target = Variable(data[0]).cuda()

        out_data = model(in_data)
        pred = torch.max(out_data, 1)[1]
        all_loss += loss_fn(out_data, target).cpu().data.numpy()
        results = pred == target

        #
        for i in range(list(pred.shape)[0]):
            fout.write('%d, %d\n' % (pred[i], target[i]))
        #

        for i in range(results.size()[0]):
            if not bool(results[i].cpu().data.numpy()):
                wrong_class[target.cpu().data.numpy().astype('int')[i]] += 1
            samples_class[target.cpu().data.numpy().astype('int')[i]] += 1
        correct_points = torch.sum(results.long())

        all_correct_points += correct_points
        all_points += results.size()[0]

    #
    fout.close()
    #

    print('Total # of test models: ', all_points)
    val_mean_class_acc = np.mean((samples_class - wrong_class) / samples_class)
    acc = all_correct_points.float() / all_points
    val_overall_acc = acc.cpu().data.numpy()
    loss = all_loss / len(val_loader)

    print('val mean class acc. : ', val_mean_class_acc)
    print('val overall acc. : ', val_overall_acc)
    print('val loss : ', loss)

with torch.no_grad():
    update_validation_accuracy(model=cnet_2, val_loader=val_loader, log_dir=log_dir, num_class=args.num_class)

