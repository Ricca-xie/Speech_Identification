import torch
# from vit_model import ViT
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import datetime
import time
from dataloader import SpeechDataset
from dataloader import create_loader
# from my_dataloader import data_loader
# from vit_model.pit import PiT
# from vit_model.levit import LeViT
from timm.utils import accuracy
from timm.models import create_model
import levit
import levit_c

device = torch.device('cuda:0')

def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--data_path', type=str, default='../speech_identify/code/preprocess/concate/spec-n/')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/levit384_spec_single_')
    parser.add_argument('--finetune', type=str, default='../pretrain_model/LeViT-384.pth')
    parser.add_argument('--eval', default=False)
    parser.add_argument('--model', type = str, default='LeViT_384_sin') 
    parser.add_argument('--distillation-type', default='none',
                        choices=['none', 'soft', 'hard'], type=str, help="")

    return parser.parse_args()


def valid(model):
    print('Validating...')
    test_data_dir = '../datasets/concat-I/spec-n/X_test_spec-n.myarray'
    test_label_dir = '../datasets/concat-I/spec-n/Y_test_spec-n.myarray'
    # dataset_dev = SpeechDataset(args.data_path, 'X_dev_test-n.myarray', 'Y_test_spec-n.myarray', (13755, 3, 1, 300, 161),
    #                             (13755, 1251))
    # dev_data = create_loader(dataset_dev,
    #                          batch_size=16,
    #                          is_training=False)
    # dev_data = data_loader('../datasets/concat-I/spec-n', 'X_test_spec-n.myarray', 'Y_test_spec-n.myarray', (8251, 3, 1, 300, 161), (8251, 1251), batch_size=16, is_shuffle=False)
    dataset_dev = SpeechDataset(data_path=test_data_dir, label_path=test_label_dir,data_length = 8251, filters = 161)
    dev_data = create_loader(dataset_dev, batch_size=args.batchsize, is_training=False)
    model.eval()
    loss_total = 0.
    acc1_total = 0.
    acc5_total = 0.
    with torch.no_grad():
        for step, (x, label) in enumerate(tqdm(dev_data)):
            x = x.to(device)
            x = x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
            x = x[:,0].unsqueeze(1)

            label = label.to(device)
            label = torch.argmax(label, dim=1)

            input = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)

            output = model(input)
            pred = nn.Softmax(dim=1)(output)
            # _, pred = torch.max(pred, dim=1)
            # accuracy = (torch.eq(pred, label).float()).mean().item()
            acc1, acc5 = accuracy(pred, label, topk=(1, 5)) #torch
            acc1, acc5 = acc1.item()/100, acc5.item()/100 #float
            loss = nn.CrossEntropyLoss()(output, label)

            # accuracy_total += accuracy
            loss_total += float(loss.item())
            acc1_total += acc1
            acc5_total += acc5

    print("Valid_loss: {}, Valid_acc1:{}, Valid_acc5: {}".format(loss_total / (step+1), acc1_total / (step+1), acc5_total / (step+1) ))
    return acc1_total / (step+1), acc5_total / (step+1)

# if __name__ == '__main__':
def train(args):
    # train_data = data_loader('../speech_identify/code/preprocess/concate/spec-n', 'X_train_spec-n.myarray',
    #                          'Y_train_spec-n.myarray',
    #                          (125995, 3, 1, 300, 161), (125995, 1251), batch_size=16, is_training=True)
    train_data_dir = '../datasets/concat-I/spec-n/X_train_spec-n.myarray'
    train_label_dir = '../datasets/concat-I/spec-n/Y_train_spec-n.myarray'
    # valid_data_dir = '../../speech_identify/code/preprocess/concate/spec-n/X_dev_spec-n.myarray'
    # valid_label_dir = '../../speech_identify/code/preprocess/concate/spec-n/Y_dev_spec-n.myarray'
    dataset_train = SpeechDataset(data_path=train_data_dir, label_path=train_label_dir,data_length = 145265, filters = 161 )
    # train_data = create_loader( dataset_train,batch_size = args.batchsize, is_training=True)
    # model = LeViT(
    #     image_size = 224,
    #     num_classes = 1251,
    #     stages = 3,             # number of stages
    #     dim = (256, 384, 512),  # dimensions at each stage
    #     depth = 4,              # transformer of depth 4 at each stage
    #     heads = (4, 6, 8),      # heads at each stage
    #     mlp_mult = 2,
    #     dropout = 0.1
    # )

    model = create_model(
        args.model,
        num_classes=1251,
        distillation=(args.distillation_type != 'none'),
        pretrained=False,
        fuse=False,
    )

    if args.finetune:
        # print('finetune')
        if args.finetune.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.finetune, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.finetune, map_location='cpu')

        checkpoint_model = checkpoint['model']
        state_dict = model.state_dict()
        # 'head_dist.l.weight', 'head_dist.l.bias'
        for k in ['head.weight', 'head.bias',
                  'head_dist.weight', 'head_dist.bias','head.l.weight', 'head.l.bias','patch_embed.0.c.weight']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                # print('k')
                # print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        model.load_state_dict(checkpoint_model, strict=False)


    model = model.to(device)
    # print(model)
    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 0.00005)
    best_epoch = -1
    best_acc1 = 0
    best_acc5 = 0
    acc = valid(model)
    # import pdb;pdb.set_trace()
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        # train_data = data_loader('../../speech_identify/code/preprocess/concate/mel', 'X_train_spec-n.myarray', 'Y_train_spec-n.myarray', (145265, 3, 1, 300, 161), (145265, 1251), batch_size=args.batchsize, type = 'mel' ,is_append = True, is_shuffle=True)
        train_data = create_loader(dataset_train, batch_size = args.batchsize, is_training=True)
        acc1_total = 0.
        acc5_total = 0.
        loss_total = 0.

        for step, (x, label) in enumerate(train_data):
            x = x.to(device)
            x = x.view(x.shape[0], x.shape[1], x.shape[3], x.shape[4])
            x = x[:,0].unsqueeze(1)

            label = label.to(device)
            label = torch.argmax(label, dim=1)
            input = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=True)

            model.zero_grad()
            optimizer.zero_grad()

            output = model(input)

            pred = nn.Softmax(dim=1)(output)
            # _, pred = torch.max(pred, dim =1)
            # accuracy = (torch.eq(pred, label).float()).mean().item()

            acc1, acc5 = accuracy(pred, label, topk=(1, 5)) #torch
            acc1, acc5 = acc1.item()/100, acc5.item()/100 #float

            loss = loss_fct(output, label)
            loss.backward()
            optimizer.step()

            loss_total += float(loss.item())
            acc1_total += acc1
            acc5_total += acc5

            if step % 100 == 0 and step != 0:
                print('epoch %d, step %d, step_loss %.4f, step_acc1 %.4f, step_acc5 %.4f' % (
                epoch, step, loss_total / args.print_every, acc1_total / args.print_every, acc5_total / args.print_every))
                loss_total = 0.
                acc1_total = 0.
                acc5_total = 0.

        if epoch % 10 == 0 and epoch != 0:
            torch.save(model.state_dict(), args.checkpoint_path+str(epoch)+'.pt')

        acc1, acc5 = valid(model)
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            best_epoch = epoch
        print('best acc1 is :{}, acc5 is :{}, in epoch {}'.format(best_acc1, best_acc5, best_epoch))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))





if __name__ == '__main__':
    args = parse_config()
    train(args)
