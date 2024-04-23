import torch
from torch import nn,optim
from Vgg16 import VGGNet16
from vgg16dataset import ImgDataset

# Check cuda availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():

    # 初始化模型
    model = VGGNet16().to(device)
    model.load_state_dict(torch.load('log/Mutiview/Vgg16_modelnet10.pth'))
    # model = ResNet50().to(device)
    # 构造损失函数和优化器
    crterion = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.01, momentum=0.8, weight_decay=0.001)

    # 动态更新学习率------每隔step_size : lr = lr * gamma
    #scheduler = optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.6, last_epoch=-1)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.96)
    trainpath = "data/modelnet40_images_12views/*/train"
    testpath = "data/modelnet40_images_12views/*/test"

    train_dataset = ImgDataset(trainpath)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)

    val_dataset = ImgDataset(testpath)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)

    print('num_train_files: ' + str(len(train_dataset.filepaths)))
    print('num_val_files: ' + str(len(val_dataset.filepaths)))

    # Train
    # EPOCHES = 30
    # model.train()  # model.train()是保证BN层用每一批数据的均值和方差
    # for ep in range(EPOCHES):
    #     batch_id = 1
    #     correct, total, total_loss = 0, 0, 0.
    #     for inputs, labels in train_loader:
    #         opt.zero_grad()
    #         inputs = inputs.to(device)
    #         labels = labels.to(device)
    #         outputs = model(inputs)  # logits 全连接层的输出 softmax的输入
    #         # Compute loss & accuracy
    #         loss = criterion(outputs, labels).to(device)
    #         pred = outputs.argmax(dim=1)  # 返回每一行中最大值元素索引
    #
    #
    #         correct += torch.eq(pred, labels).sum().item()
    #         # dim=1 返回一行中最大值的序号 判断输出的类别是否和labels相等
    #
    #         total += len(labels)
    #         accuracy = correct / total
    #         total_loss += loss
    #         loss.backward()  # compute gradient (backpropagation)
    #         opt.step()  # update model with optimizer
    #         print('Epoch {}, batch {}, train_loss: {}, train_accuracy: {}'.format(ep + 1,
    #                                                                               batch_id,
    #                                                                               total_loss / batch_id,
    #                                                                               accuracy))
    #         batch_id += 1
    #     scheduler.step(ep)
    #     print('Total loss for epoch {}: {}'.format(ep + 1, total_loss))
    #     # Save model
    # torch.save(model.state_dict(), 'Vgg16_modelnet10.pth'.format(EPOCHES, correct / total))

    # Eval
    model.eval()  # ，model.eval()是保证BN用全部训练数据的均值和方差
    correct, total = 0, 0
    for inputs, labels in val_loader:
        with torch.no_grad():
        # Add channels = 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs= model(inputs)
        pred = torch.argmax(outputs, dim=1)  # argmax函数 要求每一行最大的列标号，我们就要指定dim=1
        correct += torch.eq(pred, labels).sum().item()
        total += len(labels)
    print('Accuracy: {}'.format(correct / total))




if __name__ == '__main__':
    main()
