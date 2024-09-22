import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
import random
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score

def set_seed(seed=0):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    set_seed(200)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.FashionMNIST(root='../real-time data/data', train=True,
                                            download=True, transform=transform)

    # 设置每个类别需要的样本数
    #class_samples = {0: 5000, 1: 330, 2: 500, 3: 1500, 4: 450, 5: 850, 6: 350, 7: 450, 8: 1000, 9: 3000}
    class_samples = {0: 5000, 1: 5000, 2: 5000, 3: 5000, 4: 5000, 5: 5000, 6: 5000, 7: 5000, 8: 5000, 9: 5000}
    # 从原始数据集中筛选出指定数量的样本
    filtered_indices = []
    for idx, (img, label) in enumerate(trainset):
        if label in class_samples:
            if class_samples[label] > 0:
                filtered_indices.append(idx)
                class_samples[label] -= 1
        if sum(class_samples.values()) == 0:
            break
    # 创建新的数据集
    filtered_dataset = torch.utils.data.Subset(trainset, filtered_indices)
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.FashionMNIST(root='../real-time data/data', train=False,
                                           download=True, transform=transform)
    # 选择前三个类别的索引
    class_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 从原始测试集中筛选出前两个类别的数据
    filtered_test_data = [(img, label) for img, label in testset if label in class_indices]

    # 创建新的测试集
    filtered_test_dataset = torch.utils.data.Subset(testset,
                                                    [i for i in range(len(testset)) if testset[i][1] in class_indices])

    # 创建测试集的数据加载器
    testloader = torch.utils.data.DataLoader(filtered_test_dataset, batch_size=128,
                                             shuffle=False, num_workers=2)

    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot')

    correct_classified = {classname: 0 for classname in classes}
    total_class = {classname: 0 for classname in classes}
    count = 1
    result = []
    recalls = []


    start_lr = 0.001
    end_lr = 0.1
    num_steps = 5
    # 使用对数尺度进行均匀分割
    learning_rates = np.geomspace(start_lr, end_lr, num=num_steps)
    start_dropout = 0.01
    end_dropout = 0.25
    num_steps_d = 5
    Dout = np.geomspace(start_dropout, end_dropout, num=num_steps_d)

    for cs in [3, 5]:
        for lr in learning_rates:
            for d1 in Dout:
                for d2 in Dout:
                    print('第', count, '次训练开始---------------')
                    class Net(nn.Module):
                        def __init__(self):
                            super(Net, self).__init__()
                            self.conv1 = nn.Conv2d(3, 103, cs, stride=1, padding=1)
                            self.pool = nn.MaxPool2d(2, 2, padding=1)
                            self.norm1 = nn.BatchNorm2d(103)
                            self.conv2 = nn.Conv2d(103, 105, cs, stride=1, padding=1)
                            self.norm2 = nn.BatchNorm2d(105)
                            self.conv3 = nn.Conv2d(105, 107, cs, stride=1, padding=1)
                            self.conv4 = nn.Conv2d(107, 109, cs, stride=1, padding=1)
                            self.conv5 = nn.Conv2d(109, 179, cs, stride=1, padding=1)
                            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                            self.fc = nn.Linear(179, 10)
                            self.dropout1 = nn.Dropout(d1)
                            self.dropout2 = nn.Dropout(d2)
                        def forward(self, x):
                            x = self.pool(self.norm1(F.relu(self.conv1(x))))
                            x = self.dropout1(x)
                            x = self.pool(self.norm2(F.relu(self.conv2(x))))
                            x = self.dropout1(x)
                            x = F.relu(self.conv3(x))
                            x = F.relu(self.conv4(x))
                            x = F.relu(self.conv5(x))
                            x = self.global_pool(x)
                            x = x.view(-1, 179)
                            x = self.dropout2(x)
                            x = self.fc(x)
                            return x


                    net = Net().to(device)

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)#1231231231231231231231231241234234234234234

                    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

                    e = 1e-2
                    last_loss = 0
                    last_accuracy = 0
                    min_delta = 0.001
                    no_improve_epoch = 0  # 用于记录连续没有性能提升的epoch数。在某些早停策略中，如果连续多个epoch没有性能提升，就会停止训练。
                    patience = 10  # 表示在早停策略中允许的最大连续没有性能提升的epoch数。超过这个数值后，训练将会停止。
                    best_accuracy = 0
                    correct = 0
                    total = 0
                    ssum = 0
                    Acc_F1 = []
                    Acc_Pmain = []
                    ggg = []

                    perF1 = {}

                    perGm = {}
                    perPm = {}
                    perRecall = []
                    for epoch in range(200):
                        running_loss = 0.0
                        perF1l = []
                        for i, data in enumerate(trainloader, 0):
                            inputs, labels = data[0].to(device), data[1].to(device)
                            optimizer.zero_grad()
                            outputs = net(inputs)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            # running_loss += loss.item()
                            ssum += labels.size(0)
                        scheduler.step()
                        #print('训练的样本数', ssum)

                        num_classes = len(classes)
                        conf_matrix = np.zeros((num_classes, num_classes))
                        true_labels = []
                        predicted_labels = []
                        with torch.no_grad():
                            for data in testloader:
                                images, labels = data[0].to(device), data[1].to(device)
                                outputs = net(images)
                                _, predicted = torch.max(outputs.data, 1)
                                true_labels.extend(labels.cpu().numpy())
                                predicted_labels.extend(predicted.cpu().numpy())
                                total += labels.size(0)
                                correct += (predicted == labels).sum().item()

                        curr_accuracy = 100 * correct / total

                        conf_matrix = confusion_matrix(true_labels, predicted_labels)
                        sensitivities = []
                        specificities = []
                        for i in range(num_classes):
                            tp = conf_matrix[i, i]
                            fn = np.sum(conf_matrix[i, :]) - tp
                            fp = np.sum(conf_matrix[:, i]) - tp
                            tn = np.sum(conf_matrix) - tp - fn - fp
                            sensitivity = tp / (tp + fn)
                            specificity = tn / (tn + fp)
                            sensitivities.append(sensitivity)
                            specificities.append(specificity)

                        # 计算每个类别的 G-mean
                        class_g_means = [np.sqrt(s * p) for s, p in zip(sensitivities, specificities)]
                        print('class_g_means',class_g_means)
                        perGm[curr_accuracy] = class_g_means
                        # 计算整体的 G-mean
                        g_mean = np.mean(class_g_means)
                        ggg.append(g_mean)


                        # 计算每个类别的 Precision
                        class_precisions = precision_score(true_labels, predicted_labels, average=None)
                        print('class_precisions',class_precisions)
                        perPm[curr_accuracy] = class_precisions
                        # 计算 P-mean
                        p_mean = np.mean(class_precisions)


                        #计算F1
                        true_labels = np.array(true_labels)
                        predicted_labels = np.array(predicted_labels)
                        macro_f1 = f1_score(true_labels, predicted_labels, average='macro')
                        #print('21312312312',macro_f1)

                        Acc_F1.append({curr_accuracy:macro_f1})
                        Acc_Pmain.append({curr_accuracy:p_mean})
                        class_f1_scores = classification_report(true_labels, predicted_labels, target_names=classes, output_dict=True)
                        for classname, score_dict in class_f1_scores.items():
                            if classname in classes:  # 仅输出属于指定类别的结果
                                perF1l.append(score_dict["f1-score"])
                                perF1[curr_accuracy] = perF1l
                                print(f'F1-score for class: {classname:5s} is {score_dict["f1-score"]:.2f}')

                        #print('231231231231',perF1)
                        print(count,'--- 第', epoch, '个epoch的Accuracy : {:.2f} %'.format(curr_accuracy), "|G-mean:", g_mean, "|P-mean:", p_mean, '|cs:',cs, '|lr:',lr)
                        #print("G-mean:", g_mean)
                        #print("P-mean:", p_mean)
                        #print('第', epoch, '个epoch的正确数和总数', correct, '|', total)
                        if curr_accuracy - best_accuracy > min_delta:
                            best_accuracy = curr_accuracy
                            no_improve_epoch = 0

                        else:
                            no_improve_epoch += 1

                        if no_improve_epoch >= patience:
                            #print(f'Early stopping at epoch {epoch + 1}')
                            break

                        correct = 0
                        total = 0
                        ssum = 0


                    #print('第',count,'次训练结束')
                    m = max([list(d.keys())[0] for d in Acc_F1])
                    #print('精度最大值：', m)
                    #print(Acc_Pmain)
                    for item in Acc_Pmain:
                        if m in item:
                            ppp = item[m]
                            break
                    correct = 0
                    total = 0

                    with torch.no_grad():
                        for data in testloader:
                            images, labels = data[0].to(device), data[1].to(device)
                            outputs = net(images)
                            _, predicted = torch.max(outputs.data, 1)
                            total += labels.size(0)
                            correct += (predicted == labels).sum().item()

                            for label, prediction in zip(labels, predicted):
                                if label == prediction:
                                    correct_classified[classes[label]] += 1
                                total_class[classes[label]] += 1
                    #print(correct_classified, '正总', total_class)
                    print('Accuracy of the network on the 10000 test images: {:.2f} %'.format(m))
                    F1c = 0
                    for acc in Acc_F1:
                        if list(acc.keys())[0] == m:
                            F1c = acc[m]
                    print('精度最大值对应的F1分数：',F1c)
                    rrr = []
                    for classname, correct_count in correct_classified.items():
                        recall = 100 * float(correct_count) / total_class[classname]
                        rrr.append(recall)
                        print(f'Recall for class: {classname:5s} is {recall:.2f} %')
                        recalls.append({classname: recall})
                    recall_avg = 0
                    #quanzhong = [5000, 330, 500, 1500, 450, 850, 350, 450, 1000, 3000]
                    quanzhong = [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]
                    for i in range(10):
                        recall_avg = recall_avg + quanzhong[i]/50000*rrr[i]
                    print('recall_avg:', recall_avg)

                    del net
                    torch.cuda.empty_cache()
                    result.append({
                        'count': count,
                        'Acc': m,
                        'F1': F1c,
                        'Gm': max(ggg),
                        'Pm': ppp,
                        'Recall': recall_avg,
                        'lr': lr,
                        'd1': d1,
                        'd2': d2,
                        'cs': cs,
                        'perF1':perF1[m],
                        'perGm':perGm[m],
                        'perPm':perPm[m],
                        'perRecall':rrr
                    })
                    count = count + 1
    print(result)

    df = pd.DataFrame(result)

    # 将 DataFrame 存储到 Excel 文件
    df.to_excel('./data/fcfx_FM_250_1.xlsx', index=False)

    #with open('./data/fcfx2.json', 'w') as f:
     #   json.dump(result, f)