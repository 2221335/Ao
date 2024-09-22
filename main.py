# 对于cifar10先跑通一个随机参数分布的CNN,产生loss，accuracy，以及十类样本的recall
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
if __name__ == '__main__':


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../real-time data/data', train=True,
                                            download=True, transform=transform)

    # 设置每个类别需要的样本数
    #class_samples = {0: 5000, 1: 500}
    #class_samples = {0: 5000, 1: 1000, 2: 500}
    class_samples = {0: 4000, 1: 280, 2: 500, 3: 1500, 4: 350, 5: 800, 6: 2000, 7: 400, 8: 1000, 9: 3000}
    # 从原始数据集中筛选出指定数量的样本
    filtered_indices = []
    for idx, (img, label) in enumerate(trainset):
        if label in class_samples:
            if class_samples[label] > 0:
                #if label == 1:
                 #   for i in range(4):
                  #      filtered_indices.append(idx)
                #if label == 2:
                 #   for i in range(9):
                  #      filtered_indices.append(idx)
                filtered_indices.append(idx)
                class_samples[label] -= 1
        if sum(class_samples.values()) == 0:
            break
    #print(filtered_indices,'1231321123')
    #print(len(filtered_indices))
    # 创建新的数据集
    filtered_dataset = torch.utils.data.Subset(trainset, filtered_indices)

    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../real-time data/data', train=False,
                                           download=True, transform=transform)
    # 选择前三个类别的索引
    #class_indices = [0, 1]
    #class_indices = [0, 1, 2]
    class_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 从原始测试集中筛选出前两个类别的数据
    filtered_test_data = [(img, label) for img, label in testset if label in class_indices]

    # 创建新的测试集
    filtered_test_dataset = torch.utils.data.Subset(testset,
                                                    [i for i in range(len(testset)) if testset[i][1] in class_indices])

    # 创建测试集的数据加载器
    testloader = torch.utils.data.DataLoader(filtered_test_dataset, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship')
    #classes = ('plane', 'car', 'bird')
    #classes = ('plane', 'car')
    correct_classified = {classname: 0 for classname in classes}
    total_class = {classname: 0 for classname in classes}


    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(3, 2, padding=1)
            self.norm1 = nn.BatchNorm2d(32)
            self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
            self.norm2 = nn.BatchNorm2d(64)
            self.conv3 = nn.Conv2d(64, 96, 3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(96, 128, 3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            x = self.pool(self.norm1(F.relu(self.conv1(x))))
            x = self.pool(self.norm2(F.relu(self.conv2(x))))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = self.global_pool(x)
            x = x.view(-1, 256)
            x = self.fc(x)
            return x


    net = Net().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)

    e = 1e-2
    last_loss = 0
    last_accuracy = 0
    min_delta = 0.001
    no_improve_epoch = 0#用于记录连续没有性能提升的epoch数。在某些早停策略中，如果连续多个epoch没有性能提升，就会停止训练。
    patience = 10#表示在早停策略中允许的最大连续没有性能提升的epoch数。超过这个数值后，训练将会停止。
    best_accuracy = 0
    correct = 0
    total = 0
    ssum = 0
    Acc=[]
    for epoch in range(100):
        running_loss = 0.0
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
        print('训练的样本数',ssum)
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        curr_accuracy = 100 * correct / total

        Acc.append(curr_accuracy)
        print('第',epoch,'个epoch的Accuracy : {:.2f} %'.format(curr_accuracy))
        print('第',epoch,'个epoch的正确数和总数',correct,'|',total)
        if curr_accuracy - best_accuracy > min_delta:
            best_accuracy = curr_accuracy
            no_improve_epoch = 0

        else:
            no_improve_epoch += 1

        if no_improve_epoch >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

        correct = 0
        total = 0
        ssum = 0
        # curr_loss = running_loss/len(trainloader)
        # print(f'Epoch {epoch+1}, Loss: {running_loss/len(trainloader)}')

        # if np.abs(last_loss - curr_loss) / curr_loss < e:
        # break

    print('Finished Training')
    print('精度最大值：',max(Acc))
    correct = 0
    total = 0
    yhx = True
    if yhx:
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
    print(correct_classified,'正总',total_class)
    print('Accuracy of the network on the 10000 test images: {:.2f} %'.format(max(Acc)))
    for classname, correct_count in correct_classified.items():
        recall = 100 * float(correct_count) / total_class[classname]
        print(f'Recall for class: {classname:5s} is {recall:.2f} %')

    # 筛选出一些能量值最大的解minimize
    from scipy.optimize import minimize
    import numpy as np

    learned_param = np.zeros((4, 5))
    energy_learned = np.zeros(4)


    # 1
    def objective(x):
        n1, n2, n3, n4, n5 = x
        return -np.log(n1) * np.log(n2) * np.log(n3) * np.log(n4) * np.log(n5)


    def eq_constraint(x):
        n1, n2, n3, n4, n5 = x
        return 3 * n1 + n1 * n2 + n2 * n3 + n3 * n4 + n4 * n5 - 53424


    def ineq_constraint1(x):
        return x[0] - 3 - 1


    def ineq_constraint2(x):
        return x[1] - x[0] - 1


    def ineq_constraint3(x):
        return x[2] - x[1] - 1


    def ineq_constraint4(x):
        return x[3] - x[2] - 1


    def ineq_constraint5(x):
        return x[4] - x[3] - 1


    cons = (
        {'type': 'eq', 'fun': eq_constraint},
        {'type': 'ineq', 'fun': ineq_constraint1},
        {'type': 'ineq', 'fun': ineq_constraint2},
        {'type': 'ineq', 'fun': ineq_constraint3},
        {'type': 'ineq', 'fun': ineq_constraint4},
        {'type': 'ineq', 'fun': ineq_constraint5}
    )

    x0 = np.array([10, 10, 10, 10, 10])

    bnds = ((1, None), (1, None), (1, None), (1, None), (1, None))

    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

    print("learned params distribution: ", np.round(sol.x))
    n = sol.x
    energy = np.log(round(n[0])) * np.log(round(n[1])) * np.log(round(n[2])) * np.log(round(n[3])) * np.log(round(n[4]))
    print("Energy value: ", energy)
    learned_param[0, :] = np.round(sol.x)
    energy_learned[0] = energy


    # 2
    def ineq_constraint1(x):
        return x[0] - 3 - 2


    def ineq_constraint2(x):
        return x[1] - x[0] - 2


    def ineq_constraint3(x):
        return x[2] - x[1] - 2


    def ineq_constraint4(x):
        return x[3] - x[2] - 2


    def ineq_constraint5(x):
        return x[4] - x[3] - 2


    cons = (
        {'type': 'eq', 'fun': eq_constraint},
        {'type': 'ineq', 'fun': ineq_constraint1},
        {'type': 'ineq', 'fun': ineq_constraint2},
        {'type': 'ineq', 'fun': ineq_constraint3},
        {'type': 'ineq', 'fun': ineq_constraint4},
        {'type': 'ineq', 'fun': ineq_constraint5}
    )

    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

    print("learned params distribution: ", np.round(sol.x))
    n = sol.x
    energy = np.log(round(n[0])) * np.log(round(n[1])) * np.log(round(n[2])) * np.log(round(n[3])) * np.log(round(n[4]))
    print("Energy value: ", energy)
    learned_param[1, :] = np.round(sol.x)
    energy_learned[1] = energy


    # 3
    def ineq_constraint1(x):
        return x[0] - 3 - 3


    def ineq_constraint2(x):
        return x[1] - x[0] - 3


    def ineq_constraint3(x):
        return x[2] - x[1] - 3


    def ineq_constraint4(x):
        return x[3] - x[2] - 3


    def ineq_constraint5(x):
        return x[4] - x[3] - 3


    cons = (
        {'type': 'eq', 'fun': eq_constraint},
        {'type': 'ineq', 'fun': ineq_constraint1},
        {'type': 'ineq', 'fun': ineq_constraint2},
        {'type': 'ineq', 'fun': ineq_constraint3},
        {'type': 'ineq', 'fun': ineq_constraint4},
        {'type': 'ineq', 'fun': ineq_constraint5}
    )

    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

    print("learned params distribution: ", np.round(sol.x))
    n = sol.x
    energy = np.log(round(n[0])) * np.log(round(n[1])) * np.log(round(n[2])) * np.log(round(n[3])) * np.log(round(n[4]))
    print("Energy value: ", energy)
    learned_param[2, :] = np.round(sol.x)
    energy_learned[2] = energy


    # 4
    def ineq_constraint1(x):
        return x[0] - 3 - 2


    def ineq_constraint2(x):
        return x[1] - x[0] - 1


    def ineq_constraint3(x):
        return x[2] - x[1] - 3


    def ineq_constraint4(x):
        return x[3] - x[2] - 2


    def ineq_constraint5(x):
        return x[4] - x[3] - 3


    cons = (
        {'type': 'eq', 'fun': eq_constraint},
        {'type': 'ineq', 'fun': ineq_constraint1},
        {'type': 'ineq', 'fun': ineq_constraint2},
        {'type': 'ineq', 'fun': ineq_constraint3},
        {'type': 'ineq', 'fun': ineq_constraint4},
        {'type': 'ineq', 'fun': ineq_constraint5}
    )

    sol = minimize(objective, x0, method='SLSQP', bounds=bnds, constraints=cons)

    print("learned params distribution: ", np.round(sol.x))
    n = sol.x
    energy = np.log(round(n[0])) * np.log(round(n[1])) * np.log(round(n[2])) * np.log(round(n[3])) * np.log(round(n[4]))
    print("Energy value: ", energy)
    learned_param[3, :] = np.round(sol.x)
    energy_learned[3] = energy
    learned_param = learned_param.astype(int)
    print(learned_param)

    # 随机参数分布
    import pandas as pd
    import numpy as np

    filename = './random_param.xlsx'

    df = pd.read_excel(filename, usecols=[0], header=None, engine='openpyxl')

    random_param = np.zeros((49, 5), dtype=int)
    energy_random = np.zeros(49)

    for i, row in enumerate(df[0]):
        row_data = list(map(int, row.split()))
        random_param[i] = row_data

    for i in range(49):
        n = random_param[i, :]
        energy_random[i] = np.log(n[0]) * np.log(n[1]) * np.log(n[2]) * np.log(n[3]) * np.log(n[4])
        #energy_random[i] = 0.0531564 * n[0] + 0.1104975 * n[1] + 0.13592092 * n[2] + 0.26522045 * n[3] + 0.08426947 * n[4]
    print(energy_random)

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torchvision
    import torchvision.transforms as transforms
    import torch.nn.functional as F
    import numpy as np
    import matplotlib.pyplot as plt

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # 把param和energy拼起来(前四行是learned)
    params = np.vstack((learned_param, random_param))
    energys = np.concatenate((energy_learned, energy_random))
    print(params)
    print('=========================================')
    #energys[0] = energys[0] - 2370
    #energys[1] = energys[1] - 2370
    #energys[2] = energys[2] - 2370
    #energys[3] = energys[3] - 2370
    print(energys)

    # params的整体accuracy
    Accuracy = np.zeros(53)
    # params的10类样本召回率
    Recall = np.zeros((10, 53))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='../real-time data/data', train=True,
                                            download=True, transform=transform)

    # 设置每个类别需要的样本数
    #class_samples = {0: 5000, 1: 1000, 2: 500}
    #class_samples = {0: 5000, 1: 500}
    class_samples = {0: 4000, 1: 280, 2: 500, 3: 1500, 4: 350, 5: 800, 6: 2000, 7: 400, 8: 1000, 9: 3000}
    # 从原始数据集中筛选出指定数量的样本
    filtered_indices = []
    for idx, (img, label) in enumerate(trainset):
        if label in class_samples:
            if class_samples[label] > 0:
                #if label == 1:
                 #   for i in range(4):
                  #      filtered_indices.append(idx)
                #if label == 2:
                 #   for i in range(9):
                  #      filtered_indices.append(idx)
                filtered_indices.append(idx)
                class_samples[label] -= 1
        if sum(class_samples.values()) == 0:
            break

    # 创建新的数据集
    filtered_dataset = torch.utils.data.Subset(trainset, filtered_indices)

    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(filtered_dataset, batch_size=128,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='../real-time data/data', train=False,
                                           download=True, transform=transform)
    # 选择前三个类别的索引
    #class_indices = [0, 1, 2]
    #class_indices = [0, 1]
    class_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 从原始测试集中筛选出前两个类别的数据
    filtered_test_data = [(img, label) for img, label in testset if label in class_indices]

    # 创建新的测试集
    filtered_test_dataset = torch.utils.data.Subset(testset,
                                                    [i for i in range(len(testset)) if testset[i][1] in class_indices])

    # 创建测试集的数据加载器
    testloader = torch.utils.data.DataLoader(filtered_test_dataset, batch_size=128,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    #classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship')
    #classes = ('plane', 'car', 'bird')
    #classes = ('plane', 'car')
    # CNN模型定义
    class Net(nn.Module):
        def __init__(self, conv_channels):
            super(Net, self).__init__()
            ch1, ch2, ch3, ch4, ch5 = conv_channels
            self.conv1 = nn.Conv2d(3, ch1, 3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(3, 2, padding=1)
            self.norm1 = nn.BatchNorm2d(ch1)
            self.conv2 = nn.Conv2d(ch1, ch2, 3, stride=1, padding=1)
            self.norm2 = nn.BatchNorm2d(ch2)
            self.conv3 = nn.Conv2d(ch2, ch3, 3, stride=1, padding=1)
            self.conv4 = nn.Conv2d(ch3, ch4, 3, stride=1, padding=1)
            self.conv5 = nn.Conv2d(ch4, ch5, 3, stride=1, padding=1)
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(ch5, 10)

            self.ch5_ = conv_channels[4]

        def forward(self, x):
            x = self.pool(self.norm1(F.relu(self.conv1(x))))
            x = self.pool(self.norm2(F.relu(self.conv2(x))))
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv5(x))
            x = self.global_pool(x)
            x = x.view(-1, self.ch5_)
            x = self.fc(x)
            return x


    criterion = nn.CrossEntropyLoss()

    min_delta = 0.001
    patience = 5

    # 训练random params distribution的CNN
    for i in range(53):
        correct_classified = {classname: 0 for classname in classes}
        total_class = {classname: 0 for classname in classes}

        no_improve_epoch = 0
        best_accuracy = 0
        correct = 0
        total = 0

        net = Net(params[i, :])
        net.to(device)
        optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9)

        # 学习率调度器
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 100, 150], gamma=0.1)
        acc=[]
        for epoch in range(100):
            running_loss = 0.0
            for j, data in enumerate(trainloader, 0):
                inputs, labels = data[0].to(device), data[1].to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # running_loss += loss.item()
            scheduler.step()

            with torch.no_grad():
                for data in testloader:
                    images, labels = data[0].to(device), data[1].to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            curr_accuracy = 100 * correct / total
            acc.append(curr_accuracy)
            # print('Accuracy : {:.2f} %'.format(curr_accuracy))

            if curr_accuracy - best_accuracy > min_delta:
                best_accuracy = curr_accuracy
                no_improve_epoch = 0

            else:
                no_improve_epoch += 1

            if no_improve_epoch >= patience:
                # print(f'Early stopping at epoch {epoch+1}')
                break

            correct = 0
            total = 0

        correct = 0
        total = 0
        if yhx:
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

        #accuracy = 100 * correct / total
        print('Accuracy of the network {}: {:.2f} %'.format(i, max(acc)))
        Accuracy[i] = max(acc)
        j = 0
        for classname, correct_count in correct_classified.items():
            recall = 100 * float(correct_count) / total_class[classname]
            Recall[j, i] = recall
            j = j + 1
    print('召回率：',Recall)
    print('精度：',Accuracy)
    # 画散点图
    plt.scatter(energys[:4], Accuracy[:4], color='red')
    plt.scatter(energys[4:], Accuracy[4:], color='blue')
    plt.xlabel('Energy Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.savefig('./picture/result1.png')
    #plt.show()

    for i in range(10):
        plt.clf()  # 清除当前图形
        plt.scatter(energys[:4], Recall[i, :4], color='red')
        plt.scatter(energys[4:], Recall[i, 4:], color='blue')
        plt.xlabel('Energy Value')
        plt.ylabel('Recall')
        plt.title(classes[i])
        plt.savefig('./picture/result2_{}.png'.format(i))
        #plt.show()

    plt.clf()  # 清除当前图形
    plt.scatter(energys[:4], Accuracy[:4], color='red')
    plt.scatter(energys[4:], Accuracy[4:], color='blue')
    plt.xlabel('Energy Value')
    plt.ylabel('Accuracy')
    plt.title('Accuracy')
    plt.grid(True)
    plt.savefig('./picture/result3.png')
    #plt.show()

    for i in range(10):
        plt.clf()  # 清除当前图形
        plt.scatter(energys[:4], Recall[i, :4], color='red')
        plt.scatter(energys[4:], Recall[i, 4:], color='blue')
        plt.xlabel('Energy Value')
        plt.ylabel('Recall')
        plt.title(classes[i])
        plt.grid(True)
        plt.savefig('./picture/result4_{}.png'.format(i))
        #plt.show()

    print('结束')
