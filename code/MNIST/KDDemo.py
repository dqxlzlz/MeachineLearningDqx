import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary

# 设置随机种子的目的是为了确保实验的可重复性。在许多深度学习任务中，随机性是一个重要的因素。例如，初始化神经网络的权重或在训练时打乱数据集。通过固定随机种子，您可以确保每次运行代码时都会得到相同的结果。
# 设置随机种子
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用cuda进行加速卷积运算
torch.backends.cudnn.benchmark = True
# 载入训练集

# batch_size是指用于训练神经网络时一次性输入模型的样本数量。Batch size的设置会影响模型训练的速度和准确性。较小的batch size可能会导致训练速度较慢，但可以提供更精确的梯度估计。较大的batch size可以加速训练，但可能导致梯度估计不准确。选择合适的batch size取决于任务、模型和计算资源。
train_dataset = torchvision.datasets.MNIST(root="dataset/", train=True, transform=transforms.ToTensor(), download=True)
test_dateset = torchvision.datasets.MNIST(root="dataset/", train=False, transform=transforms.ToTensor(), download=True)
train_dataloder = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloder = DataLoader(test_dateset, batch_size=32, shuffle=True)


# 搭建网络
# teacher
class Teacher_model(nn.Module):
    def __init__(self, in_channels=1, num_class=10):
        # 全连接层的设计选择了将784维的输入向量映射到1200维的向量，然后再映射到另一个1200维的向量。这些数字（784 - 1200 - 1200）是超参数，可以根据实验结果和经验进行选择
        super(Teacher_model, self).__init__()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        # rele激活函数是一种非线性激活函数，它在神经网络中广泛使用，因为它的计算速度快且能有效地缓解梯度消失问题。ReLU的公式如下:f(x) = max(0, x)
        self.relu = nn.ReLU()
        # Dropout是一种正则化技术，用于防止神经网络过拟合。在训练过程中，Dropout层以一定概率随机丢弃神经元的输出，这样可以强制网络在训练过程中学习更加鲁棒的特征表示。
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = Teacher_model()
model = model.to(device)

# 损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# epoches = 6：设置训练轮数（epoch）为6。一个epoch表示遍历整个训练数据集一次。
epoches = 6
for epoch in range(epoches):
    model.train()
    for image, label in train_dataloder:
        image, label = image.to(device), label.to(device)
        # 在进行反向传播之前，先将模型参数的梯度清零
        optim.zero_grad()
        # 将输入图像传递给模型，获得输出
        out = model(image)
        # 计算预测值和真实标签之间的损失值
        loss = loss_function(out, label)
        # 根据损失值进行反向传播，计算模型参数的梯度
        loss.backward()
        # 使用优化器（如SGD、Adam等）根据计算出的梯度更新模型参数
        optim.step()

    model.eval()
    num_correct = 0
    num_samples = 0

    # 在评估模式下，不需要计算梯度，这可以节省计算资源和内存
    with torch.no_grad():
        for image, label in test_dataloder:
            image = image.to(device)
            label = label.to(device)
            out = model(image)
            pre = out.max(1).indices
            num_correct += (pre == label).sum()
            num_samples += pre.size(0)
        acc = (num_correct / num_samples).item()

    model.train()
    print("epoches:{},accurate={}".format(epoch, acc))

teacher_model = model


# 构建学生模型
class Student_model(nn.Module):
    def __init__(self, in_channels=1, num_class=10):
        super(Student_model, self).__init__()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.dropout(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


model = Student_model()
model = model.to(device)

# 损失函数和优化器
loss_function = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=0.0001)

# 开始进行知识蒸馏算法
teacher_model.eval()
model = Student_model()
model = model.to(device)
# 蒸馏温度
T = 7
hard_loss = nn.CrossEntropyLoss()
alpha = 0.3
soft_loss = nn.KLDivLoss(reduction="batchmean")
optim = torch.optim.Adam(model.parameters(), lr=0.0001)
