import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim

#1.下载CIFAR10数据
#torchvision数据集的输出是范围在[0,1]的PILImage图像。我们转换它们成有着标准化范围[-1,1]的张量
transform = transforms.Compose(
    [transforms.ToTensor(),
    #这里的参数可以进行修改，改成transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#已经是很小的batch-size了
batch_size = 4
#后面调参时，download设置为False
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    # 反正则化
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# 随机得到一些训练图像
dataiter = iter(trainloader) #生成迭代器
images, labels = dataiter.next() #每次运行next()就会调用trainloader，获得一个之前定义的batch_size=4的批处理图片集，即4张图片

# 展示图像
imshow(torchvision.utils.make_grid(images)) #make_grid的作用是将若干幅图像拼成一幅图像,在想要展示一批数据的时候十分有用
# 输出图像标签
print(' '.join('%5s' % classes[labels[j]] for j in range(batch_size)))

#2.定义一个卷积神经网络
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # 声明继承
        self.conv1 = nn.Conv2d(3, 6, 5)  # 卷积层，参数为（inchannel,outchannel=number of filter,siza of filter,stride,padding）
        self.pool = nn.MaxPool2d(2, 2)  # 池化层
        self.conv2 = nn.Conv2d(6, 16, 5)  # 卷积层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)  # 全连接层
        self.fc3 = nn.Linear(84, 10)  # 全连接层，最后输出10个神经元，用于判断该图为哪个类别

    def forward(self, x):  # 实现前向传播
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)  # 用来将x展平成16 * 5 * 5，然后就可以进行下面的全连接层操作
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

#3.定义一个损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

#4.训练网络
for epoch in range(2):  # 多次循环数据集，这里循环训练整个数据集两次
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0): #enumerate枚举数据并从下标0开始
        # 得到输入数据
        inputs, labels = data

        # 将参数的梯度都设为0
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs) #forward
        loss = criterion(outputs, labels) #计算损失
        loss.backward() #后向传播
        optimizer.step() #将优化后的参数输入网络，再次进行训练

        #打印数据
        running_loss += loss.item() #用于从tensor中获取python数字
        if i % 2000 == 1999:    # 每处理2000次小批处理数据后打印一次结果
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0 #然后清0

print('Finished Training')

#5.在测试数据中测试网络
dataiter = iter(testloader)
images, labels = dataiter.next()

# 打印图片
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))
#看神经网络的判断
outputs = net(images)
#输出结果
_, predicted = torch.max(outputs, 1)
print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

#查看整体表现
correct = 0
total = 0
with torch.no_grad(): #设置为不计算梯度
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item() #相等

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

#查看具体表现
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad(): #设置在进行forward时不计算梯度
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))




