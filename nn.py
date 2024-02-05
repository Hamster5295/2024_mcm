import torch.optim
from torch import nn

from progress import ProgressBar

# data_train = datasets.FashionMNIST(root="Data", download=True, transform=ToTensor(), train=True)
# data_test = datasets.FashionMNIST(root="Data", download=True, transform=ToTensor(), train=True)

# dataset_train = DataLoader(data_train, batch_size=64)
# dataset_test = DataLoader(data_test, batch_size=64)

device = "cuda"
print("Using CUDA as device")


class UniformNetwork(nn.Module):
    def __init__(self, process: nn.Sequential):
        super().__init__()
        self.flatten = nn.Flatten()
        self.process = process

    def forward(self, x):
        x = self.flatten(x)
        y = self.process(x)
        return y


def train(model, dataloader, epochs=5, optimizer=None, loss_fn=nn.CrossEntropyLoss(), verbose=True):
    """
    训练一个模型
    :param model: 模型
    :param dataloader: 数据集对应的 DataLoader
    :param epochs: 训练周期数
    :param optimizer: 优化器，默认为 torch.nn.SGD
    :param loss_fn: 损失函数，默认为 nn.CrossEntropyLoss()
    :return: 训练好的模型
    """

    def train_epoch(epoch):
        model.train()
        size = len(dataloader.dataset)

        progress = ProgressBar(size, auto_linechange=False)

        if verbose:
            if epoch != 0:
                print()
            print(f"第 {epoch + 1} 轮训练:")
        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device).float(), y.to(device)

            pred = model(x)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if verbose:
                progress.add(dataloader.batch_size)
                print(f"     当前损失: {loss.item():>7f}", end="")

        if verbose:
            print(f"\n完成!")

    if optimizer is None:
        optimizer = torch.optim.SGD(model.parameters(), 1e-3)

    model = model.to(device)
    print("开始训练!")
    for i in range(epochs):
        train_epoch(i)
    print("训练完毕! \n")
    return model


def test(model, dataloader, loss_fn=nn.CrossEntropyLoss()):
    """
    测试模型的拟合程度
    :param model: 模型
    :param dataloader: 测试数据集
    :param loss_fn: 损失函数，默认为 nn.CrossEntropyLoss()
    """
    model = model.to(device)
    model.eval()
    size = len(dataloader.dataset)
    total_loss, total_correct = 0, 0
    print("开始测试!")
    progress = ProgressBar(size)
    for X, y in dataloader:
        X, y = X.to(device).float(), y.to(device)
        pred = model(X)
        total_loss += loss_fn(pred, y).item()
        total_correct += (pred.argmax(1) == y.argmax()).type(torch.float).sum().item()
        progress.add(X.shape[0])
    print(f"测试结果:\n正确率: {total_correct / size * 100:>3f} %\n平均损失: {total_loss / size:>7f}")


def save(model, path):
    """
    保存 pytorch 模型
    :param model: 模型
    :param path: 路径
    """
    torch.save(model.state_dict(), path)
    print(f"模型已保存至 {path} !")


def load(model, path):
    """
    读取nn模型。需要提供模型模板（一个同类型的空对象）
    :param model: 模型模板
    :param path: 路径
    :return: 读取后的模型
    """
    model = model.to(device)
    model.load_state_dict(torch.load(path))
    print("模型已读取!")
    return model
