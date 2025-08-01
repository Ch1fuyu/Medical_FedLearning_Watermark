import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

def accuracy(output, target, top_k=(1,)):
    with torch.no_grad():
        max_k = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(max_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class TesterPrivate(object):
    def __init__(self, model, device, verbose=True):
        self.model = model
        self.device = device
        self.verbose = verbose

    def test(self, dataloader):
        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        run_count = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)

                pred = self.model(data)
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item()
                pred = pred.max(1, keepdim=True)[1]
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                run_count += data.size(0)

        loss_meter /= run_count
        acc_meter /= run_count

        return loss_meter, acc_meter

class TrainerPrivate(object):
    def __init__(self, model, device, dp, sigma, random_positions):
        self.optimizer = None
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tester = TesterPrivate(model, device)
        self.dp = dp
        self.sigma = sigma
        self.random_positions = random_positions  # 加入随机位置列表

    def local_update(self, dataloader, local_ep, lr, client_id):
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr,
                                   momentum=0.9,
                                   weight_decay=0.0005)

        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        train_ldr = dataloader

        # 获取该客户端的随机参数位置
        position_dict = self.random_positions[client_id]

        for epoch in range(local_ep):
            loss_meter = 0
            acc_meter = 0

            for batch_idx, (x, y) in enumerate(train_ldr):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                loss = torch.tensor(0.).to(self.device)

                # 获取模型预测结果
                pred = self.model(x)
                loss += F.cross_entropy(pred, y)
                acc_meter += accuracy(pred, y)[0].item()

                # params = torch.cat([param.view(-1) for param in self.model.parameters()])
                # # 在每个batch结束后操作指定的随机位置参数
                # for layer_name, param_idx in positions:
                #     print(params.data[param_idx])

                loss.backward()
                self.optimizer.step()

                loss_meter += loss.item()

            loss_meter /= len(train_ldr)
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)

            # 将整个参数空间展开成一维张量
            all_params = torch.cat([param.view(-1) for param in self.model.parameters()])
            # 在每轮训练结束后，将指定位置的参数设置为 0.5
            for _, param_idx in position_dict:
                all_params[param_idx] = torch.tensor(0.5, device=self.device)

            # 将修改后的一维张量重新赋值给模型参数
            start_idx = 0
            for param in self.model.parameters():
                param.data = all_params[start_idx:start_idx + param.numel()].view(param.size())
                start_idx += param.numel()

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss)

    def local_update_with_no_fl(self, dataloader, local_ep, lr):
        self.optimizer = optim.SGD(self.model.parameters(),
                                   lr,
                                   momentum=0.9,
                                   weight_decay=0.0005)

        self.model.to(self.device)
        self.model.train()

        for epoch in range(local_ep):
            loss_meter = 0
            acc_meter = 0

            for batch_idx, (x, y) in enumerate(dataloader):
                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()

                pred = self.model(x)
                loss = F.cross_entropy(pred, y)
                acc_meter += accuracy(pred, y)[0].item()

                loss.backward()
                self.optimizer.step()

                loss_meter += loss.item()

            loss_meter /= len(dataloader)
            acc_meter /= len(dataloader)

        # 如果启用差分隐私（DP），为每个参数添加噪声
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

    def test(self, dataloader):
        return self.tester.test(dataloader)
