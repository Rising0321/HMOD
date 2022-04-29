import torch
class OD_loss(torch.nn.Module):
    def __init__(self):
        super(OD_loss, self).__init__()
        self.pro = torch.nn.ReLU()

    def forward(self, predict, truth):
        mask = (truth < 1)
        mask2 = (predict > 0)
        loss = torch.mean(((predict - truth) ** 2) * ~mask + ((mask2 * predict - truth) ** 2) * mask)
        return loss