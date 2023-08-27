import torch


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    target = target.squeeze()
    output = output.float()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        try:
            _, pred = output.topk(maxk, 1, True, True)
        except:
            res = accuracy(output, target, topk[:-1])
            return res + [0.]
            
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / (batch_size)).item())
        return res
    