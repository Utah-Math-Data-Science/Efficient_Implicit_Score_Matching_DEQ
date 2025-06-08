import torch
import torch.autograd as autograd
from tqdm import tqdm
import time
import logging

########################################
# The exact score matching does not
# work for deep energy learning, not only
# because of computational complexity but
# also because of memory.
########################################
def score_matching(energy_net, samples):
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples)[0]
    loss1 = (torch.norm(grad1, dim=-1) ** 2 / 2.).detach()

    loss2 = torch.zeros(samples.shape[0], device=samples.device)
    for i in tqdm.tqdm(range(samples.shape[1])):
        logp = -energy_net(samples).sum()
        grad1 = autograd.grad(logp, samples, create_graph=True)[0]
        grad = autograd.grad(grad1[:, i].sum(), samples)[0][:, i]
        loss2 += grad.detach()

    loss = loss1 + loss2
    return loss.mean()


def exact_score_matching_old(energy_net, samples, train=False):
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
    if train:
        loss1.mean().backward(retain_graph=True)

    loss2 = torch.zeros(samples.shape[0], device=samples.device)
    for i in range(samples.shape[1]):
        grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=True)[0][:, i]
        if train:
            grad.mean().backward(retain_graph=True)
        loss2 += grad.detach()

    loss = loss1 + loss2
    return loss.mean().detach()

def exact_score_matching(energy_net, samples, train=False):
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = autograd.grad(logp, samples, create_graph=True)[0]
    loss1 = torch.norm(grad1, dim=-1) ** 2 / 2.
    loss2 = torch.zeros(samples.shape[0], device=samples.device)

    # if samples.shape[1] > 100:
    #     iterator = tqdm(range(samples.shape[1]))
    # else:
    iterator = range(samples.shape[1])
    scale_ = torch.eye(samples.shape[-1], device=grad1.device)
    if train:
        grad2 = autograd.grad(grad1.sum(dim=0), samples, grad_outputs=scale_, create_graph=True, is_grads_batched=True)[0] 
    if not train:
        grad2 = autograd.grad(grad1.sum(dim=0), samples, grad_outputs=scale_, create_graph=False, is_grads_batched=True)[0] 
    for i in iterator:
        grad = grad2[i, :, i]
        if not train:
            grad = grad.detach()
        loss2 += grad
    # for i in iterator:
    #     if train:
    #         grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=True, retain_graph=True)[0][:, i]
    #     if not train:
    #         grad = autograd.grad(grad1[:, i].sum(), samples, create_graph=False, retain_graph=True)[0][:, i]
    #         grad = grad.detach()
    #     loss2 += grad

    loss = loss1 + loss2

    if not train:
        loss = loss.detach()

    return loss
