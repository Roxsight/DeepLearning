import torch
import torch.nn.functional as F


def cons_loss(pv, pa, labels, tau=0.07, delta=2):
    real = labels < 0.5
    if real.sum() == 0:
        return pv.new_zeros(()).float()
    pv = pv[real].float()
    pa = pa[real].float()
    sim = torch.einsum('btd,bkd->btk', pv, pa) / tau
    T = sim.size(1)
    idx = torch.arange(T, device=sim.device)
    far = (idx[None, :] - idx[:, None]).abs() > delta
    sim = sim.masked_fill(far[None], float('-inf'))
    logp = sim.log_softmax(dim=-1)
    return -logp.diagonal(dim1=1, dim2=2).mean()


def get_loss(out, labels, lam=0.5, tau=0.07, delta=2, pos_weight=None, use_cons=True):
    labels = labels.float()
    pw = None
    if pos_weight is not None:
        pw = torch.tensor(pos_weight, device=labels.device)
    l_cls = F.binary_cross_entropy_with_logits(out['logit'].float(), labels, pos_weight=pw)
    if use_cons:
        l_cons = cons_loss(out['pv'], out['pa'], labels, tau, delta)
    else:
        l_cons = l_cls.new_zeros(())
    loss = l_cls + lam * l_cons
    return loss, {'cls': l_cls.item(), 'cons': float(l_cons.detach())}
