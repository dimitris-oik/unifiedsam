# Based on the repo: https://github.com/weizeming/SAM_AT
from typing import Iterable
import torch
from torch import nn


class unifiedSAM(torch.optim.Optimizer):
    def __init__(self, 
                 params: Iterable[nn.parameter.Parameter], 
                 base_optimizer: torch.optim.Optimizer, 
                 rho: float, 
                 lambd, 
                 **kwargs
                ):
        """
        lambd is either float or str
        Common choices:
        lambd = 0.0 gives USAM
        lambd = 1.0 gives NSAM
        lambd = 0.5 gives (USAM+NSAM)/2
        lambd = '1/t' gives NSAM -> USAM
        lambd = '1-1/t' gives USAM -> NSAM
        """
        defaults = dict(rho=rho, lambd=lambd, **kwargs)
        super(unifiedSAM, self).__init__(params, defaults)

        self.rho = rho
        self.lambd = lambd
        self.number_steps = 0
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        self.number_steps += 1
        grad_norm = self._grad_norm()

        if self.lambd == '1/t':
            scale = self.rho * (1 - 1/self.number_steps + 1/(self.number_steps*(grad_norm + 1e-12)))
        elif self.lambd == '1-1/t':
            scale = self.rho * (1/self.number_steps + (1-1/self.number_steps)/(grad_norm + 1e-12))
        else:
            lambd = float(self.lambd)
            scale = self.rho * (1 - lambd + lambd/(grad_norm + 1e-12))
        
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(torch.stack([p.grad.norm(p=2).to(shared_device) for group in self.param_groups for p in group["params"] if p.grad is not None]), p=2)
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups

