from __future__ import division

import torch
# from src.base_net import *


class Sampler:
    def chol_scale_invert_kron_factor(factor, prior_scale, data_scale, upper=False):
        scaled_factor = data_scale * factor + prior_scale * torch.eye(
            factor.shape[0]
        ).type(factor.type())
        inv_factor = torch.inverse(scaled_factor)
        chol_inv_factor = torch.cholesky(inv_factor, upper=upper)
        return chol_inv_factor

    def sample_K_laplace_MN(self, MAP, upper_Qinv, lower_HHinv):
        # H = Qi (kron) HHi
        # sample isotropic unit variance mtrix normal
        Z = MAP.data.new(MAP.size()).normal_(mean=0, std=1)
        # AAT = HHi
        #     A = torch.cholesky(HHinv, upper=False)
        # BTB = Qi
        #     B = torch.cholesky(Qinv, upper=True)
        all_mtx_sample = MAP + torch.matmul(torch.matmul(lower_HHinv, Z), upper_Qinv)

        weight_mtx_sample = all_mtx_sample[:, :-1]
        bias_mtx_sample = all_mtx_sample[:, -1]

        return weight_mtx_sample, bias_mtx_sample
