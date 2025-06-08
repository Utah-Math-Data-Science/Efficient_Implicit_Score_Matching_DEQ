import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch
from functools import partial
from . import get_sigmas
from .layers import *
from .normalization import get_normalization


class NCSNv2(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = num_classes = config.model.num_classes

        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.deq_C = 2
        self.normalizer = self.norm(ngf, self.num_classes)
        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)
        self.deq = config.model.deq
        self.shallow = config.model.shallow
        self.attention = config.model.attention

        if self.deq:
            
            if self.attention:
                self.conv_deq = nn.ModuleList([
                                                ResidualBlock(1 * self.ngf, 1 * self.ngf, resample=None, act=act,normalization=self.norm),
                                            ])
                self.attn = AttentionModule_stage1_cifar(1*self.ngf, 1*self.ngf, size1=(32, 32), size2=(16, 16)) 
                self.refine_deq = nn.ModuleList([
                                                RefineBlock([1 * self.ngf], self.ngf, act=act, end=True, deq=True),
                                                ])

            else:
                
                self.conv_deq = nn.ModuleList([
                                                # DeqBlock(self.ngf, self.ngf, resample=None, act=act,
                                                #             normalization=self.norm),
                                                ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                                                            normalization=self.norm),
                                                # ResidualBlock(self.ngf, self.ngf * 2, resample=None, act=act,
                                                #             normalization=self.norm)
                                                # ResidualBlock(1 * self.ngf, 1 * self.ngf, resample=None, act=act,
                                                # normalization=self.norm),

                                                # ResidualBlock(1 * self.ngf, 2 * self.ngf, resample='down', act=act,
                                                # normalization=self.norm, dilation=2),

                                                # ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                                                # normalization=self.norm, dilation=4)
                                             ])
                self.conv_deq_end = nn.ModuleList([
                                                ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                                                            normalization=self.norm),
                                                ResidualBlock(self.ngf, self.ngf * 2, resample=None, act=act,
                                                            normalization=self.norm)
                                             ])
                # self.refine_deq = nn.ModuleList([
                #                                 RefineBlock([2 * self.ngf, 1 * self.ngf], self.ngf, act=act, end=False, deq=True),
                #                                 RefineBlock([1 * self.ngf], self.ngf, act=act, end=True, deq=True)
                #                                 ])
                self.refine_deq0 = RefineBlock([2 * self.ngf, 1 * self.ngf], self.ngf, act=act, end=False, deq=True)
                self.refine_deq1 = RefineBlock([1 * self.ngf, 1 * self.ngf], self.ngf, act=act, end=True, deq=True)
                self.z_opt = {}

        elif self.shallow:
            self.res = nn.ModuleList([
                        ResidualBlock(1 * self.ngf, 1 * self.ngf, resample=None, act=act,
                                    normalization=self.norm),

                                    ResidualBlock(1 * self.ngf, 2 * self.ngf, resample='down', act=act,
                                    normalization=self.norm, dilation=2),

                                    ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                                    normalization=self.norm, dilation=4) 
                                    ] )

            self.refine = RefineBlock([self.ngf * 2, self.ngf], self.ngf, act=act, end=True)

            
        else:
            self.res1 = nn.ModuleList([
                ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                            normalization=self.norm),
                ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                            normalization=self.norm)]
            )

            self.res2 = nn.ModuleList([
                ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                            normalization=self.norm),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                            normalization=self.norm)]
            )

            self.res3 = nn.ModuleList([
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                            normalization=self.norm, dilation=2),
                ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                            normalization=self.norm, dilation=2)]
            )

            if config.data.image_size == 28:
                self.res4 = nn.ModuleList([
                    ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                                normalization=self.norm, adjust_padding=True, dilation=4),
                    ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                                normalization=self.norm, dilation=4)]
                )
            else:
                self.res4 = nn.ModuleList([
                    ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                                normalization=self.norm, adjust_padding=False, dilation=4),
                    ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                                normalization=self.norm, dilation=4)]
                )

            self.refine1 = RefineBlock([2 * self.ngf], 2 * self.ngf, act=act, start=True)
            self.refine2 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
            self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
            self.refine4 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

        
    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)
        
        if self.deq:
        ### deq ####
            with torch.no_grad():
                str_name = str(output.shape[0]) + '_' + str(output.shape[1]) + '_' + str(output.shape[2]) + '_' + str(output.shape[3]) + '_CUDA{}'.format(str(output.device).split(':')[1])
                try:
                    z = self.z_opt[str_name]
                    # print(z)
                except:
                    z = torch.zeros_like(output)
                w_norm = self.deq_C * (self.conv_deq[0].conv1.weight.norm(p=2) * self.conv_deq[0].conv2.weight.norm(p=2)+1e-12)
                zz = torch.randn_like(output)

                for iter in range(128):
                    zzn = self.conv_deq[0](zz) / w_norm + output
                    err = (zz-zzn).norm().item()
                    zz_norm = zzn.norm().item()
                    zz = zzn
                    if err < 1e-5:
                        self.z_opt.update({str_name : zz.detach()})
                        break
                # print('DEQ RES ** - ** iter: {} | err: {:.6f} | y_norm: {:.2f} | w_norm {:.2f} '.format(iter+1, err, zz_norm, w_norm))
                conv_w_zz = self.conv_deq[0](self.conv_deq[0](zz)) / w_norm 
                # w_norm = 1  

            deq_out = self.conv_deq[0](conv_w_zz+output) / w_norm  + output

            # z = self.conv_deq[0](conv_w_zz+output) / w_norm  + output
            # output = self.conv_deq[0](z) / w_norm  + output

            # z = self.conv_deq[0](conv_w_zz+output) / w_norm  + output
            # z = self.conv_deq[0](z) / w_norm  + output
            # output = self.conv_deq[0](z) / w_norm  + output

            if not self.attention: 
                z = self._compute_cond_module(self.conv_deq_end, deq_out)
                z = self.refine_deq0([z, deq_out], z.shape[2:])
                output = self.refine_deq1([z, output], z.shape[2:])

            else:
                output = self.attn(output)
                output = self.refine_deq([output], output.shape[2:])
            #### deq end ####
        elif self.shallow:
            layer = self.res[1](output)
            layer = self.res[2](layer)
            output = self.refine([layer, output], layer.shape[2:])
        else:
            layer1 = self._compute_cond_module(self.res1, output)
            layer2 = self._compute_cond_module(self.res2, layer1)
            layer3 = self._compute_cond_module(self.res3, layer2)
            layer4 = self._compute_cond_module(self.res4, layer3)
            ref1 = self.refine1([layer4], layer4.shape[2:])
            ref2 = self.refine2([layer3, ref1], layer3.shape[2:])
            ref3 = self.refine3([layer2, ref2], layer2.shape[2:])
            output = self.refine4([layer1, ref3], layer1.shape[2:])
        
        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))
        output = output / used_sigmas

        return output


class NCSNv2Deeper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer4 = self._compute_cond_module(self.res4, layer3)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref3 = self.refine3([layer3, ref2], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output


class NCSNv2Deepest(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.logit_transform = config.data.logit_transform
        self.rescaled = config.data.rescaled
        self.norm = get_normalization(config, conditional=False)
        self.ngf = ngf = config.model.ngf
        self.num_classes = config.model.num_classes
        self.act = act = get_act(config)
        self.register_buffer('sigmas', get_sigmas(config))
        self.config = config

        self.begin_conv = nn.Conv2d(config.data.channels, ngf, 3, stride=1, padding=1)
        self.normalizer = self.norm(ngf, self.num_classes)

        self.end_conv = nn.Conv2d(ngf, config.data.channels, 3, stride=1, padding=1)

        self.res1 = nn.ModuleList([
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm),
            ResidualBlock(self.ngf, self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res2 = nn.ModuleList([
            ResidualBlock(self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res3 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res31 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample='down', act=act,
                          normalization=self.norm),
            ResidualBlock(2 * self.ngf, 2 * self.ngf, resample=None, act=act,
                          normalization=self.norm)]
        )

        self.res4 = nn.ModuleList([
            ResidualBlock(2 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=2),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=2)]
        )

        self.res5 = nn.ModuleList([
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample='down', act=act,
                          normalization=self.norm, dilation=4),
            ResidualBlock(4 * self.ngf, 4 * self.ngf, resample=None, act=act,
                          normalization=self.norm, dilation=4)]
        )

        self.refine1 = RefineBlock([4 * self.ngf], 4 * self.ngf, act=act, start=True)
        self.refine2 = RefineBlock([4 * self.ngf, 4 * self.ngf], 2 * self.ngf, act=act)
        self.refine3 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine31 = RefineBlock([2 * self.ngf, 2 * self.ngf], 2 * self.ngf, act=act)
        self.refine4 = RefineBlock([2 * self.ngf, 2 * self.ngf], self.ngf, act=act)
        self.refine5 = RefineBlock([self.ngf, self.ngf], self.ngf, act=act, end=True)

    def _compute_cond_module(self, module, x):
        for m in module:
            x = m(x)
        return x

    def forward(self, x, y):
        if not self.logit_transform and not self.rescaled:
            h = 2 * x - 1.
        else:
            h = x

        output = self.begin_conv(h)

        layer1 = self._compute_cond_module(self.res1, output)
        layer2 = self._compute_cond_module(self.res2, layer1)
        layer3 = self._compute_cond_module(self.res3, layer2)
        layer31 = self._compute_cond_module(self.res31, layer3)
        layer4 = self._compute_cond_module(self.res4, layer31)
        layer5 = self._compute_cond_module(self.res5, layer4)

        ref1 = self.refine1([layer5], layer5.shape[2:])
        ref2 = self.refine2([layer4, ref1], layer4.shape[2:])
        ref31 = self.refine31([layer31, ref2], layer31.shape[2:])
        ref3 = self.refine3([layer3, ref31], layer3.shape[2:])
        ref4 = self.refine4([layer2, ref3], layer2.shape[2:])
        output = self.refine5([layer1, ref4], layer1.shape[2:])

        output = self.normalizer(output)
        output = self.act(output)
        output = self.end_conv(output)

        used_sigmas = self.sigmas[y].view(x.shape[0], *([1] * len(x.shape[1:])))

        output = output / used_sigmas

        return output


