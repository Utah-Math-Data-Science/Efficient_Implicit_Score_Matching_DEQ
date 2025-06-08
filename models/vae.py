import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.main = nn.Sequential(
            nn.Linear(784, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh()
        )
        self.mean = nn.Linear(256, self.z_dim)
        self.log_std = nn.Linear(256, self.z_dim)

    def forward(self, inputs):
        h = self.main(inputs.view(inputs.shape[0], -1))
        mean = self.mean(h)
        log_std = self.log_std(h)
        return mean, log_std


class MLPDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.main = nn.Sequential(
            nn.Linear(self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 784)
        )

    def forward(self, inputs):
        return self.main(inputs.view(inputs.shape[0], -1)).view(inputs.shape[0],
                                                                self.channels,
                                                                self.image_size,
                                                                self.image_size)


class MLPScore(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.z_dim = config.model.z_dim
        self.main = nn.Sequential(
            nn.Linear(784 + self.z_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.z_dim)
        )

    def forward(self, X, z):
        X = X.view(X.shape[0], -1)
        Xz = torch.cat([X, z], dim=-1)
        h = self.main(Xz)
        return h


class MLPImplicitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.eps_dim = config.model.eps_dim
        self.z_dim = config.model.z_dim
        self.main = nn.Sequential(
            nn.Linear(784 + self.eps_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, self.z_dim)
        )

    def forward(self, X):
        eps = torch.randn(X.shape[0], self.eps_dim, device=X.device)
        flattened_X = X.view(X.shape[0], -1)
        X_eps = torch.cat([flattened_X, eps], dim=-1)
        z = self.main(X_eps)
        return z


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset

        if self.dataset == 'CELEBA':
            self.main = nn.Sequential(
                nn.Conv2d(self.channels, self.nef * 1, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2),
                nn.ReLU(inplace=True)
            )

            self.flatten = nn.Sequential(
                nn.Linear((self.image_size // 2 ** 4) ** 2 * self.nef * 8, 512),
                nn.ReLU(inplace=True)
            )

        elif self.dataset == 'CIFAR10':
            self.main = nn.Sequential(
                nn.Conv2d(self.channels, self.nef * 1, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.nef * 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.nef * 1, self.nef * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.nef * 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.nef * 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.nef * 2, self.nef * 1, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.nef * 1),
                nn.ReLU(inplace=True)
            )

            self.flatten = nn.Sequential(
                nn.Linear((self.image_size // 2 ** 2) ** 2 * self.nef * 1, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True)
            )

        self.mean = nn.Linear(512, self.z_dim)
        self.log_std = nn.Linear(512, self.z_dim)


    def forward(self, inputs):

        h = self.main(inputs)
        h = h.view(h.shape[0], -1)
        h = self.flatten(h)
        mean = self.mean(h)
        log_std = self.log_std(h)

        return mean, log_std


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ndf = config.model.ndf
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.fix_dec = config.model.fix_dec
        if config.model.fix_ssm:
            self.scale = 1.0
        else:
            self.scale = 1.00

        if self.dataset == 'CIFAR10':

            self.fc = nn.Sequential(
                    nn.Linear(self.z_dim, self.z_dim*2),
                    nn.BatchNorm1d(self.z_dim*2),
                    nn.ReLU(),  

                    nn.Linear(self.z_dim*2, self.ndf * 1 * 8 * 8),
                    nn.BatchNorm1d(self.ndf * 1 * 8 * 8),
                    nn.ReLU(),  
                    )
            
            ## cifar ##
            self.main = nn.Sequential(
                nn.ConvTranspose2d(self.ndf * 1, self.ndf * 2, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(self.ndf * 2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(self.ndf * 2, self.ndf * 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.ndf * 2),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(self.ndf * 2, self.ndf*1, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(self.ndf * 1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(self.ndf*1, self.channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.channels),
                nn.Tanh()
            )

        elif self.dataset == 'CELEBA':

            self.main = nn.Sequential(
                nn.ConvTranspose2d(self.ndf * 8, self.ndf * 4, 5, stride=2, padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.ndf * 4, self.ndf * 2, 5, stride=2, padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.ndf * 2, self.ndf, 5, stride=2, padding=2, output_padding=1),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(self.ndf, self.channels, 5, stride=2, padding=2, output_padding=1),
                nn.Tanh()
            )

            self.fc = nn.Linear(self.z_dim, self.ndf * 8 * 4 * 4)  

        

        self.log_std = nn.Parameter(torch.ones(1, self.channels, self.image_size, self.image_size) * -0.602,
                                    requires_grad=False)
        self.y_opt = {}
        self.relu1 = nn.ReLU()
        self.linear1 = nn.Linear(self.ndf * 8 * 4 * 4, self.ndf * 8 * 4 * 4, bias=False)

    def forward(self, inputs):

        h = self.fc(inputs)
        if self.dataset == 'CIFAR10': 
            h = h.view(-1, self.ndf * 1, 8, 8) 
            h = (self.main(h) + 1) / 2
            if self.training: 
                h[:,0,:,:] = (h[:,0,:,:] - 0.4914) / 0.2023 * 1.00 
                h[:,1,:,:] = (h[:,1,:,:] - 0.4822) / 0.1994 * 1.00
                h[:,2,:,:] = (h[:,2,:,:] - 0.4465) / 0.2010 * 1.00
            else:
                h[:,0,:,:] = (h[:,0,:,:] - 0.4914) / 0.2023 * self.scale 
                h[:,1,:,:] = (h[:,1,:,:] - 0.4822) / 0.1994 * self.scale
                h[:,2,:,:] = (h[:,2,:,:] - 0.4465) / 0.2010 * self.scale 
        else:
            h = h.view(-1, self.ndf * 8, 4, 4) 
            h = self.main(h)


        mean = h.view(-1, self.channels, self.image_size, self.image_size)
        
        return mean, self.log_std


class Score(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.z_dim = config.model.z_dim
        self.channels = config.data.channels
        self.image_size = config.data.image_size
        self.dataset = config.data.dataset
        self.fix_ssm = config.model.fix_ssm
        self.deep_ssm = config.model.deep_ssm
        self.origin_ssm = config.model.origin_ssm
        self.deep_num = 16
        self.C = 1e1
        self.alpha = 5e-2
        self.val_iter = 4
        self.zfc = nn.Sequential(
            nn.Linear(self.z_dim, self.image_size ** 2),
            nn.BatchNorm1d(self.image_size ** 2),
            nn.Softplus()
        )
        self.y_opt = {}
        self.err_opt = {}
        if self.dataset == 'CIFAR10':
            self.score = nn.Linear(self.z_dim*2, self.z_dim)
        else:
            self.score = nn.Linear(512, self.z_dim)

        if config.model.fix_ssm or config.model.deep_ssm:
            
            ## DEQ for CIFAR
            if config.data.dataset == 'CIFAR10':
                self.conv_pd = nn.Sequential(
                    nn.Conv2d(self.channels + 1, self.nef*2, 3, stride=2, padding=1),
                    # nn.BatchNorm2d(self.nef * 1),
                    nn.Softplus()
                )
                self.main_fix_NN = nn.ModuleList([
                    nn.Sequential(
                        nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1),
                        # nn.BatchNorm2d(self.nef * 2),
                        nn.Softplus()),

                    nn.Sequential(
                        nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1),
                        # nn.BatchNorm2d(self.nef * 2),
                        nn.Softplus()),
                    
                    nn.Sequential(
                        nn.Conv2d(self.nef * 2, self.nef * 4, 3, stride=2, padding=1),
                        # nn.BatchNorm2d(self.nef * 2),
                        nn.Softplus()),
                    ])

                self.main_fix_ = nn.Sequential(
                    nn.Conv2d(self.nef * 1, self.nef * 1, 3, stride=2, padding=1),
                    # nn.BatchNorm2d(self.nef * 1),
                    nn.Softplus(),

                    nn.Conv2d(self.nef * 1, self.nef * 2, 3, stride=2, padding=1),
                    # nn.BatchNorm2d(self.nef * 2),
                    nn.Softplus(),

                    nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1),
                    # nn.BatchNorm2d(self.nef * 4),
                    nn.Softplus(),

                    nn.Conv2d(self.nef * 2, self.nef * 1, 3, stride=2, padding=1),
                    # nn.BatchNorm2d(self.nef * 1),
                    nn.Softplus()
                    )

                self.flatten = nn.Sequential(
                    nn.Linear((self.image_size // 2**2) ** 2 * self.nef * 4, self.z_dim*2),
                    nn.BatchNorm1d(self.z_dim*2),
                    nn.Softplus()
                )

                self.flatten_ = nn.Sequential(
                    nn.Linear((self.image_size // 2**3) ** 2 * self.nef * 1, 512),
                    nn.Softplus()
                )

                self.conv_pd_deep = nn.Sequential(
                    nn.Conv2d(self.channels + 1, self.nef*1, 3, stride=1, padding=1),
                    nn.Softplus()
                )

                if config.model.deep_ssm:
                    self.conv_deep = nn.ModuleList(
                                        [nn.Sequential(
                                            nn.Conv2d(self.nef * 1, self.nef * 2, 3, stride=1, padding=1),
                                            nn.Softplus(),
                                            nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1),
                                            nn.Softplus(),
                                            nn.Conv2d(self.nef * 2, self.nef * 1, 3, stride=1, padding=1),
                                            nn.Softplus()
                                            )for i in range(self.deep_num)]
                                    )
                elif config.model.fix_ssm:
                    self.conv_lin_list = nn.ModuleList([nn.Sequential(
                                                        nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1, bias=False),
                                                        nn.Softplus(),
                                                        ) for i in range(4)])

            ## DEQ for CeleBa
            else:
                self.conv_pd = nn.Sequential(
                    nn.Conv2d(self.channels + 1, self.nef, 5, stride=1, padding=2),
                    nn.Softplus()
                )
                self.main_fix = nn.Sequential(
                    nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
                    nn.Softplus(),
                    nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
                    nn.Softplus(),
                    nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2),
                    nn.Softplus(),
                )
                
                self.flatten = nn.Sequential(
                    nn.Linear((self.image_size // 2**3) ** 2 * self.nef * 8, 512),
                    nn.Softplus()
                )

                if config.model.deep_ssm:
                    self.conv_deep = nn.ModuleList(
                                        [nn.Sequential(
                                        nn.Conv2d(self.nef*1, self.nef*1, 5, stride=1, padding=2, bias=False),
                                        nn.ReLU(),
                                        ) for i in range(self.deep_num)]
                                    )
                elif config.model.fix_ssm:
                    self.conv_lin = nn.Sequential(
                                            nn.Conv2d(self.nef, self.nef, 5, stride=1, padding=2, bias=False),
                                            nn.ReLU(),
                                            nn.Conv2d(self.nef, self.nef, 5, stride=1, padding=2, bias=False),
                                            )
                    self.conv_lin_list = nn.ModuleList([nn.Sequential(
                                                        nn.Conv2d(self.nef*1, self.nef*1, 5, stride=1, padding=2, bias=False),
                                                        nn.Softplus(),
                                                        ) for i in range(4)])


        else:
            ## SCORE for CIFAR
            if config.data.dataset == 'CIFAR10':

                ### cifar original main ###
                self.main = nn.Sequential(
                    nn.Conv2d(self.channels + 1, self.nef * 1, 3, stride=2, padding=1),
                    nn.Softplus(),

                    nn.Conv2d(self.nef * 1, self.nef * 2, 3, stride=2, padding=1),
                    nn.Softplus(),

                    nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1),
                    nn.Softplus(),

                    nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=2, padding=1),
                    nn.Softplus()
                    )

                self.flatten = nn.Sequential(
                    nn.Linear((self.image_size // 2**3) ** 2 * self.nef * 2, 2*self.z_dim),
                    nn.Softplus()
                )

            ## SCORE for Celeba    
            else:
                self.main = nn.Sequential(
                nn.Conv2d(self.channels + 1, self.nef * 1, 5, stride=2, padding=2),
                nn.Softplus(),
                nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
                nn.Softplus(),
                nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
                nn.Softplus(),
                nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2),
                nn.Softplus()
                )
                self.flatten = nn.Sequential(
                    nn.Linear((self.image_size // 2**4) ** 2 * self.nef * 8, 512),
                    nn.Softplus()
                )

    def fix_main_func(self, h):
        if self.dataset == 'CIFAR10':
            h = self.main_fix_NN[0](h) + h
            # h = self.main_fix_NN[1](h) + h
            h = self.main_fix_NN[2](h)
        else:
            h = self.conv_lin_list[1](h) + h
            h = self.main_fix(h)

        return h

    def forward(self, X, z):
        z = self.zfc(z).view(X.shape[0], 1, X.shape[2], X.shape[3])
        Xz = torch.cat([X, z], dim=1)
        if self.origin_ssm:
            h = self.main(Xz)
            h = h.view(h.shape[0], -1)
            h = self.flatten(h)
            score = self.score(h)
        else:
            if self.fix_ssm:
                with torch.no_grad():
                    # h = Xz
                    h = self.conv_pd(Xz)
                    w_norm = (self.conv_lin_list[0][0].weight.norm(p=2) + 1e-6) * self.C 
                    
                    ## store the fix point of current deq
                    str_name = str(h.shape[0]) + '_' + str(h.shape[1]) + '_' + str(h.shape[2]) + '_' + str(h.shape[3])
                    try:
                        y = self.y_opt[str_name]
                    except:
                        y = torch.zeros_like(h)

                    for iter in range(128):
                        yn = (self.conv_lin_list[0](y)) / w_norm + h
                        err = (y-yn).norm().item()
                        y = yn
                        if err < 5e-6:
                            self.y_opt.update({str_name : y.detach()})
                            break
                    conv_w_y = (self.conv_lin_list[0](y)) / w_norm
                
                h_ = self.conv_pd(Xz) 
                if self.dataset == 'CIFAR10':
                    ## beta # p gradient
                    h_ = self.conv_lin_list[0](h_+ conv_w_y)/ w_norm + h_
                    if not self.training:
                        for ik in range(self.val_iter):
                            h_ = (self.conv_lin_list[0](h_)) / w_norm +  self.conv_pd(Xz) 
                    h_ = self.fix_main_func(h_)
                else:
                    h_ = self.conv_lin_list[0](h_+ conv_w_y)/ w_norm + h_
                    if not self.training:
                        for ik in range(self.val_iter):
                            h_ = (self.conv_lin_list[0](h_)) / w_norm +  self.conv_pd(Xz) 
                    h_ = self.fix_main_func(h_)

                h_ = h_.view(h.shape[0], -1)
                h_ = self.flatten(h_)
             
                # out           
                score = self.score(h_)

            elif self.deep_ssm:
                h = self.conv_pd_deep(Xz)
                for k in range(self.deep_num):
                    h = self.conv_deep[k](h) + self.alpha * self.conv_pd_deep(Xz)
                h = self.main_fix_(h)
                h = h.view(h.shape[0], -1)
                h = self.flatten_(h)
                score = self.score(h)

            else:
                ('SSM Moded Name Error!')


        return score


class ImplicitEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nef = config.model.nef
        self.z_dim = config.model.z_dim
        self.eps_dim = config.model.eps_dim
        self.image_size = config.data.image_size
        self.channels = config.data.channels
        self.dataset = config.data.dataset
        self.noise_fc = nn.Sequential(
            nn.Linear(self.eps_dim, self.image_size ** 2),
            nn.BatchNorm1d(self.image_size ** 2),
            nn.ReLU(inplace=True)
        )
        if config.data.dataset == 'CIFAR10':
            self.main = nn.Sequential(
                nn.Conv2d(self.channels+1, self.nef * 1, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.nef * 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.nef * 1, self.nef * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.nef * 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.nef * 2, self.nef * 2, 3, stride=1, padding=1),
                nn.BatchNorm2d(self.nef * 2),
                nn.ReLU(inplace=True),

                nn.Conv2d(self.nef * 2, self.nef * 1, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.nef * 1),
                nn.ReLU(inplace=True)
            )

            self.flatten = nn.Sequential(
                nn.Linear((self.image_size // 2 ** 2) ** 2 * self.nef * 1, self.z_dim * 2),
                nn.BatchNorm1d(self.z_dim * 2),
                nn.ReLU(inplace=True)
            )
            self.mean = nn.Sequential(
                        nn.Linear(512, self.z_dim),
                        )
            self.log_std = nn.Sequential(
                            nn.Linear(512, self.z_dim),
                            )
            self.fc = nn.Linear(self.z_dim * 2, self.z_dim)
        else:
            self.main = nn.Sequential(
                nn.Conv2d(self.channels + 1, self.nef * 1, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nef * 1, self.nef * 2, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nef * 2, self.nef * 4, 5, stride=2, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.nef * 4, self.nef * 8, 5, stride=2, padding=2)
            )

            self.flatten = nn.Sequential(
                nn.Linear((self.image_size // 2 ** 4) ** 2 * self.nef * 8, 512),
                nn.ReLU(inplace=True)
            )
            self.fc = nn.Linear(512, self.z_dim)


    def forward(self, inputs):

        noise = torch.randn(inputs.shape[0], self.eps_dim, device=inputs.device)
        noise = self.noise_fc(noise).view(inputs.shape[0], 1, inputs.shape[2], inputs.shape[3])
        noisy_inputs = torch.cat([inputs, noise], dim=1)
        h = self.main(noisy_inputs)
        h = h.view(h.shape[0], -1)
        h = self.flatten(h)
        h = self.fc(h)

        mean = torch.mean(h, dim=0)
        log_std = torch.log(torch.std(h, dim=0))

        try:
            kl = -log_std + ((log_std * 2).exp() + mean ** 2) / 2. - 0.5
            kl = kl
        except:
            kl = None

        return h, kl
