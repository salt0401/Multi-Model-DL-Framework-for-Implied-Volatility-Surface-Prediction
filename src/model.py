import torch
import torch.nn as nn
import torch.nn.functional as F

class BSModel(nn.Module):
    def __init__(self, atm_fun):
        super(BSModel, self).__init__()

    def forward(self, yATM):
        return yATM

class SSVIModel(nn.Module):
    def __init__(self, device='cpu', phi_fun='power_law'):
        super(SSVIModel, self).__init__()
        self.device = device
        self.phi_fun = phi_fun

        self.raw_rho = nn.Parameter(torch.Tensor([0.5]))
        if self.phi_fun == 'heston_like':
            self.raw_lambda = nn.Parameter(torch.Tensor([0]))
        elif self.phi_fun == 'power_law':
            self.raw_eta = nn.Parameter(torch.Tensor([0]))
            self.raw_gamma = nn.Parameter(torch.Tensor([0]))
        self.tanh = nn.Tanh()

    def forward(self, logm, yATM): # (invm, tau): (inverse moneyness, time to maturity)
        logm.requires_grad = True
        rho = self.tanh(self.raw_rho)

        if self.phi_fun == 'heston_like':
            lambdaa = torch.exp(self.raw_lambda)
            phi = 1 / (lambdaa * yATM) * (1 - (1 - torch.exp(-lambdaa * yATM)) / (lambdaa * yATM))
        elif self.phi_fun == 'power_law':
            eta = torch.exp(self.raw_eta)
            gamma = torch.exp(self.raw_gamma)
            phi = eta / (torch.pow(yATM, gamma) * torch.pow(1 + yATM, 1 - gamma))

        output = yATM/2 * (1 + rho*phi*logm + torch.sqrt(torch.square(phi*logm) + 2*rho*phi*logm + 1))
        
        total_output = torch.sum(output)
        grad_ttm1 = torch.zeros_like(logm)
        grad_logm1 = torch.autograd.grad(total_output, logm, retain_graph=True, create_graph=True)[0]
        total_grad_logm1 = torch.sum(grad_logm1)
        grad_logm2 = torch.autograd.grad(total_grad_logm1, logm)[0]

        return output, grad_ttm1, grad_logm1, grad_logm2

class SmileModel(nn.Module):
    def __init__(self, hidden_sizes=[5]*3, device='cpu', activation='softplus'):
        super(SmileModel, self).__init__()
        self.device = device
        self.J = hidden_sizes[0]
        self.weights_logm = nn.Parameter(nn.init.normal_(torch.empty(1, self.J, dtype=torch.float64), mean=0, std=0.01))
        self.bias_logm = nn.Parameter(nn.init.normal_(torch.empty(1, self.J, dtype=torch.float64), mean=0, std=0.01))
        self.weights_ttm = nn.Parameter(nn.init.normal_(torch.empty(1, self.J, dtype=torch.float64), mean=0, std=0.01))
        self.bias_ttm = nn.Parameter(nn.init.normal_(torch.empty(1, self.J, dtype=torch.float64), mean=0, std=0.01))
        self.weights_exp = nn.Parameter(nn.init.normal_(torch.empty(self.J, 1, dtype=torch.float64), mean=0, std=0.01))
        self.bias_exp = nn.Parameter(nn.init.normal_(torch.empty(1, self.J, dtype=torch.float64), mean=0, std=0.01))
        
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_sizes[i], hidden_sizes[i+1]) for i in range(len(hidden_sizes)-1)])
        self.output_layers = nn.Linear(hidden_sizes[-1], 1)

        self.sigmoid = nn.Sigmoid()
        
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'softplus':
            self.activation = nn.Softplus()

        self.smile_function = lambda logm : torch.sqrt(logm*torch.tanh(logm+0.5) + torch.tanh(-logm/2)+0.0005) #+0.0005 to avoid value too small

    def forward(self, ttm, logm): # (invm, tau): (inverse moneyness, time to maturity)
        ttm.requires_grad = True
        logm.requires_grad = True
        batch_size = ttm.shape[0]

        term_logm = self.smile_function(torch.tile(self.bias_logm, (batch_size, 1)) + logm @ torch.exp(self.weights_logm)) #matrix multiplication
        term_ttm = self.sigmoid(torch.tile(self.bias_ttm, (batch_size, 1)) + ttm @ torch.exp(self.weights_ttm)) #matrix multiplication
        out_hidden = (term_logm * term_ttm) @ torch.exp(self.weights_exp) + torch.tile(self.bias_exp, (batch_size, 1))
        #print(term_logm)
        #print(term_ttm)
        for hidden_layer in self.hidden_layers:
            out_hidden = self.activation(hidden_layer(out_hidden))
        
        output = self.output_layers(out_hidden)

        total_output = torch.sum(output)
        grad1 = torch.autograd.grad(total_output, (ttm, logm), retain_graph=True, create_graph=True)
        grad_ttm1 = grad1[0].clone()
        grad_logm1 = grad1[1].clone()
        total_grad_logm1 = torch.sum(grad_logm1)
        grad_logm2 = torch.autograd.grad(total_grad_logm1, logm)[0]
        return output, grad_ttm1, grad_logm1, grad_logm2

class SingleModel(nn.Module):
    def __init__(self, hidden_sizes, prior='SSVI', device='cpu'):
        super(SingleModel, self).__init__()
        self.device = device
        if prior == 'BS':
            self.Prior = BSModel()
        elif prior == 'SSVI':
            self.Prior = SSVIModel()
        self.NN = SmileModel(hidden_sizes)

    def forward(self, tau, logm, yATM):
        output_Prior, grad_ttm1_prior, grad_logm1_prior, grad_logm2_prior = self.Prior(logm, yATM)
        output_NN, grad_ttm1_NN, grad_logm1_NN, grad_logm2_NN = self.NN(tau, logm)
        output = output_Prior * output_NN

        grad_ttm1 = grad_ttm1_prior*output_NN + grad_ttm1_NN*output_Prior
        grad_logm1 = grad_logm1_prior*output_NN + grad_logm1_NN*output_Prior
        grad_logm2 = grad_logm2_prior*output_NN + grad_logm1_prior*grad_logm1_NN*2 + grad_logm2_NN*output_Prior
        return output, grad_ttm1, grad_logm1, grad_logm2
    
class SoftmaxModel(nn.Module):
    def __init__(self, ensemble_num):
        super(SoftmaxModel, self).__init__()
        self.sigmoid = nn.Sigmoid()  # sigma_2
        self.in_hid_weights = nn.Parameter(nn.init.normal_(torch.empty(2, ensemble_num, dtype=torch.float64), mean=0, std=0.01))
        self.in_hid_bias = nn.Parameter(nn.init.normal_(torch.empty(1, ensemble_num, dtype=torch.float64), mean=0, std=0.01))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, logm, ttm):
        batch_size = ttm.shape[0]
        weight_numerator = self.sigmoid(torch.hstack((logm, ttm)) @ self.in_hid_weights + torch.tile(self.in_hid_bias, (batch_size, 1)))  # (batch,2)*(2*5)=(batch,5)
        weight_denominator = self.softmax(weight_numerator)

        return weight_denominator #size = (1, ensemble num)

class MultiModel(nn.Module):
    def __init__(self, hidden_sizes=[5]*3, ensemble_num=5, device='cpu'):
        super(MultiModel, self).__init__()
        self.device = device
        self.ensemble_num = ensemble_num # the number of single models in Multi
        self.ensemble_list = nn.ModuleList()
        self.SoftmaxModel = SoftmaxModel(self.ensemble_num)

        for _ in range(self.ensemble_num):
            self.ensemble_list.append(SingleModel(hidden_sizes))

    def forward(self, ttm, logm, yATM):
        outputs = []
        grad_ttm1s = []
        grad_logm1s = []
        grad_logm2s = []
        for model in self.ensemble_list:
            output, grad_ttm1, grad_logm1, grad_logm2 = model(ttm, logm, yATM)
            outputs.append(output)
            grad_ttm1s.append(grad_ttm1)
            grad_logm1s.append(grad_logm1)
            grad_logm2s.append(grad_logm2)
        outputs = torch.cat(outputs, dim=1)
        grad_ttm1s = torch.cat(grad_ttm1s, dim=1)
        grad_logm1s = torch.cat(grad_logm1s, dim=1)
        grad_logm2s = torch.cat(grad_logm2s, dim=1)
        
        weights = self.SoftmaxModel(ttm, logm) #(batch, ensenble)
        
        output = torch.sum(outputs*weights, dim=1, keepdim=True)
        grad_ttm1 = torch.sum(grad_ttm1s*weights, dim=1, keepdim=True)
        grad_logm1 = torch.sum(grad_logm1s*weights, dim=1, keepdim=True)
        grad_logm2 = torch.sum(grad_logm2s*weights, dim=1, keepdim=True)

        return output, grad_ttm1, grad_logm1, grad_logm2
    
class SimpleLoss(nn.Module):
    def __init__(self):
        super(SimpleLoss, self).__init__()
        self.RMSELoss = RMSELoss()
        self.MAPELoss = MAPELoss()

    def forward(self, y_pred, y_true):
        return self.RMSELoss(y_pred, y_true)+self.MAPELoss(y_pred, y_true)
    
class WeightedSumLoss(nn.Module):
    def __init__(self, weights=[1,1,10,10,10,10]):
        super(WeightedSumLoss, self).__init__()
        self.RMSELoss = RMSELoss()
        self.MAPELoss = MAPELoss()
        self.CalenderLoss = Loss_calendar()
        self.ButterflyLoss = Loss_butterfly()
        self.LinearLoss = Loss_linear()
        self.UpperBoundLoss = Loss_upperbound()
        self.weights = torch.IntTensor(weights)

    def forward(self, y_pred, y_true, logm, grad_tau, grad_logm, grad_logm_2nd, y_pred_c6, logm_c6, grad_logm_c6_2nd):
        losses = torch.FloatTensor([self.RMSELoss(y_pred, y_true),
                                    self.MAPELoss(y_pred, y_true),
                                    self.CalenderLoss(y_pred, grad_tau),
                                    self.ButterflyLoss(y_pred, logm, grad_logm, grad_logm_2nd),
                                    self.LinearLoss(y_pred_c6, grad_logm_c6_2nd),
                                    self.UpperBoundLoss(y_pred_c6, logm_c6)])
        Total_loss = torch.dot(self.weights, losses)
        return Total_loss, losses
    
class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        rmse = torch.sqrt(mse)
        
        return rmse

class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        epsilon = 0.005
        absolute_percentage_error = torch.abs((y_true - y_pred) / (y_true + epsilon))
        mape = torch.mean(absolute_percentage_error)
        
        return mape
    
class Loss_calendar(nn.Module):
    def __init__(self):
        super(Loss_calendar, self).__init__()

    def forward(self, y_pred, grad_tau):
        #ttm.requires_grad = True
        #neg_dy_dt = -1*torch.autograd.grad(y_pred, ttm, create_graph=True, retain_graph=True)[0]

        loss = torch.mean(torch.relu(-grad_tau))
        return loss
    
class Loss_butterfly(nn.Module):
    def __init__(self) -> None:
        super(Loss_butterfly, self).__init__()

    def forward(self, y_pred, logm, grad_logm, grad_logm_2nd):
        #logm.requires_grad = True  # Enable gradients for input x
        #dy_dlogm = torch.autograd.grad(y_pred, logm, create_graph=True, retain_graph=True)[0]
        #d2y_dlogm2 = torch.autograd.grad(dy_dlogm, logm, create_graph=True, retain_graph=True)[0]

        g_k = (1-(logm * grad_logm)/(2*y_pred))**2 - grad_logm/4*(1/y_pred+0.25) + grad_logm_2nd/2

        loss = torch.mean(torch.relu(g_k))
        return loss
    
class Loss_linear(nn.Module):
    def __init__(self):
        super(Loss_linear, self).__init__()

    def forward(self, y_pred, logm, grad_logm_2nd):
        #logm.requires_grad = True  # Enable gradients for input x
        #dy_dlogm = torch.autograd.grad(y_pred, logm, create_graph=True, retain_graph=True)[0]
        #d2y_dlogm2 = torch.autograd.grad(dy_dlogm, logm, create_graph=True, retain_graph=True)[0]

        loss = torch.mean(grad_logm_2nd)
        return loss
    
class Loss_upperbound(nn.Module):
    def __init__(self):
        super(Loss_upperbound, self).__init__()

    def forward(self, y_pred, logm):
        diff = y_pred - torch.abs(2 * logm)

        loss = torch.mean(torch.relu(diff))
        return loss