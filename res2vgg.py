import torch
import torch.nn as nn
import copy

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1
    def __init__(self, in_channels=3, out_channels=20, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        #shortcut
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
    def forward(self, x, iter = None, max_iter = None):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))



class BasicBlock2(nn.Module):
    """Basic Block for resnet 18 and resnet 34

    """
    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1
    def __init__(self, in_channels=3, out_channels=20, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=1, padding=0, bias=False)
        )
        #shortcut
        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
            )
    def forward(self, x, iter = None, max_iter = None):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


a = BasicBlock()
a = a.eval()
b = copy.deepcopy(a)
m = copy.deepcopy(a)

x = torch.rand(1, 3, 32, 32)

def fuse(m):
    #m: basic block residual
    #nn.utils.fusion.fuse_conv_bn_eval
    conv1, bn1 = m.residual_function[0], m.residual_function[1]
    conv2, bn2 = m.residual_function[3], m.residual_function[4]
    convr, bnr = m.shortcut[0], m.shortcut[1]
    convbn1 = nn.utils.fusion.fuse_conv_bn_eval(conv1, bn1)
    convbn2 = nn.utils.fusion.fuse_conv_bn_eval(conv2, bn2)
    convbnr = nn.utils.fusion.fuse_conv_bn_eval(convr, bnr)
    m.residual_function[0], m.residual_function[3] = convbn1, convbn2
    del m.residual_function[1]
    del m.residual_function[3]
    #del m.shortcut[1]
    #m.shortcut[0] = convbnr
    return m

c = fuse(b)




ya = a(x)
yc = c(x)

diff = ya - yc

diff[diff!=0].min(), diff[diff!=0].max()




d = copy.deepcopy(c)
U, W, R = d.residual_function[0].weight.data + 0, d.residual_function[2].weight.data + 0, d.shortcut[0].weight.data + 0


#R = W*S, solve the s
#R^T(:, j, h, w) =  W(:, :, h, w) S(:, j, h, w)
#W(:, :, h, w)--Pinvert  R^T(:, j, h, w) = S(:, j, h, w)

def convert(U, W, R):
    S = U + 0 
    R_res = R + 0
    I, J, Hight, Width = S.shape
    for j in range(J):
        for hight in range(Hight):
            for width in range(Width):
                #print(hight, width)
                w = W[:, :, hight, width]
                r = R[:, j, hight, width]
                S[:, j, hight, width] = torch.inverse(w).mm(r.unsqueeze(1)).squeeze()
                #print('...\n')
                #print(torch.inverse(w).mm(w))
                #print(w.mm(S[:, j, hight, width].unsqueeze(1)) - r.unsqueeze(1))
                R_res[:, j, hight, width] = w.mm(S[:, j, hight, width].unsqueeze(1)).squeeze()
                #S[:, j, hight, width] = torch.pinverse(w).mm(r.unsqueeze(1)).squeeze()
    sd = BasicBlock2()
    sd.residual_function[0].weight.data = sd.residual_function[0].weight.data*0 + U + 0
    sd.residual_function[2].weight.data = sd.residual_function[2].weight.data*0 + W + 0
    sd.shortcut[0].weight.data = sd.shortcut[0].weight.data*0 + S + 0
    sd.shortcut[1].weight.data = sd.shortcut[1].weight.data*0 + W + 0
    
    test = copy.deepcopy(d)
    test.residual_function[0].weight.data = test.residual_function[0].weight.data*0 + U + 0
    test.residual_function[2].weight.data = test.residual_function[2].weight.data*0 + W + 0
    test.shortcut[0].weight.data = test.shortcut[0].weight.data*0 + R_res + 0
    return sd.eval(), test.eval()

sd, test = convert(U, W, R)

ysd = sd(x)
ytest = test(x)


diff_sd = ya - ysd
diff_sd[diff_sd!=0].max(), diff_sd[diff_sd!=0].min()


diff_test = ytest - ya
diff_test[diff_test!=0].max(), diff_test[diff_test!=0].min()


diff_sd = ytest - ysd
diff_sd[diff_sd!=0].max(), diff_sd[diff_sd!=0].min()
























