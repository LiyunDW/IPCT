import torch.nn as nn
import torch, numbers
import torch.nn.functional as F
from torch.nn import init
from einops import rearrange


# Gate Feed-forward Network
class GFN(nn.Module):
    def __init__(self, dim):
        super(GFN, self).__init__()
        self.LN = LayerNorm(dim)
        self.PW = nn.Conv2d(dim, dim * 4, 1, 1)
        self.projection1 = nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, groups=dim * 4)
        self.act = nn.Sigmoid()
        self.projection2 = nn.Conv2d(dim * 4, dim * 4, 3, 1, 1, groups=dim * 4)
        self.projection_out = nn.Conv2d(dim * 4, dim, 1, 1)

    def forward(self, x):
        lnx = self.PW(self.LN(x))
        projection1 = self.act(self.projection1(lnx))
        projection2 = self.projection2(lnx)
        out = self.projection_out(torch.mul(projection1, projection2)) + x

        return out


class EfficientAttention(nn.Module):

    def __init__(self, dim, head_count):
        super(EfficientAttention, self).__init__()
        self.channels = dim
        self.key_channels = dim
        self.value_channels = dim
        self.head_count = head_count

        self.normQ = LayerNorm(dim)
        self.normK = LayerNorm(dim)
        self.normV = LayerNorm(dim)

        self.keys = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.queries = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.values = nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim))
        self.reprojection = nn.Conv2d(dim, dim, 1, 1)

    def forward(self, K, Q, V):
        n, _, h, w = K.size()
        keys = self.keys(self.normK(K)).reshape((n, self.key_channels, h * w))
        queries = self.queries(self.normQ(Q)).reshape(n, self.key_channels, h * w)
        values = self.values(self.normV(V)).reshape((n, self.value_channels, h * w))
        head_key_channels = self.key_channels // self.head_count
        head_value_channels = self.value_channels // self.head_count
        attended_values = []

        for i in range(self.head_count):
            key = F.softmax(keys[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=2)
            query = F.softmax(queries[:, i * head_key_channels: (i + 1) * head_key_channels, :], dim=1)
            value = values[:, i * head_value_channels: (i + 1) * head_value_channels, :]
            context = torch.matmul(key, value.transpose(1, 2))
            attended_value = torch.matmul(context.transpose(1, 2), query).reshape(n, head_value_channels, h, w)
            attended_values.append(attended_value)

        aggregated_values = torch.cat(attended_values, dim=1)
        reprojected_value = self.reprojection(aggregated_values)
        attention = reprojected_value + V

        return attention


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()
        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.body(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x


class DESA(nn.Module):
    def __init__(self, dim):
        super(DESA, self).__init__()
        self.EA = EfficientAttention(dim * 2, 4)

        self.Dconv = nn.Conv2d(dim, dim * 2, 2, 2)
        self.Uconv = nn.ConvTranspose2d(dim * 2, dim, 2, 2)

        self.alpha = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.mp = nn.MaxPool2d(2, 2)
        self.ap = nn.AvgPool2d(2, 2)

        self.balance = nn.Sequential(
            LayerNorm(dim),
            nn.Conv2d(dim, dim * 2, 3, 1, 1),
            nn.Conv2d(dim * 2, dim * 2, 1),
            nn.GELU())

    def forward(self, x):
        # DE reduce 50% data of x and improve efficiency while maintain quality
        x_input = x.clone()
        x = self.Dconv(x)
        balance = self.balance(self.alpha * self.mp(x_input) + self.beta * self.ap(x_input))
        x = x + balance

        # SA model the inter-channel relationship
        x = self.EA(x, x, x)

        x = self.Uconv(x)

        return x


class Phase(nn.Module):
    def __init__(self, dim):
        super(Phase, self).__init__()
        self.CGA = DESA(dim)
        self.merge = nn.Conv2d(dim + 3, dim, 3, 1, 1)
        self.G = nn.GELU()

        self.PW1 = nn.Conv2d(dim, dim, 1, 1)
        self.PW2 = nn.Conv2d(dim, dim, 1, 1)
        self.xout = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, 1, 1),
            nn.Conv2d(dim // 2, 1, 1, 1))  # To ID

        self.z_pre = nn.Conv2d(dim, dim // 2, 3, 1, 1)
        self.z_cur = nn.Conv2d(dim, dim // 2, 3, 1, 1)

        self.bfn_first = GFN(dim)
        self.bfn_last = GFN(dim)

    def forward(self, x, z, y, Phi, PhiT):
        # PLPM Generate Pk for DE-SA
        x_in, z_in = x.clone(), z.clone()
        Fx = torch.cat([x, z, PhiT(Phi(x)), PhiT(y)], dim=1)
        P_k = self.G(self.merge(Fx))

        # DE-SA for X update
        x = self.CGA(P_k)
        x = self.PW1(x) + P_k
        x_ID = self.xout(x) + x_in

        # PGFN for Z update
        # GFN
        x = self.PW2(x)
        z = self.bfn_first(x)
        # IGFN
        z = torch.cat([self.z_pre(z_in), self.z_cur(z)], dim=1)
        z_FD = self.bfn_last(z) + x

        return x_ID, z_FD


class IPCT(torch.nn.Module):
    def __init__(self, layer, cs_ratio, dim):
        super(IPCT, self).__init__()
        Net = []
        self.layer = layer
        self.patch_size = 32
        self.N = int(cs_ratio * self.patch_size * self.patch_size)
        self.Phiweight = nn.Parameter(init.xavier_normal_(
            torch.Tensor(self.N, 1, self.patch_size, self.patch_size)))
        self.Phi = lambda w: F.conv2d(w, self.Phiweight.to(w.device), stride=self.patch_size)
        self.PhiT = lambda w: F.conv_transpose2d(w, self.Phiweight.to(w.device), stride=self.patch_size)
        self.z = nn.Conv2d(1, dim, 3, 1, 1)
        for i in range(layer): Net.append(Phase(dim))
        self.net = nn.ModuleList(Net)

    def forward(self, x):
        Phi, PhiT = self.Phi, self.PhiT
        y = Phi(x)
        x = PhiT(y)
        z = self.z(x)
        for i in range(self.layer):
            x, z = self.net[i](x, z, y, Phi, PhiT)
        return x


if __name__ == '__main__':
    model = IPCT(16, 0.1, 32)
    para = sum(p.numel() for p in model.parameters())
    phi = model.Phiweight.numel()
    print("total para num: %d" % para)
    print("Phi Weight num: %d" % phi)
    print("Net Weight num: %d" % (para - phi))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    input_tensor = torch.ones(1, 1, 256, 256).to(device)
    output_tensor = model(input_tensor)
    print(output_tensor.shape)

    from thop import profile, clever_format

    macs, params = profile(model, inputs=(input_tensor,))
    FLOPs, params = clever_format([macs * 2, params], "%.3f")
    print(f"FLOPs: {FLOPs}, Params: {params}")
