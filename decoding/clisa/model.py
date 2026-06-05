"""CLISA model definitions (from Chen et al. 2023, unchanged)."""
import torch.nn as nn
import torch.nn.functional as F
import torch


def stratified_norm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_str[n_samples*i: n_samples*(i+1), :] = (out[n_samples*i: n_samples*(i+1), :] - out[n_samples*i: n_samples*(i+1), :].mean(
            dim=0)) / (out[n_samples*i: n_samples*(i+1), :].std(dim=0) + 1e-3)
    return out_str


def stratified_layerNorm(out, n_samples):
    n_subs = int(out.shape[0] / n_samples)
    out_str = out.clone()
    for i in range(n_subs):
        out_oneSub = out[n_samples*i: n_samples*(i+1)]
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0], -1, out_oneSub.shape[-1]).permute(0,2,1)
        out_oneSub = out_oneSub.reshape(out_oneSub.shape[0]*out_oneSub.shape[1], -1)
        out_oneSub_str = (out_oneSub - out_oneSub.mean(dim=0)) / (out_oneSub.std(dim=0) + 1e-3)
        out_str[n_samples*i: n_samples*(i+1)] = out_oneSub_str.reshape(n_samples, -1, out_oneSub_str.shape[1]).permute(
            0,2,1).reshape(n_samples, out.shape[1], out.shape[2], -1)
    return out_str


class ConvNet_baseNonlinearHead(nn.Module):
    def __init__(self, n_spatialFilters, n_timeFilters, timeFilterLen, n_channs, stratified, multiFact, isMaxPool, args):
        super(ConvNet_baseNonlinearHead, self).__init__()
        self.spatialConv = nn.Conv2d(1, n_spatialFilters, (n_channs, 1))
        self.timeConv = nn.Conv2d(1, n_timeFilters, (1, timeFilterLen), padding=(0, (timeFilterLen-1)//2))
        self.avgpool = nn.AvgPool2d((1, 30))
        self.spatialConv2 = nn.Conv2d(n_timeFilters, n_timeFilters*multiFact, (n_spatialFilters, 1), groups=n_timeFilters)
        self.timeConv2 = nn.Conv2d(n_timeFilters*multiFact, n_timeFilters*multiFact*multiFact, (1, 6), groups=n_timeFilters*multiFact)
        self.n_spatialFilters = n_spatialFilters
        self.n_timeFilters = n_timeFilters
        self.stratified = stratified
        self.isMaxPool = isMaxPool
        self.args = args

    def forward(self, input):
        if 'initial' in self.stratified:
            input = stratified_layerNorm(input, int(input.shape[0]/2))

        out = self.spatialConv(input)
        out = out.permute(0,2,1,3)
        out = self.timeConv(out)
        out = F.elu(out)
        out = self.avgpool(out)

        if 'middle1' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))

        out = F.elu(self.spatialConv2(out))
        out = F.elu(self.timeConv2(out))

        if 'middle2' in self.stratified:
            out = stratified_layerNorm(out, int(out.shape[0]/2))

        if self.isMaxPool:
            _, indices = torch.topk(out.mean(dim=3), out.shape[1]//2, dim=1)
            out_pooled = torch.zeros((out.shape[0], out.shape[1]//2, out.shape[2], out.shape[3])).to(self.args.device)
            for i in range(out.shape[0]):
                out_pooled[i,:,:,:] = out[i,indices[i,:,0]]
            out_pooled = out_pooled.reshape(out_pooled.shape[0], -1)
            return out_pooled, indices
        else:
            out = out.reshape(out.shape[0], -1)
            return out


class simpleNN3(nn.Module):
    def __init__(self, inp_dim, hidden_dim, out_dim, n_samples, stratified):
        super(simpleNN3, self).__init__()
        self.fc1 = nn.Linear(inp_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, out_dim)
        self.n_samples = n_samples
        self.stratified = stratified

    def forward(self, input):
        if self.stratified:
            input = stratified_norm(input, self.n_samples)
        out = F.relu(self.fc1(input))
        out = F.relu(self.fc2(out))
        if self.stratified:
            out = stratified_norm(out, self.n_samples)
        out = self.fc3(out)
        return out
