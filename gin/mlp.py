import torch

class MLP(torch.nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, layer_num=3):
        super(MLP, self).__init__()
        self.linears = torch.nn.ModuleList()
        for l in range(layer_num):
            self.linears.append(torch.nn.Linear(
                input_dim if l == 0 else hidden_dim,
                output_dim if l == layer_num - 1 else hidden_dim
            ))
        
    def forward(self, x, adj):
        h = torch.mm(adj, x)
        for l, linear in enumerate(self.linears):
            h = linear(h)
            if l != len(self.linears) - 1:
                h = torch.nn.functional.relu(h)
        return h
