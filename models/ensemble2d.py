import torch
import torch.nn as nn

# class Ensemble2D(nn.Module):
#     def __init__(self, models, weights=None):
#         super(Ensemble2D, self).__init__()
#         self.models = nn.ModuleList(models)
#         self.weights = weights or [1.0] * len(models)

#     def forward(self, x):
#         outputs = [w * m(x) for m, w in zip(self.models, self.weights)]
#         return torch.stack(outputs).sum(dim=0) / sum(self.weights)

class Ensemble2D(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.meta = nn.Sequential(
            nn.Linear(len(models), 1)
        )

    def forward(self, x):
        
        outputs = [torch.sigmoid(m(x)) for m in self.models]
        logits = torch.cat(outputs, dim=1)  # shape: [batch_size, num_models]
        return self.meta(logits)

