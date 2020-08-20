import torch
import numpy as np
import math


class Baka(torch.nn.Module):
    weight: torch.Tensor

    def __init__(self, in_features, out_features, bias=True):
        super(Baka, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = torch.Tensor(out_features, in_features, in_features + 1)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
        self.weight[:, :, 1:] = self.weight[:, :, 1:] * 0 + 1
        self.weight = torch.nn.Parameter(self.weight)

    def forward(self, input):
        coeff = self.weight[:, :, 0]
        powers = self.weight[:, :, 1:]

        intermediate_embeddings = []
        for o in range(self.out_features):
            l_powers = powers[o]
            out_embedding = []
            for i, sample in enumerate(input):
                if len(sample.shape)>1:
                    print('test')
                powered_feature = sample ** l_powers
                powered_feature = torch.prod(powered_feature, axis=1) # [8, 27]
                out_embedding.append(powered_feature)
            out_embedding = torch.stack(out_embedding)
            # Multiply the weights
            out_embedding = torch.matmul(coeff[o].reshape(1, self.in_features), out_embedding.transpose(0, 1))
            intermediate_embeddings.append(out_embedding[0])

        retval =  torch.stack(intermediate_embeddings).transpose(0, 1)
        return retval

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == "__main__":
    x1 = np.absolute(np.random.randn(2000))
    x2 = np.absolute(np.random.randn(2000))
    y = 5 * x1**2 + 3 * x1**3 * x2 **3
    x = np.stack([x1, x2]).transpose()


    # Create a FCN to evaluate the results
    fcn = torch.nn.Sequential(
        torch.nn.Linear(2, 20),
        torch.nn.Linear(20,20),
        torch.nn.Linear(20,1)
    )

    loss_func = torch.nn.MSELoss(reduction='sum')
    optimizer = torch.optim.Adam(fcn.parameters())

    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).float().reshape((2000, 1))
    for t in range(500):
        y_pred = fcn(x)

        loss = loss_func(y_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


    # Create a Baka FCN
    bcn = torch.nn.Sequential(
        Baka(2,20),
        Baka(20,1)
    )

    baka_loss_fn = torch.nn.MSELoss(reduction='sum')
    baka_optim = torch.optim.RMSprop(bcn.parameters())
    torch.nn.utils.clip_grad_norm_(bcn.parameters(), 0)
    torch.nn.utils.clip_grad_value_(bcn.parameters(), 2)
    for t in range(500):
        by_pred=  bcn(x.float())
        loss = baka_loss_fn(by_pred, y)
        if t % 100 == 99:
            print(t, loss.item())

        baka_optim.zero_grad()
        loss.backward()
        baka_optim.step()

#     x = torch.from_numpy(np.array([[1,2,3], [2,4,6]]))
#     baka_layer = Baka(3, 2)
#     y = baka_layer(x.float())
#     print(y)
#     pass
#