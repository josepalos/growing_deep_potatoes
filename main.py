import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


SOBEL_X = torch.tensor([
    [-1, 0, +1],
    [-2, 0, +2],
    [-1, 0, +1]
], dtype=torch.float, device=device)
SOBEL_Y = SOBEL_X.T.to(device)  # Transpose SOBEL_X
CELL_IDENTITY = torch.tensor([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
], dtype=torch.float, device=device)


WIDTH = 32
HEIGHT = 32
CHANNELS = 16


class GrowingNet(nn.Module):
    def __init__(self):
        super(GrowingNet, self).__init__()

        # Input shape: [CHANNELS, WIDTH, HEIGHT]
        self.perception_filters = torch.stack([
            SOBEL_X,
            SOBEL_Y,
            CELL_IDENTITY
        ]).to(device)

        self.dense_128 = nn.Conv2d(CHANNELS * self.perception_filters.shape[0], out_channels=128, kernel_size=(1, 1)).cuda()
        self.dense_16 = nn.Conv2d(128, out_channels=CHANNELS, kernel_size=(1, 1)).cuda()

    def forward(self, state_grid):
        perception_result_shape = (
            state_grid.shape[0],  # Batch
            CHANNELS * len(self.perception_filters),
            WIDTH,
            HEIGHT
        )

        perception_result = torch.empty(perception_result_shape, device=device)
        for i, perception_filter in enumerate(self.perception_filters):
            start = i * CHANNELS
            end = (i + 1) * CHANNELS
            perception_result[:, start:end, :, :] = self.perceive(state_grid, perception_filter)

        dx = self.dense_128(perception_result)
        dx = self.dense_16(dx)

        x = self.stochastic_update(state_grid, dx)
        x = self.alive_masking(state_grid)
        return x

    @staticmethod
    def perceive(state_grid, kernel):
        # Pytorch conv2d expects weights like (out_channels, in_channels/groups, kernelW, kernelH)
        convolution = kernel.view(1, 1, *kernel.shape).repeat(1, CHANNELS, 1, 1)

        result = F.conv2d(state_grid, weight=convolution, padding=1).cuda()
        return result

    @staticmethod
    def stochastic_update(state_grid, ds_grid):
        rand_mask = torch.randint(0, 1, (WIDTH, HEIGHT), device=device)
        return state_grid + ds_grid * rand_mask

    @staticmethod
    def alive_masking(state_grid):
        alive = F.max_pool2d(state_grid[:3, :, :], kernel_size=(3, 3), stride=1, padding=1).cuda() > 0.1
        return state_grid * alive


def main():
    grid = torch.zeros((WIDTH, HEIGHT, CHANNELS))

    net = GrowingNet()
    net.to(device)

    random = torch.randn(1, CHANNELS, WIDTH, HEIGHT, device=device)
    net.forward(random)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


if __name__ == "__main__":
    main()