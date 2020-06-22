import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms
import tqdm

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
BATCH_SIZE = 8
STEPS_TO_SHOW = 500
STEPS_TO_LOG = 100


class GrowingNet(nn.Module):
    def __init__(self):
        super(GrowingNet, self).__init__()
        self.perception_filters = torch.stack([
            SOBEL_X.view(1, 1, *SOBEL_X.shape).repeat(CHANNELS, 1, 1, 1),
            SOBEL_Y.view(1, 1, *SOBEL_Y.shape).repeat(CHANNELS, 1, 1, 1),
            CELL_IDENTITY.view(1, 1, *CELL_IDENTITY.shape).repeat(CHANNELS, 1, 1, 1)
        ]).to(device)

        self.dense_128 = nn.Conv2d(CHANNELS * self.perception_filters.shape[0],
                                   out_channels=128,
                                   kernel_size=1)
        self.dense_16 = nn.Conv2d(128,
                                  out_channels=CHANNELS,
                                  kernel_size=1)
        nn.init.zeros_(self.dense_16.weight)
        nn.init.zeros_(self.dense_16.bias)

    def _squash_perception(self, x):
        x = self.dense_128(x)
        x = F.relu(x)
        x = self.dense_16(x)
        return x

    def forward(self, state_grid):
        perception = self.perceive(state_grid)
        dx = self._squash_perception(perception)

        x = self.stochastic_update(state_grid, dx)

        alive_mask = self.get_alive_mask(state_grid) & self.get_alive_mask(x)
        alive_mask = alive_mask.type_as(state_grid)\
            .repeat(1, CHANNELS, 1, 1)

        x *= alive_mask
        return torch.clamp(x, 0.0, 1.0)

    def perceive(self, state_grid):
        perception_result_shape = (
            state_grid.shape[0],  # Batch
            CHANNELS * self.perception_filters.shape[0],
            WIDTH,
            HEIGHT
        )
        perception_result = torch.empty(perception_result_shape, device=device)
        for i, perception_filter in enumerate(self.perception_filters):
            start = i * CHANNELS
            end = (i + 1) * CHANNELS

            convolution = perception_filter
            perception = F.conv2d(state_grid, weight=convolution, padding=1, groups=CHANNELS)
            perception_result[:, start:end, :, :] = perception

        return perception_result

    @staticmethod
    def stochastic_update(state_grid, ds_grid):
        random_shape = list(state_grid.shape)
        random_shape[-3] = 1  # channels
        rand_mask = torch.randint(0, 1+1,  # [low, high)
                                  random_shape, device=device)\
            .repeat(1, CHANNELS, 1, 1)  # same mask for all channels
        return state_grid + ds_grid * rand_mask

    @staticmethod
    def get_alive_mask(state_grid):
        alpha = state_grid[:, 3:4, :, :]
        alive = F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > 0.1
        return alive


def get_initial_grid():
    grid = torch.zeros((CHANNELS, WIDTH, HEIGHT), device=device)
    grid[3:, WIDTH // 2, HEIGHT // 2] = 1.0
    return grid


def load_image_as_tensor(path):
    image = PIL.Image.open(path).convert("RGBA")
    image.thumbnail((WIDTH, HEIGHT))
    transform = torchvision.transforms.ToTensor()
    return transform(image).to(device=device)


def main():
    tensor2pil = torchvision.transforms.ToPILImage()
    target = load_image_as_tensor("sprites/potato32x32_transparent.png")
    target_batch = target.repeat(BATCH_SIZE, 1, 1, 1)  # same target for the entire batch

    net = GrowingNet()
    net.to(device)

    loss_f = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=2e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)

    for p in net.parameters():
        p.register_hook(lambda grad: grad / (torch.norm(grad, 2) + 1e-8))

    plt.ion()

    running_loss = 0.
    grid = None
    for step in tqdm.trange(10001):
        grid = get_initial_grid().repeat(BATCH_SIZE, 1, 1, 1)

        iter_n = torch.randint(64, 96, [1]).item()

        for i in range(iter_n):
            grid = net.forward(grid)

            if (step + 1) % STEPS_TO_SHOW == 0:
                img = tensor2pil(grid[0, :4, :, :].cpu())
                plt.clf()
                plt.imshow(np.array(img))
                plt.title(f"Step {step}, iteration {i}")
                plt.draw()
                plt.pause(0.000001)

        optimizer.zero_grad()
        loss = loss_f(grid[:, :4, :, :], target_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if (step+1) % STEPS_TO_LOG == 0:
            print(f'\n\tStep {step + 1:5d} mean loss: {running_loss / STEPS_TO_SHOW :.5f}')
            running_loss = 0.0

    print(f'\n\tEnd mean loss: {running_loss / STEPS_TO_SHOW :.5f}')

    iter_n = torch.randint(64, 96, [1]).item()

    for i in range(iter_n):
        grid = net.forward(grid)
        img = tensor2pil(grid[0, :4, :, :].cpu())
        plt.clf()
        plt.imshow(np.array(img))
        plt.title(f"Last potato, iteration {i}")
        plt.draw()
        plt.pause(0.0001)

    plt.waitforbuttonpress()
    plt.ioff()


if __name__ == "__main__":
    main()