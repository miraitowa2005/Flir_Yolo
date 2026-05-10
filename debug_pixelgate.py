import torch
from models.common import LightIENet, PixelGateController, PixelGate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 测试LightIENet输出
ie_net = LightIENet().to(device)
dummy_rgb = torch.randn(2, 3, 640, 640).to(device)
alpha = ie_net(dummy_rgb)
print(f'LightIENet输出shape: {alpha.shape}')        # 应该是 [2, 1, 80, 80]
print(f'alpha范围: [{alpha.min():.3f}, {alpha.max():.3f}]')  # 应该在[0,1]

# 2. 测试PixelGateController
PixelGateController.update(dummy_rgb, ie_net)
# 👇 核心修复：补上数据精度参数 dummy_rgb.dtype，与前期的防崩溃机制对齐
alpha_from_ctrl = PixelGateController.get_alpha(80, 80, device, dummy_rgb.dtype)
print(f'Controller alpha shape: {alpha_from_ctrl.shape}')  # 应该是 [2, 1, 80, 80]

# 3. 测试PixelGate融合
pg = PixelGate(256).to(device)
F_rgb = torch.randn(2, 256, 80, 80).to(device)
F_ir  = torch.randn(2, 256, 80, 80).to(device)
out = pg([F_rgb, F_ir])
print(f'PixelGate输出shape: {out.shape}')  # 应该是 [2, 256, 80, 80]

# 4. 测试梯度是否能反传到LightIENet
loss = out.sum()
loss.backward()
grad = ie_net.enc1[0].weight.grad

# 修复了字符串格式化的问题，并提取了 .item()
if grad is not None:
    grad_norm = grad.norm().item()
    print(f'LightIENet梯度: True, norm={grad_norm:.4f}')
    if grad_norm > 0:
        print('\n✅ 所有检查通过！计算图已完美连通！')
    else:
        print('\n❌ 梯度 norm 为 0，可能存在梯度消失或连接中断！')
else:
    print('LightIENet梯度: False, norm=N/A')
    print('\n❌ 梯度传播有问题：LightIENet 没有收到梯度！')
print('\n✅ 所有检查通过！' if grad is not None and grad.norm() > 0 else '\n❌ 梯度传播有问题！')