# 4.1 VAE：变分自编码器

**变分自编码器**（Variational Autoencoder, VAE）是深度生成模型的重要里程碑。它结合了神经网络的表达能力和变分推断的理论框架，学习数据的低维隐表示。在现代视觉生成系统中，VAE 主要作为图像的编码器-解码器，将高维像素空间压缩到低维潜空间，扩散模型在此潜空间中进行生成。

## 4.1.1 自编码器回顾

### 基础自编码器

**自编码器**（Autoencoder, AE）由编码器 $f_\phi$ 和解码器 $g_\theta$ 组成：

$$\mathbf{z} = f_\phi(\mathbf{x}), \quad \hat{\mathbf{x}} = g_\theta(\mathbf{z})$$

训练目标是重建损失：

$$\mathcal{L}_{\text{AE}} = \|\mathbf{x} - \hat{\mathbf{x}}\|^2$$

编码器将高维输入 $\mathbf{x} \in \mathbb{R}^D$ 压缩为低维隐向量 $\mathbf{z} \in \mathbb{R}^d$（$d \ll D$），解码器从隐向量重建原始输入。

### 自编码器的局限

自编码器学到的隐空间是**不规则的**：

1. 隐空间可能有"空洞"——某些 $\mathbf{z}$ 不对应任何有意义的图像
2. 相邻的隐向量可能对应非常不同的图像
3. 无法从隐空间中采样生成新数据

自编码器是确定性的映射，不是生成模型。

## 4.1.2 变分推断基础

### 隐变量模型

假设数据 $\mathbf{x}$ 由隐变量 $\mathbf{z}$ 生成：

$$p(\mathbf{x}) = \int p(\mathbf{x} | \mathbf{z}) p(\mathbf{z}) d\mathbf{z}$$

目标是最大化数据的对数似然 $\log p(\mathbf{x})$。但直接计算这个积分通常是不可行的（intractable）。

### 变分下界（ELBO）

引入近似后验 $q_\phi(\mathbf{z} | \mathbf{x})$ 来近似真实后验 $p(\mathbf{z} | \mathbf{x})$。通过 Jensen 不等式：

$$\log p(\mathbf{x}) \geq \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x} | \mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

右边称为**证据下界**（Evidence Lower Bound, ELBO）或**变分下界**。

### ELBO 的分解

ELBO 可以分解为两项：

$$\mathcal{L}_{\text{ELBO}} = \underbrace{\mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x} | \mathbf{z})]}_{\text{重建项}} - \underbrace{D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))}_{\text{正则项}}$$

**重建项**：希望从隐变量能准确重建原始数据。

**正则项（KL 散度）**：希望近似后验 $q_\phi$ 接近先验 $p(\mathbf{z})$，使隐空间结构化。

## 4.1.3 VAE 的实现

### 网络结构

**编码器**：输出近似后验的参数（均值和方差）

$$q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\mathbf{z}; \boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$$

**解码器**：从隐变量生成数据

$$p_\theta(\mathbf{x}|\mathbf{z}) = \mathcal{N}(\mathbf{x}; \boldsymbol{\mu}_\theta(\mathbf{z}), \sigma^2 \mathbf{I})$$

或者对于图像，直接输出像素值，用 MSE 作为重建损失。

### 重参数化技巧

从 $q_\phi(\mathbf{z}|\mathbf{x})$ 采样 $\mathbf{z}$ 的操作不可微。**重参数化技巧**（Reparameterization Trick）将随机性从参数中分离：

$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

现在 $\mathbf{z}$ 是 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 的确定性函数加上外部噪声，梯度可以通过 $\boldsymbol{\mu}$ 和 $\boldsymbol{\sigma}$ 反向传播。

### KL 散度的解析解

当先验是标准正态分布 $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$，近似后验是对角高斯时，KL 散度有解析解：

$$D_{\text{KL}}(q_\phi \| p) = -\frac{1}{2} \sum_{j=1}^d \left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

### 完整损失函数

$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})}\left[\|\mathbf{x} - g_\theta(\boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\epsilon})\|^2\right] + \beta \cdot D_{\text{KL}}$$

$\beta$ 是平衡重建和正则化的超参数（$\beta$-VAE）。

## 4.1.4 VAE 变种

### β-VAE

$\beta$-VAE 通过调节 KL 项的权重控制隐空间的"解耦"程度：

$$\mathcal{L}_{\beta\text{-VAE}} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}$$

- $\beta > 1$：更强的正则化，隐空间更解耦，但重建质量下降
- $\beta < 1$：更好的重建，但隐空间可能不够规整

### VQ-VAE

**向量量化 VAE**（VQ-VAE）用离散的 codebook 替代连续的高斯隐空间。

编码器输出连续向量，通过**最近邻查找**量化到 codebook 中的向量：

$$\mathbf{z}_q = \text{Quantize}(\mathbf{z}_e) = \arg\min_{\mathbf{e}_k \in \mathcal{C}} \|\mathbf{z}_e - \mathbf{e}_k\|$$

训练时使用 **Straight-Through Estimator**：前向用量化后的向量，反向梯度直接传给编码器输出。

VQ-VAE 的离散隐空间更适合与自回归模型结合，是 DALL-E（第一版）的核心组件。

### VAE-GAN

VAE 生成的图像往往模糊，因为 MSE 损失倾向于产生平均化的结果。VAE-GAN 引入判别器提供对抗损失，改善生成质量。

## 4.1.5 Latent Diffusion 中的 VAE

### 为什么需要潜空间

直接在像素空间做扩散计算量巨大。$512 \times 512 \times 3$ 的图像有约 80 万维。

**潜空间扩散**（Latent Diffusion）先用 VAE 将图像压缩到潜空间，再在潜空间做扩散：

$$\mathbf{x} \xrightarrow{\text{Encoder}} \mathbf{z} \xrightarrow{\text{Diffusion}} \tilde{\mathbf{z}} \xrightarrow{\text{Decoder}} \tilde{\mathbf{x}}$$

### Stable Diffusion 的 VAE

Stable Diffusion 使用的 VAE 将 $512 \times 512 \times 3$ 图像编码为 $64 \times 64 \times 4$ 的潜向量，压缩比为 48 倍（空间 8 倍，通道 3→4）。

这个 VAE 是独立预训练的，使用大规模图像数据。编码器和解码器都是卷积网络，包含残差块和注意力层。

### KL 正则 vs VQ 正则

Stable Diffusion 的 VAE 使用 KL 正则化（连续潜空间）。也有工作使用 VQ-VAE（离散潜空间）。

连续潜空间更适合扩散模型，因为扩散过程假设数据是连续的。

## 4.1.6 VAE 的数学深入

### 后验坍缩

VAE 训练中常见的问题是**后验坍缩**（Posterior Collapse）：KL 项被优化到接近零，编码器输出接近先验，解码器忽略隐变量。

原因：强大的解码器可以在忽略 $\mathbf{z}$ 的情况下生成数据，而 KL 项鼓励 $q_\phi$ 接近先验。

解决方案：
- KL annealing：训练初期降低 KL 权重
- Free bits：每维 KL 低于阈值时不惩罚
- 更强的编码器、更弱的解码器

### 信息瓶颈视角

VAE 可以从**信息瓶颈**（Information Bottleneck）角度理解：

$$\max I(\mathbf{z}; \mathbf{x}) - \beta I(\mathbf{z}; \mathbf{x})$$

隐变量 $\mathbf{z}$ 应该保留关于 $\mathbf{x}$ 的"必要"信息，同时压缩"冗余"信息。

### 与 EM 算法的联系

VAE 的训练可以视为随机版本的 EM 算法：

- E 步：用编码器近似后验 $q_\phi(\mathbf{z}|\mathbf{x})$
- M 步：用解码器优化似然 $p_\theta(\mathbf{x}|\mathbf{z})$

两步同时用梯度下降优化。

## 4.1.7 实现示例

```python
class VAE(nn.Module):
    def __init__(self, in_channels=3, latent_dim=256):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_channels, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z).view(-1, 256, 8, 8)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss
```
