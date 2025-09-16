# 图像理解与生成统一模型——前沿模型架构理解

生成式多模态模型近年来一直是业界的研究热点。视觉语言模型（VLM）一直是多模态文本生成领域的核心路线，能够完成图像理解任务；扩散模型（Diffusion Model）则一直是图像和视频生成领域的核心方法。今年早期，同时支持图像理解和生成的统一模型如雨后春笋般浮现。统一模型受到青睐，不只是因为它同时支持理解和生成两种任务带来的通用性，更是因为大家看到了任务有机结合带来的多模态学习潜力。一方面，两种任务有机结合使模型能在两种任务上联合优化，提升了图文交错数据的利用率，同时也让学术界看到了任务间互相促进的潜力。另一方面，输出支持多模态，让统一模型在当下火热的模型推理上有了更多的玩法，比如可以开发出基于生成图的推理和基于推理的图生成。

从理解和生成任务出发，在统一模型之前，理解任务由 Vision Language Model 完成，走自回归 AR 路线；而生成任务则由 Diffusion Model 完成，走 DDPM 或者 FlowMatching 的路线。因此，统一模型的技术也是围绕这两个路线出发的。本文将目前典型的统一模型工作分成四类，分别是纯自回归路线，AR + Diffusion 串联结构，AR + Diffusion 并联结构和单一模型同时做 AR + Diffusion。由于 Diffusion 做文本理解的技术和生态尚不成熟，本文不涉及仅基于 Diffusion 的统一模型。

注意，本文并不是一篇完整的综述，可能不会涵盖所有文章，这四个类别是为了更好的抓住不同模型之间的差异。理解一个统一模型最快的思路就是理清它是怎么应对图像理解和生成这两个任务的。对于图像理解任务，现在的统一模型一般都是沿用 VLM 的思路，将图像编码成 Embedding 后，由 LLM 统一处理，这个路线是经过广泛验证的；**不同路线方法在理解任务上的差别在于图像编码的方式**。对图像生成任务的实现方式，不同模型的差异主要体现在几个方面，**分别是是图像如何编码、如何生成图像编码、图像如何解码。**以下每个路线我们都会重点关注这些差异点。

##  纯自回归路线的统一模型

自回归即根据输入序列预测下一个 Token，并将预测的 Token 送回输入进行递归预测。纯自回归路线的统一模型可以看作是 LLM 的文本 Token 预测与 VQGAN\[1\] 这个工作中的图像 Token 预测的结合。典型的工作包括 LWM\[2\], Chameleon\[3\], Emu3\[4\], Janus\[5\] 和 Janus-Pro\[6\] 等。LWM 和 Chameleon 是相对较早的文本和图像统一训练的工作，Emu3 则是进一步扩展了视频生成的模态，Janus 和 Janus-Pro 则是将图像理解的编码独立开来。在这里我们以 Chameleon 和 Janus来分析这个路线的模型。

Chameleon 的模型架构如下图所示。对于图像理解任务，先使用 VQ-VAE 的 Encoder 作为 Image Tokenizer，将图像编码成离散的 Embedding，然后由自回归模型预测文本输出。对于图像生成任务，使用 VQ-VAE 的 Decoder 作为 Image De-Tokenizer，将自回归模型预测的离散图像 Token 解码成图像。

![截屏2025-07-31 11.04.43.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/409da4d8.png)

在这个架构基础上，Janus 团队则认为 VQ-VAE 的 Encoder 是通过重建任务训练得到，并不适合于语义空间的图像理解任务，因此，将图像理解任务的 Image-Tokenizer 修改为使用图文对预训练的 SigLIP。架构如下图所示：Und. Encoder（理解部分） 采用 SigLIP，Gen.Encoder（生成部分） 和 Image Decoder则采用 VQVAE。

![image.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/164d2b24.png)


简单来看，纯自回归路线的统一模型就是将 VQ-VAE 的离散图像 Token 也纳入 LLM 或 VLM的统一训练中去，它的优点是离散图像 Token 的预测任务与 LLM 的预训练范式高度吻合，也非常符合 AR 模型的特性。但从图像质量来看，这个路线的模型生成的图像质量不尽人意，一方面是由于图像编码空间的离散化带来的效果损失，另一方面则是由于自回归模型无法像扩散模型一样做分布建模。此外，由于无法引入随机噪声，生成图像的多样性差也是这一路线的一大挑战。
##  AR + Diffusion 串联结构的统一模型

 AR 和 Diffusion 分别是图像理解和生成领域的主流路线，因此可能是最简单有效的统一模型路线就是将 AR 和 Diffusion 串联起来，AR 模型完成理解任务，AR 模型的输入作为 Diffusion 的条件，完成生成任务，架构如下图所示（图源：统一模型综述 \[7\]）。

![截屏2025-08-22 15.25.01.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/244534b3.png)

对于理解任务，图像由语义编码器（一般是 CLIP、SigLIP 或者经过图文对齐训练的 ViT）编码成连续的 Embedding。对于生成（文生图）任务，则由模型处理文本输入后，输出一个中间 Embedding，作为 Diffusion 模型生成图像的条件。在这里，生成任务的图像编码就是 AR 和 Diffusion 模型的中间 Embedding，由 AR 模型直接产生。在这里，我们根据是否显式地监督中间 Embedding，将几个典型的工作分成两类。

### 2.1 使用语义 Embedding 监督的统一建模方法

顾名思义，这类方法是使用损失函数直接监督 AR 模型的输出图像 Embedding，使其有一个明确的 Embedding 输出目标，同时使用这些 Embedding 来训练 Diffusion 模型进行图像重建。典型的方法包括 MetaMorph\[8\], Nexus-Gen\[9\] 和 Blip-3o\[10\]。其典型架构如下图所示：

![截屏2025-08-22 15.53.03.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/de912c54.png)

如上图b所示，MetaMorph 和 Nexus-Gen 会使用 Image Loss（一般为 MSE 或者余弦相似度损失） 来使监督 AR 模型，使其学会预测目标图像的语义 Embedding。这样做的原因一方面是**想要从 Joint Training 的角度回答为什么要做统一模型**，在 MetaMorph 的实验中，图像理解和生成任务一起从零开始训练能互相促进彼此的效果。另一方面，Nexus-Gen 的 Unified Image Embedding Space 将理解和生成建模成了一个逆向任务，潜在的好处是可以直接对生成的 Embedding 做理解，从而有多轮推理的潜力。此外，在这一个子路线还存在一个问题，即监督自回归模型来预测连续的图像 Embedding 会导致严重的误差累计问题，MetaMorph 忽略了这一现象，而 Nexus-Gen 则采用了预填充自回归的策略来解决，这个策略本质上与其他工作 (Blip-3o, MetaQuery\[11\]) 的 Learnable Query 是一致的。

Blip-3o 也是类似 Embedding 监督训练思路，但他们额外使用 FlowMatching 来对语义 Embedding 做分布建模， 算是在这个架构中分析了自回归模型中无法做分布建模的问题。如下图所示：

![截屏2025-08-22 16.09.16.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/a40097db.png)

### 2.2 直接训练 Diffusion 模型的方法


这类方法一般将 AR 模型冻住，直接使用 AR 模型输出的 hidden states 作为Diffusion 模型的条件，只训练 Diffusion 做图像生成。换一个角度，可以把这个路线的方法看成 Diffusion 技术的演进，即将 Diffusion 模型常用的 T5 Text Encoder 换成了 一个更大的多模态生成式模型（例如Qwen2.5-VL-7B ）。典型的方法包括 Uniworld\[12\], MetaQuery (特殊说明，MetaQuery 使用的条件提取方式是 Learnable Query，不是 hidden states), Qwen-Image\[13\] 和 OmniGen2\[14\]。在图像生成任务上，其典型架构如下图所示（图源：Qwen-Image），文本 Prompt 输入 Qwen2.5-Vl，直接输出这些 Token 对应的 hidden states，作为后续 Diffusion Transformer 的文本条件。
![截屏2025-08-22 16.26.38.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/1067ae96.png)

除了图像生成外，统一模型一个潜力在于其图像编辑能力，因此，我们在这里额外分析一下这几个模型的图像编辑架构。相比于图像生成任务，使用 Diffusion 模型进行图像编辑时有一个额外的输入条件，即待编辑图像的编码信息。 待编辑图像可以使用两种编码，第一种是语义编码，编码器采用SigLIP等语义编码器。第二种是重建编码，编码器采用VAE。

1.  语义编码架构：以Uniworld为例，采用语义编码的架构如下图所示，着重关注SigLIP部分，如图所示，待编辑图像直接通过SigLIP 和 MLP 之后，作为一个条件输入到 DiT 中。Nexus-Gen也支持图像编辑，采用的条件注入架构也是这种语义编码特征。从 Nexus-Gen 在图像编辑上的实验经验来看，语义特征编码相比于 VAE 编码，编码空间更接近VLM 的输出语义空间，仅需要少量数据训练就可以建立起文本条件和图像条件之间的关系，潜在的优势是更好的指令遵循能力。 这种编码架构的劣势在于，语义编码存在信息损失，重建效果和编码 Token 数量强相关，往往不能做到一对一重建。从图像编辑的重建效果来看，GPT4o-Image 很可能也是采用语义编码。
    

![截屏2025-08-22 16.37.21.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/f697f06f.png)

2.  VAE 编码架构：Qwen-Image 和 OmniGen2 采用的是这种架构，目光放远，早些时候开源的 Step1X-Edit 与 Flux-Kontext 都是一模一样的架构。再把目光放远，这种架构和 In-Context LoRA 与 OmniControl 的思路是一致的。以Qwen-Image 为例，架构如下图所示。重点关注 Input Image, 它经过 VAE Encoder 后，作为条件输入 DiT中。在这个架构中，一般会使用位置编码来区分输入图像和去噪图像，像 Qwen-Image\[15\] 和 Flux-Kontext\[16\] 都是直接在位置编码的第一维（帧id）做区分。
    

![截屏2025-08-22 16.54.17.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/fe100074.png)

## AR + Diffusion 并联结构

以上的串联结构是利用 Embedding 作为 AR 和 Diffusion 的桥梁，而这里的串联结构，则是将 Attention 作为 AR 模型和 Diffusion 模型的桥梁。典型的工作包括 LlamaFusion\[17\] 和 Bagel\[18\]。采用 Bagel 中的定义，可以将这个架构叫做 Mixture-of-Transformer-Experts (MoT)。

### 3.1 LlamaFusion 冻结文本模型


LlamaFusion 的架构如下图所示。给定一个语言模型，如下左图，作者将其参数复制一份，作为右图的图像生成专用参数。对于一个包含文本和噪声图像的序列，文本 Token 使用左图的参数计算，而图像 Token 使用右图的参数训练，但是在 Attention 的计算阶段，所有 Token 拼接到一起做 Self-Attention。由于语言模型冻住，这个架构并不能改变模型的理解能力，所以不涉及图像理解任务中的编码问题。而图像生成使用的编码和解码都是 VAE，尽管模型采用的是语言模型结构，但实际是使用 Diffusion 的路线进行图像生成。
![截屏2025-08-22 17.16.34.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/b0aebf44.png)

### 3.2 Bagel 图文混合训练

Bagel 采用和 LlamaFusion 相似的架构，不同的是，模型的图像理解和生成能力全部都是重头开始训练的。_模型的图像理解采用的编码器是_ SigLIP 这类语义编码的模型，而图像生成采用的编码器和解码器是 VAE 这样的重建模型。理解任务使用 AR 的方式自回归地生成文本 Token，而生成任务则采用 Diffusion 的方式生成图像的 VAE 特征。

![截屏2025-08-22 17.03.53.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/2def5fb3.png)

严格来讲， Bagel 算是第一个进行了超大规模预训练的统一模型，其独特之处真正做了基模级别的混合模态数据的训练（上一个还是Chameleon），并且论文中也提到了这样 setting 下的 emerging capabilities。

## 单一模型同时做 AR + Diffusion

与前几中路线的 AR + Diffusion 路线不同，这里指的 AR 和 Diffusion 其实是指损失函数的定义。这一路线的思路是仅使用一个Transformer模型来做序列建模和分布建模，在同一个序列中，文本 Token 使用 AR 的 NTP loss来做序列建模，而图像 Token 则使用 Diffusion 的损失函数来学习图像分布。典型的方法包括 Transfusion\[19\], Show-O\[20\], Show-O2\[21\]。

以 Transfuion 为例，其架构图如下所示。模型使用了一个 7B 的 Transformer 模型来做统一的序列和分布建模，对于文本 Token 做序列建模，对于图像 Token 做分布建模。图像理解和生成任务使用的图像编码都是 VAE 特征。Show-O 与 Show-O2 采用了类似的架构，只是它们做 Diffusion 时只有一个轻量的 Flow Head 做图像去噪，这里就不额外分析了。

![截屏2025-08-22 17.56.14.png](https://raw.githubusercontent.com/modelscope/modelscope-classroom/main/Blogs/Articles/All2All-Survey/resources/75f56643.png)

## 总结

总结以上几个路线的工作，目前可以得到的一些结论如下：

1.  对于图像理解任务，图像 tokenizer 适合类语义编码器，如SigLIP。对于图像生成任务，VAE的细节重建效果更好。
    
2.  图像生成的过程中至少有一个环节做图像分布建模，才能保证更好的图像生成质量。
    

从这两点结论来看，仅使用 AR 的路线，对统一模型是不够的，也是因此，既 Janus-Pro 之后，没有太多延完全相同架构的模型开源。后续类似的最新工作，比如 Show-O2, X-Omni\[22\] 或者 NextStep\[23\] 都至少采用了轻量级 Flow Head 或者比较大的 Diffusion Transformer 来做图像生成。


对于另外几个技术路线的统一模型，目前来看 AR + Diffusion 串联路线是最稳妥也是比较容易出效果的路线。**实际上，训练数据才是模型效果的核心，Data is all you need，真正能被广泛认可和广泛使用的模型都是在架构上没有明显问题，同时训练数据准备充分的模型。**在训练数据不同的情况下，通过模型在 Benchmark 上的指标来反映模型在架构上的优劣势不现实的，况且对于图像生成来说，图像生成效果与现有的评测指标也有 Bias。因此，目前可能没有明确的结论能证明某种架构就是比其他架构好。
但目前的统一模型还需要回答一个核心的问题，任务间互相促进的潜力是否真正存在，理解和生成是否能力互相促进，统一做理解和生成是否能做到1+1>2的效果。如果不谈这些问题，只是为了做统一模型去做统一模型，就很难真正训练出有用的模型，一个比较有反思意义的例子就是很多工作都在用了 Qwen2.5-Vl-7B 作为图像理解基模，也都将基模冻住了，但是在评测图像理解能力的时候出现了很多个不同的评测数值。1+1 能否大于 2 的问题需要一些真正经过大规模训练的工作来验证，比如 Bagel 验证了图文交错数据大规模预训练的有效性，Qwen-Image 的发布也证明了更好的文本编码，能带给生成和编辑的增益是很大的，这是一个很好的开始。

尽管存在上面说的问题，统一模型方向仍然是学术界和工业界都会紧跟的方向，其通用统一的叙事也很符合大家对 AGI 的畅想，况且统一给生成效果带来的增益已经被 Qwen-Image 等前沿基模给证明了。所以，让我们继续紧跟统一模型的发展，见证理解和生成的进化吧！

参考文献：

\[1\] Taming Transformers for High-Resolution Image Synthesis

\[2\] World Model on Million-Length Video And Language With Blockwise RingAttention

\[3\] Chameleon: Mixed-Modal Early-Fusion Foundation Models

\[4\] Emu3: Next-Token Prediction is All You Need

\[5\] Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation

\[6\] Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling

\[7\] Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities

\[8\] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning

\[9\] Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space

\[10\] BLIP3-o: A Family of Fully Open Unified Multimodal Models—Architecture, Training and Dataset

\[11\] Transfer between Modalities with MetaQueries

\[12\] UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation

\[13\] Qwen-Image Technical Report

\[14\] OmniGen2: Exploration to Advanced Multimodal Generation

\[15\] Step1X-Edit: A Practical Framework for General Image Editing

\[16\] FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space


\[17\] LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation
\[18\] Emerging Properties in Unified Multimodal Pretraining

\[19\] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

\[20\] Show-o: One Single Transformer to Unify Multimodal Understanding and Generation

\[21\] Show-o2: Improved Native Unified Multimodal Models

\[22\] X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again

\[23\] NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale
