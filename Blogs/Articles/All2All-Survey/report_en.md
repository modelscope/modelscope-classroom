# Understanding Unified Models for Image Understanding and Generation——Comprehending Cutting-edge Model Architectures

Generative multimodal models have been a research hotspot in the industry in recent years. Vision Language Models (VLMs) have been the core approach in the field of multimodal text generation, capable of completing image understanding tasks; Diffusion Models have been the core method in the field of image and video generation. Earlier this year, unified models that simultaneously support image understanding and generation emerged like mushrooms after rain. Unified models are favored not only because of the versatility brought by supporting both understanding and generation tasks simultaneously, but also because people have seen the potential of multimodal learning brought by the organic combination of tasks. On one hand, the organic combination of the two tasks enables models to jointly optimize on both tasks, improving the utilization rate of interleaved text-image data, while also allowing academia to see the potential for mutual promotion between tasks. On the other hand, multimodal output support gives unified models more possibilities in the currently popular model inference, such as developing inference based on generated images and image generation based on reasoning.

Starting from understanding and generation tasks, before unified models, understanding tasks were completed by Vision Language Models following the autoregressive (AR) route; while generation tasks were completed by Diffusion Models following the DDPM or FlowMatching route. Therefore, the technology of unified models also starts from these two routes. This article categorizes current typical unified model works into four types: pure autoregressive route, AR + Diffusion serial structure, AR + Diffusion parallel structure, and single model simultaneously doing AR + Diffusion. Since the technology and ecosystem of Diffusion for text understanding are not yet mature, this article does not cover unified models based solely on Diffusion.

Note that this article is not a complete survey and may not cover all papers. These four categories are designed to better capture the differences between different models. The fastest way to understand a unified model is to clarify how it handles the two tasks of image understanding and generation. For image understanding tasks, current unified models generally follow the VLM approach, encoding images into embeddings and then processing them uniformly with LLMs, which is a widely validated route; **the difference between different route methods in understanding tasks lies in the way images are encoded**. For the implementation of image generation tasks, the differences between different models are mainly reflected in several aspects, **namely how images are encoded, how image encodings are generated, and how images are decoded.** For each route below, we will focus on these difference points.

## Pure Autoregressive Route Unified Models

Autoregression means predicting the next token based on the input sequence and feeding the predicted token back into the input for recursive prediction. Pure autoregressive route unified models can be seen as a combination of text token prediction in LLMs and image token prediction in VQGAN[1]. Typical works include LWM[2], Chameleon[3], Emu3[4], Janus[5], and Janus-Pro[6]. LWM and Chameleon are relatively early works on unified training of text and images, Emu3 further extends to video generation modality, while Janus and Janus-Pro separate the encoding for image understanding. Here we analyze models of this route using Chameleon and Janus.

The model architecture of Chameleon is shown in the figure below. For image understanding tasks, it first uses the VQ-VAE Encoder as an Image Tokenizer to encode images into discrete embeddings, then uses an autoregressive model to predict text output. For image generation tasks, it uses the VQ-VAE Decoder as an Image De-Tokenizer to decode the discrete image tokens predicted by the autoregressive model into images.

![截屏2025-07-31 11.04.43.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/409da4d8.png)

Based on this architecture, the Janus team believes that the VQ-VAE Encoder is trained through reconstruction tasks and is not suitable for image understanding tasks in semantic space. Therefore, they modified the Image-Tokenizer for image understanding tasks to use SigLIP, which is pre-trained on image-text pairs. The architecture is shown in the figure below: Und. Encoder (understanding part) uses SigLIP, while Gen.Encoder (generation part) and Image Decoder use VQVAE.

![image.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/164d2b24.png)
Simply put, the unified model of the pure autoregressive route incorporates the discrete image tokens from VQ-VAE into the unified training of LLM or VLM. Its advantage is that the discrete image token prediction task highly aligns with the pre-training paradigm of LLM and is very consistent with the characteristics of AR models. However, from the perspective of image quality, the image quality generated by models in this route is unsatisfactory. On one hand, this is due to the effect loss caused by the discretization of the image encoding space, and on the other hand, it is because autoregressive models cannot perform distribution modeling like diffusion models. Additionally, due to the inability to introduce random noise, poor diversity in generated images is also a major challenge for this route.

## Unified Models with AR + Diffusion Cascaded Structure

AR and Diffusion are respectively the mainstream routes in image understanding and generation domains, so perhaps the simplest and most effective unified model route is to cascade AR and Diffusion together, where the AR model completes understanding tasks, and the input of the AR model serves as conditions for Diffusion to complete generation tasks. The architecture is shown in the figure below (image source: unified model survey [7]).

![截屏2025-08-22 15.25.01.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/244534b3.png)

For understanding tasks, images are encoded by semantic encoders (typically CLIP, SigLIP, or ViT trained with image-text alignment) into continuous embeddings. For generation (text-to-image) tasks, the model processes text input and outputs an intermediate embedding as a condition for the diffusion model to generate images. Here, the image encoding for generation tasks is the intermediate embedding of AR and Diffusion models, directly produced by the AR model. Based on whether the intermediate embedding is explicitly supervised, we categorize several typical works into two types.

### 2.1 Unified Modeling Methods Using Semantic Embedding Supervision

As the name suggests, these methods use loss functions to directly supervise the output image embeddings of AR models, giving them a clear embedding output target, while using these embeddings to train diffusion models for image reconstruction. Typical methods include MetaMorph[8], Nexus-Gen[9], and Blip-3o[10]. Their typical architecture is shown in the figure below:

![截屏2025-08-22 15.53.03.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/de912c54.png)

As shown in figure b above, MetaMorph and Nexus-Gen use Image Loss (typically MSE or cosine similarity loss) to supervise AR models, enabling them to learn to predict semantic embeddings of target images. The reason for doing this is, on one hand, **to answer why unified models should be built from the perspective of Joint Training**. In MetaMorph's experiments, image understanding and generation tasks trained together from scratch can mutually promote each other's effectiveness. On the other hand, Nexus-Gen's Unified Image Embedding Space models understanding and generation as inverse tasks, with the potential benefit of directly performing understanding on generated embeddings, thus having the potential for multi-round reasoning. Additionally, there exists a problem in this sub-route: supervising autoregressive models to predict continuous image embeddings leads to serious error accumulation issues. MetaMorph ignores this phenomenon, while Nexus-Gen adopts a prefill autoregressive strategy to solve it, which is essentially consistent with the Learnable Query approach used in other works (Blip-3o, MetaQuery[11]).

Blip-3o also follows a similar embedding supervision training approach, but they additionally use FlowMatching to perform distribution modeling on semantic embeddings, which addresses the issue that autoregressive models cannot perform distribution modeling in this architecture. As shown in the figure below:

![截屏2025-08-22 16.09.16.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/a40097db.png)

### 2.2 Methods for Direct Training of Diffusion Models
These methods generally freeze the AR model and directly use the hidden states output by the AR model as conditions for the Diffusion model, training only the Diffusion model for image generation. From another perspective, methods in this approach can be viewed as an evolution of Diffusion technology, where the commonly used T5 Text Encoder in Diffusion models is replaced with a larger multimodal generative model (such as Qwen2.5-VL-7B). Typical methods include Uniworld[12], MetaQuery (special note: MetaQuery uses Learnable Query as the condition extraction method, not hidden states), Qwen-Image[13], and OmniGen2[14]. For image generation tasks, the typical architecture is shown in the figure below (source: Qwen-Image). The text prompt is input into Qwen2.5-Vl, which directly outputs the hidden states corresponding to these tokens as text conditions for the subsequent Diffusion Transformer.

![截屏2025-08-22 16.26.38.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/1067ae96.png)

Beyond image generation, a potential strength of unified models lies in their image editing capabilities. Therefore, we provide additional analysis of the image editing architectures of these models here. Compared to image generation tasks, using Diffusion models for image editing requires an additional input condition: the encoding information of the image to be edited. The image to be edited can use two types of encoding. The first is semantic encoding, where encoders such as SigLIP are used. The second is reconstruction encoding, where VAE encoders are used.

1. Semantic Encoding Architecture: Taking Uniworld as an example, the architecture using semantic encoding is shown in the figure below. Focus particularly on the SigLIP component. As illustrated, the image to be edited passes directly through SigLIP and MLP before being input as a condition into DiT. Nexus-Gen also supports image editing and adopts this semantic encoding feature for condition injection architecture. Based on experimental experience with Nexus-Gen in image editing, semantic feature encoding, compared to VAE encoding, has an encoding space closer to the output semantic space of VLM, requiring only a small amount of training data to establish relationships between text conditions and image conditions, with the potential advantage of better instruction-following capability. The disadvantage of this encoding architecture is that semantic encoding involves information loss, reconstruction effectiveness is strongly correlated with the number of encoded tokens, and often cannot achieve one-to-one reconstruction. From the perspective of image editing reconstruction results, GPT4o-Image likely also uses semantic encoding.

![截屏2025-08-22 16.37.21.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/f697f06f.png)

2. VAE Encoding Architecture: Qwen-Image and OmniGen2 adopt this architecture. Looking further back, earlier open-source Step1X-Edit and Flux-Kontext use exactly the same architecture. Looking even further back, this architecture is consistent with the approaches of In-Context LoRA and OmniControl. Taking Qwen-Image as an example, the architecture is shown in the figure below. Focus particularly on the Input Image, which passes through the VAE Encoder before being input as a condition into DiT. In this architecture, positional encoding is generally used to distinguish between input images and denoised images. Both Qwen-Image[15] and Flux-Kontext[16] make this distinction directly in the first dimension (frame id) of positional encoding.

![截屏2025-08-22 16.54.17.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/fe100074.png)

## AR + Diffusion Parallel Structure

The above serial structure uses Embedding as the bridge between AR and Diffusion, while the parallel structure here uses Attention as the bridge between AR models and Diffusion models. Typical works include LlamaFusion[17] and Bagel[18]. Using the definition from Bagel, this architecture can be called Mixture-of-Transformer-Experts (MoT).

### 3.1 LlamaFusion Freezing Text Model
The architecture of LlamaFusion is shown in the figure below. Given a language model, as shown in the left figure, the authors duplicate its parameters to serve as dedicated image generation parameters in the right figure. For a sequence containing text and noisy images, text tokens are computed using the parameters from the left figure, while image tokens are trained using the parameters from the right figure. However, during the attention computation stage, all tokens are concatenated together for self-attention. Since the language model is frozen, this architecture cannot change the model's understanding capability, so it does not involve encoding issues in image understanding tasks. The encoding and decoding used for image generation are both VAE. Although the model adopts a language model structure, it actually uses the diffusion route for image generation.
![截屏2025-08-22 17.16.34.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/b0aebf44.png)

### 3.2 Bagel Mixed Text-Image Training

Bagel adopts an architecture similar to LlamaFusion, with the difference being that the model's image understanding and generation capabilities are all trained from scratch. _The encoder used for the model's image understanding is_ SigLIP, a semantic encoding model, while the encoder and decoder used for image generation are VAE, a reconstruction model. Understanding tasks use the AR method to autoregressively generate text tokens, while generation tasks adopt the diffusion method to generate VAE features of images.

![截屏2025-08-22 17.03.53.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/2def5fb3.png)

Strictly speaking, Bagel is the first unified model to undergo ultra-large-scale pretraining. Its uniqueness lies in truly conducting foundation model-level mixed modal data training (the previous one was Chameleon), and the paper also mentions emerging capabilities under such settings.

## Single Model Simultaneously Doing AR + Diffusion

Different from the AR + Diffusion routes of the previous approaches, the AR and Diffusion here actually refer to the definition of loss functions. The idea of this route is to use only one Transformer model for sequence modeling and distribution modeling. In the same sequence, text tokens use AR's NTP loss for sequence modeling, while image tokens use diffusion loss functions to learn image distributions. Typical methods include Transfusion[19], Show-O[20], Show-O2[21].

Taking Transfusion as an example, its architecture diagram is shown below. The model uses a 7B Transformer model for unified sequence and distribution modeling, performing sequence modeling for text tokens and distribution modeling for image tokens. The image encoding used for both image understanding and generation tasks are VAE features. Show-O and Show-O2 adopt similar architectures, except that they only have a lightweight Flow Head for image denoising when doing diffusion, which will not be analyzed separately here.

![截屏2025-08-22 17.56.14.png](https://github.com/modelscope/modelscope-classroom/blob/main/Blogs/Articles/All2All-Survey/resources/75f56643.png)

## Summary

Summarizing the work of the above several routes, some conclusions that can be drawn currently are as follows:

1. For image understanding tasks, image tokenizers are suitable for semantic encoders like SigLIP. For image generation tasks, VAE has better detail reconstruction effects.

2. At least one stage in the image generation process must perform image distribution modeling to ensure better image generation quality.

From these two conclusions, using only the AR route is insufficient for unified models. This is why, after Janus-Pro, there haven't been many models following exactly the same architecture being open-sourced. Subsequent similar latest works, such as Show-O2, X-Omni[22], or NextStep[23], all adopt at least lightweight Flow Heads or relatively large Diffusion Transformers for image generation.
For unified models of other technical routes, the AR + Diffusion cascaded approach currently appears to be the most stable and relatively easy to achieve results. **In fact, training data is the core of model performance. Data is all you need. Models that are truly widely recognized and widely used are those that have no obvious architectural problems while having well-prepared training data.** Under different training data conditions, it is unrealistic to reflect the architectural advantages and disadvantages of models through their performance on benchmarks. Moreover, for image generation, there is also bias between image generation quality and existing evaluation metrics. Therefore, there may currently be no clear conclusions that can prove one architecture is better than others.

However, current unified models still need to answer a core question: whether the potential for mutual promotion between tasks truly exists, whether understanding and generation capabilities can mutually enhance each other, and whether unifying understanding and generation can achieve a 1+1>2 effect. Without addressing these questions and simply pursuing unified models for the sake of unification, it would be difficult to truly train useful models. A reflective example is that many works have used Qwen2.5-VL-7B as the image understanding base model and frozen the base model, but when evaluating image understanding capabilities, many different evaluation results have emerged. The question of whether 1+1 can be greater than 2 requires verification from works that have undergone truly large-scale training. For example, Bagel verified the effectiveness of large-scale pre-training on interleaved image-text data, and the release of Qwen-Image also proved that better text encoding can bring significant gains to generation and editing, which is a good start.

Despite the aforementioned issues, the unified model direction remains one that both academia and industry will closely follow. Its universal and unified narrative aligns well with everyone's vision of AGI, and the gains that unification brings to generation quality have already been proven by cutting-edge base models like Qwen-Image. Therefore, let us continue to closely follow the development of unified models and witness the evolution of understanding and generation!

References:

[1] Taming Transformers for High-Resolution Image Synthesis

[2] World Model on Million-Length Video And Language With Blockwise RingAttention

[3] Chameleon: Mixed-Modal Early-Fusion Foundation Models

[4] Emu3: Next-Token Prediction is All You Need

[5] Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation

[6] Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling

[7] Unified Multimodal Understanding and Generation Models: Advances, Challenges, and Opportunities

[8] MetaMorph: Multimodal Understanding and Generation via Instruction Tuning

[9] Nexus-Gen: Unified Image Understanding, Generation, and Editing via Prefilled Autoregression in Shared Embedding Space

[10] BLIP3-o: A Family of Fully Open Unified Multimodal Models—Architecture, Training and Dataset

[11] Transfer between Modalities with MetaQueries

[12] UniWorld-V1: High-Resolution Semantic Encoders for Unified Visual Understanding and Generation

[13] Qwen-Image Technical Report

[14] OmniGen2: Exploration to Advanced Multimodal Generation

[15] Step1X-Edit: A Practical Framework for General Image Editing

[16] FLUX.1 Kontext: Flow Matching for In-Context Image Generation and Editing in Latent Space
\[17\] LlamaFusion: Adapting Pretrained Language Models for Multimodal Generation
\[18\] Emerging Properties in Unified Multimodal Pretraining

\[19\] Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model

\[20\] Show-o: One Single Transformer to Unify Multimodal Understanding and Generation

\[21\] Show-o2: Improved Native Unified Multimodal Models

\[22\] X-Omni: Reinforcement Learning Makes Discrete Autoregressive Image Generative Models Great Again

\[23\] NextStep-1: Toward Autoregressive Image Generation with Continuous Tokens at Scale
