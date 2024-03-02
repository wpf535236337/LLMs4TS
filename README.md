# LLMs4TS

Awesome resources of Large language models **(LLMs)** for time-series dataset. These resources concentrate on *self-supervised methods*, *GNN4TS*, *MLPer* and go beyond Transformer-based models. Whether you're a professional or data scientist, these tools and techniques can enhance your time-series data analysis with LLMs or foundation models.

#### Single modality for TS

* Forecasting
  * PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting [arxiv 20 Sep 2022](https://arxiv.org/abs/2210.08964) [PISA)](https://github.com/HaoUNSW/PISA)
  * Temporal Data Meets LLM -- Explainable Financial Time Series Forecasting [arxiv 19 Jun 2023](https://arxiv.org/abs/2306.11025)
  * Time-LLM: Time Series Forecasting by Reprogramming Large Language Models [arxiv 3 Oct 2023](https://arxiv.org/abs/2310.01728)
  * TimeGPT-1 [arxiv 2023 *5 Oct 2023*](https://arxiv.org/abs/2310.03589)
  * TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting [ arxiv 8 Oct 2023](https://arxiv.org/abs/2310.04948) 
  * Pushing the Limits of Pre-training for Time Series Forecasting in the CloudOps Domain [arxiv 2023 8 Oct 2023](https://arxiv.org/abs/2310.05063)
  * Large Language Models Are Zero-Shot Time Series Forecasters [ NeurIPS 2023 11 Oct 2023](https://arxiv.org/abs/2310.07820)  [llmtime](https://github.com/ngruver/llmtime)
  * Lag-Llama: Towards Foundation Models for Time Series Forecasting [arxiv 12 Oct 2023](https://arxiv.org/abs/2310.08278)  [Lag-Llama](https://github.com/kashif/pytorch-transformer-ts)
  * iTransformer: Inverted Transformers Are Effective for Time Series Forecasting [arxiv 13  Oct 2023 ]() [iTransformer]([GitHub - thuml/iTransformer: This is the official implementation for "iTransformer: Inverted Transformers Are Effective for Time Series Forecasting".](https://github.com/thuml/iTransformer))
  * Harnessing LLMs for Temporal Data - A Study on Explainable Financial Time Series Forecasting [2023.emnlp-industry](https://aclanthology.org/2023.emnlp-industry.69.pdf)
  * LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs [arxiv16 Aug 2023](https://arxiv.org/abs/2308.08469) 
  * UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting [15 Oct 2023](https://arxiv.org/abs/2310.09751)
  * AutoTimes: Autoregressive Time Series Forecasters via Large Language Models [4 Feb 2024](https://arxiv.org/abs/2402.02370)
  * LSTPrompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting [25 Feb 2024](https://arxiv.org/abs/2402.16132v1) [LSTPrompt](https://github.com/AdityaLab/lstprompt)

* Anomaly Detection
  * Large Language Model Guided Knowledge Distillation for Time Series Anomaly Detection [26 Jan 2024](https://arxiv.org/abs/2401.15123)

* Imputation 
  * GATGPT: A Pre-trained Large Language Model with Graph Attention Network for Spatiotemporal Imputation [24 Nov 2023](https://arxiv.org/abs/2311.14332)

* Spatio-temporal prediction
  * Spatio-Temporal Graph Learning with Large Language Model [20 Sept 2023](https://openreview.net/forum?id=QUkcfqa6GX)
  * How Can Large Language Models Understand Spatial-Temporal Data? [25 Jan 2024](https://arxiv.org/abs/2401.14192)

* One Fits all
  * One Fits All:Power General Time Series Analysis by Pretrained LM [arxiv 23 Feb 2023](https://arxiv.org/abs/2302.11939) [NeurIPS2023-One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All) [no-officail reproduction](https://github.com/liaoyuhua/GPT-TS)
  * TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series [arxiv16 Aug 2023](https://arxiv.org/abs/2308.08241#:~:text=TEST%3A%20Text%20Prototype%20Aligned%20Embedding%20to%20Activate%20LLM's%20Ability%20for%20Time%20Series,-Chenxi%20Sun%2C%20Yaliang&text=This%20work%20summarizes%20two%20strategies,LLM%20to%20handle%20TS%20data.)

* Multimodal
  * Frozen Language Model Helps ECG Zero-Shot Learning [22 Mar 2023](https://arxiv.org/abs/2303.12311)
  * FinGPT: Open-Source Financial Large Language Models [arxiv 9 Jun 2023](https://arxiv.org/abs/2306.06031) [AI4Finance-Foundation/FinNLP](https://github.com/AI4Finance-Foundation/FinNLP)
  * FinVis-GPT: A Multimodal Large Language Model for Financial Chart Analysis [ (FinLLM 2023)@IJCAI 2023](https://arxiv.org/abs/2308.01430)
  * Insight Miner: A Time Series Analysis Dataset for Cross-Domain Alignment with Natural Language [NeurIPS2023-AI4Science Poster](https://openreview.net/forum?id=E1khscdUdH)


#### Multimodal LM include TS

* Meta-Transformer: A Unified Framework for Multimodal Learning [20 Jul 2023](https://arxiv.org/abs/2307.10802) [Meta-Transformer](https://kxgong.github.io/meta_transformer/)
* UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video,Point Cloud, Time-Series and Image Recognition [27 Nov 2023](https://arxiv.org/abs/2311.15599) [UniRepLKNet](https://invictus717.github.io/UniRepLKNet/)
* ViT-Lens-2: Gateway to Omni-modal Intelligence [27 Nov 2023](https://arxiv.org/abs/2311.16081) [ViT-Lens-2](https://github.com/TencentARC/ViT-Lens)

#### Patch && Tokenizers methods

* Patches Are All You Need? [*ICLR 24 Jan 2022*](https://arxiv.org/abs/2201.09792)  [convmixer](https://github.com/locuslab/convmixer)
* A Time Series is Worth 64 Words: Long-term Forecasting with Transformers [ICLR 2023 *27 Nov 2022*](https://arxiv.org/abs/2211.14730) [PatchTST](https://github.com/yuqinie98/PatchTST)
* PatchMixer: A Patch-Mixing Architecture for Long-Term Time Series Forecasting [arxiv  *1 Oct 2023*](https://arxiv.org/abs/2310.00655) [PatchMixer](https://github.com/Zeying-Gong/PatchMixer) [Chinese blog](https://mp.weixin.qq.com/s/gfAqL7bosdc0mxz8SziBbQ)
* Learning to Embed Time Series Patches Independently [NeurIPS Workshop on Self-Supervised Learning: Theory and Practice, 2023](https://arxiv.org/abs/2312.16427) [pits](https://github.com/seunghan96/pits)
* The first step is the hardest: Pitfalls of Representing and Tokenizing Temporal Data for Large Language Models [arxiv 12 Sep 2023](https://arxiv.org/abs/2309.06236)
* Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot
  Forecasting of Multivariate Time Series [8 Jan 2024](https://arxiv.org/abs/2401.03955)
* What Makes for Good Visual Tokenizers for Large Language Models? [20 May 2023](https://arxiv.org/abs/2305.12223) [GVT](https://github.com/TencentARC/GVT)
* SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models [ICLR2024](https://arxiv.org/abs/2308.16692) [SpeechTokenizer](https://0nutation.github.io/SpeechTokenizer.github.io/)

#### Mixture-of-Experts (MoE)

* Mixture-of-Linear-Experts for Long-term Time Series Forecasting [11 Dec 2023](https://arxiv.org/abs/2312.06786)
* Prompt-based Domain Discrimination for Multi-source Time Series Domain Adaptation [19 Dec 2023](https://arxiv.org/abs/2312.12276)
* A Review of Sparse Expert Models in Deep Learning [4 Sep 2022](https://arxiv.org/abs/2209.01667)
* MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts [8 Jan 2024](https://arxiv.org/abs/2401.04081) [](https://github.com/llm-random/llm-random)

#### Biosignals dataset

* Neuro-GPT: Developing A Foundation Model for EEG [arxiv 7 Nov 2023](https://arxiv.org/abs/2311.03764)
* Brant: Foundation Model for Intracranial Neural Signal [NeurIPS23](http://yangy.org/works/brainnet/NeurIPS23_Brant.pdf)
* PPi: Pretraining Brain Signal Model for Patient-independent Seizure Detection [NeurIPS23](http://yangy.org/works/brainnet/NeurIPS23_PPi.pdf)
* Large-scale training of foundation models for wearable biosignals [submit ICLR 2024](https://openreview.net/forum?id=pC3WJHf51j)
* BIOT: Cross-data Biosignal Learning in the Wild [NeurIPS23](https://arxiv.org/abs/2305.10351) [BIOT](https://github.com/ycq091044/BIOT)
* Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI [submit ICLR 2024](https://openreview.net/forum?id=QzTpTRVtrP)
* Practical intelligent diagnostic algorithm for wearable 12-lead ECG via self-supervised learning on large-scale dataset [Nature Communications 2023](https://www.nature.com/articles/s41467-023-39472-8)
*  Large AI Models in Health Informatics: Applications, Challenges, and the Future [IEEE Journal of Biomedical and Health Informatics](https://arxiv.org/abs/2303.11568) [Awesome-Healthcare-Foundation-Models](https://github.com/Jianing-Qiu/Awesome-Healthcare-Foundation-Models)
*  Data science opportunities of large language models for neuroscience and biomedicine [Neuron](https://www.sciencedirect.com/science/article/pii/S0896627324000424)
* BioSignal Copilot: Leveraging the power of LLMs in drafting reports
  for biomedical signals [July 06, 2023](https://www.medrxiv.org/content/10.1101/2023.06.28.23291916v1)
* Health-LLM: Large Language Models for Health Prediction via Wearable Sensor Data [12 Jan 2024](https://arxiv.org/abs/2401.06866)
* Self-supervised Learning for Electroencephalogram: A Systematic Survey [9 Jan 2024](https://arxiv.org/abs/2401.05446)

####  Foundation for other domains

* Foundation Models for *Weather and Climate Data* Understanding: A Comprehensive Survey [5 Dec 2023](https://arxiv.org/abs/2312.03014)
* [Towards Urban General Intelligence: A Review and Outlook of Urban Foundation Models](https://arxiv.org/abs/2402.01749 ) [usail-hkust/Awesome-Urban-Foundation-Models](https://github.com/usail-hkust/Awesome-Urban-Foundation-Models)
* 

#### Multimodal TS

* Frequency-Aware Masked Autoencoders for Multimodal Pretraining on Biosignals [12 Sep 2023 ](https://arxiv.org/abs/2309.05927)
* FOCAL: Contrastive Learning for Multimodal Time-Series Sensing Signals in Factorized Orthogonal Latent Space [*NeurIPS'23* 30 Oct 2023](https://arxiv.org/abs/2310.20071) [focal](https://github.com/tomoyoshki/focal)
* Multimodal Adaptive Emotion Transformer with Flexible Modality Inputs on A Novel Dataset with Continuous Labels [ACMMM 27 October 2023](https://dl.acm.org/doi/10.1145/3581783.3613797)
* 

#### Survey4TS

* Self-supervised Contrastive Representation Learning for Semi-supervised Time-Series Classification [TPAMI 13 Aug 2022](https://arxiv.org/abs/2208.06616) [CA-TCC](https://github.com/emadeldeen24/CA-TCC)
* Unsupervised Representation Learning for Time Series: A Review [3 Aug 2023](https://arxiv.org/abs/2308.01578) [ULTS](https://github.com/mqwfrog/ULTS)
* Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects [16 Jun 2023](https://arxiv.org/abs/2306.10125) [SSL4TS](https://github.com/qingsongedu/Awesome-SSL4TS)
* A Survey on Time-Series Pre-Trained Models [18 May 2023](https://arxiv.org/abs/2305.10716)  [time-series-ptms](https://github.com/qianlima-lab/time-series-ptms)
* Self-Supervised Learning for Time Series: Contrastive or Generative? [AI4TS workshop at IJCAI 2023](https://github.com/AI4TS/AI4TS.github.io/blob/main/CameraReadys%201-22%202/5%5CCameraReady%5CIJCAI23_TSworkshop_Jun29.pdf) [SSL_Comparison](https://github.com/DL4mHealth/SSL_Comparison)
* Self-Supervised Contrastive Learning for Medical Time Series: A Systematic Review [*Sensors* in 2023](https://www.mdpi.com/1424-8220/23/9/4221) [Contrastive-Learning-in-Medical-Time-Series-Survey](https://github.com/DL4mHealth/Contrastive-Learning-in-Medical-Time-Series-Survey)
* Applications of Self-Supervised Learning to Biomedical Signals: where are we now [**post date 2023-04-11**](https://www.techrxiv.org/articles/preprint/Applications_of_Self-Supervised_Learning_to_Biomedical_Signals_where_are_we_now/22567021/2)
* What Constitutes Good Contrastive Learning in Time-Series Forecasting? [last revised 13 Aug 2023](https://arxiv.org/abs/2306.12086)
* Large Models for Time Series and Spatio-Temporal Data: A Survey and Outlook [arxiv 16 Oct 2023](https://arxiv.org/abs/2310.10196) [Awesome-TimeSeries-SpatioTemporal-LM-LLM](https://github.com/qingsongedu/Awesome-TimeSeries-SpatioTemporal-LM-LLM)
* Toward a Foundation Model for Time Series Data [21 Oct 2023](https://dl.acm.org/doi/abs/10.1145/3583780.3615155) [code](https://sites.google.com/view/timeclr)
* Universal Time-Series Representation Learning: A Survey [8 Jan 2024](https://arxiv.org/abs/2401.03717) [itouchz/awesome-deep-time-series-representations](https://github.com/itouchz/awesome-deep-time-series-representations)
* A Review for Pre-Trained Transformer-Based Time Series Forecasting Models  [ITMS2023](https://ieeexplore.ieee.org/abstract/document/10317721)
* Large Language Models for Time Series: A Survey [2 Feb 2024](https://arxiv.org/abs/2402.01801) [xiyuanzh/awesome-llm-time-series](https://github.com/xiyuanzh/awesome-llm-time-series)
* Position Paper:What Can Large Language Models Tell Us about Time Series Analysis [5 Feb 2024](https://arxiv.org/abs/2402.02713)
* A Survey of Deep Learning and Foundation Models for Time Series Forecasting [25 Jan 2024](https://arxiv.org/abs/2401.13912)

#### Project

* [ddz16/TSFpaper](https://github.com/ddz16/TSFpaper)
* [Time_Series_Instruct](https://github.com/michael-wzhu/Time_Series_Instruct)
* [qingsongedu/Awesome-TimeSeries-AIOps-LM-LLM](https://github.com/qingsongedu/Awesome-TimeSeries-AIOps-LM-LLM)
* [LLM for Time Series](https://github.com/liaoyuhua/LLM4TS)
* [Time Series AI Papers](https://github.com/xiyuanzh/time-series-papers)
* [Multivariate Time Series Transformer Framework](https://github.com/gzerveas/mvts_transformer)
* [xiyuanzh/time-series-papers](https://github.com/xiyuanzh/time-series-papers)
* [vincentlux/Awesome-Multimodal-LLM](https://github.com/vincentlux/Awesome-Multimodal-LLM)
* [yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)
* [SitaoLuan/LLM4Graph](https://github.com/SitaoLuan/LLM4Graph)
* Foundation-Models
  * [uncbiag/Awesome-Foundation-Models](https://github.com/uncbiag/Awesome-Foundation-Models)
  * [UbiquitousLearning/Efficient_Foundation_Model_Survey](https://github.com/ubiquitouslearning/efficient_foundation_model_survey)
  * [zhanghm1995/Forge_VFM4AD](https://github.com/zhanghm1995/Forge_VFM4AD)
  * [Yangyi-Chen/Multimodal-AND-Large-Language-Models](https://github.com/Yangyi-Chen/Multimodal-AND-Large-Language-Models)
  * [MM-LLMs: Recent Advances in MultiModal Large Language Models](https://mm-llms.github.io/)
  * [Learning on Multimodal Graphs: A Survey](https://arxiv.org/abs/2402.05322)
  * [The (R)Evolution of Multimodal Large Language Models: A Survey](https://arxiv.org/abs/2402.12451)
* 


#### Blog list

* [Anomaly Detection in Time Series using ChatGPT]( https://medium.com/@sztistvan/anomaly-detection-in-time-series-using-chatgpt-3fc48f958c88)
* [Change Point Detection in Time Series using ChatGPT](https://medium.com/@sztistvan/change-point-detection-in-time-series-using-chatgpt-22cc9172a130)
* [Huggingface 镜像站](https://hf-mirror.com/)
* 

#### Laboratory

* [Deep Learning for Mobile Health Lab](https://github.com/DL4mHealth)

### Course

* [Time Series Analysis from nan jing university](https://www.lamda.nju.edu.cn/yehj/TSA2023/) 
* [Time-Series Analysis from intel company ](https://www.intel.com/content/www/us/en/developer/topic-technology/artificial-intelligence/training/course-time-series-analysis.html)

#### Others

* 
