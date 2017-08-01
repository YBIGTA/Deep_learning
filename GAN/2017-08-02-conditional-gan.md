# Conditional Generative Adversarial Nets

<h6 align="right">강병규</h6>

## Introduction

GAN은 까다로운 확률 계산의 어려움으로부터 벗어나, generative 모델을 훈련시킬 수 있는 새로운 대안으로 떠오르고 있습니다. Adversarial Nets는 마르코프 체인과 달리 여러 이점을 가집니다. 이를테면 gradient를 구하기 위해 backpropagation만 하면 되며, 학습과정에서 어떠한 추론도 필요하지 않다는 것 등입니다. 특히 log-likelihood를 만들어 낼 수 있다는 점을 주목할 만 합니다.

Unconditioned GAN에서는 만들어지는 데이터를 조작할 수 없었습니다. 하지만 모델에 추가적인 정보를 주기만하면, 데이터를 만들어내는 과정을 지시할 수 있게 됩니다. 이때 추가적인 정보는 class label같은 것들을 생각하면 됩니다. 혹은 다른 양상(modality)에서 온 데이터여도 상관 없습니다.

## Conditional Adversarial Nets

### Generative Adversarial Nets

GAN은 두가지 'Adversarial'한 모델을 가집니다. 데이터의 분포를 파악하는 generative 모델 ${G}$, 그리고 어떠한 sample이 G가 아닌, 실제 데이터에서 왔을 확률을 추정하는 discriminative model ${D}$가 그 둘입니다. ${G}$와 ${D}$ 모두 non-linear function, 이를테면 Multi-Layer Perceptron이라는 점이 특징이죠.
이때 generator는 실제 데이터 ${x}$에 대한 분포를 모방하는 generator의 분포 ${p_g}$를 학습해야합니다. 이때 noise의 분포 ${p_z{(z)}}$에서 실제 데이터 공간으로의 mapping function을 ${G(z;\theta_g)}$라고 정의할 수 있습니다. discriminator의 경우 ${D(x;\theta_d)}$는 ${x}$가 ${p_g}$가 아닌 실제 데이터에서 왔을 확률을 나타내는 한개의 scalar를 결과값으로 만들어냅니다.

${G}$와 ${D}$는 동시에 학습이 진행됩니다. 이러한 관계는 two-player min-max game, $${\min_G \max_D = \mathbb{E}_{x\sim {p_{data}(x)}}[\log(D(x))] + \mathbb{E}_{z\sim {p_z(z)}} [log(1-D(G(z)))]}$$으로 나타낼 수 있습니다.


### Conditional Adversarial Nets

GAN의 generator와 discriminator에 어떤 추가적인 정보 ${y}$만 넣어주면 Conditional 모델을 만들 수 있습니다. ${y}$는 어떠한 정보여도 상관없습니다. ${y}$를 입력으로 받는 추가적인 input layer를 discriminator와 generator 모두 더해주기만 하면 됩니다. 이렇게 말이죠.

![cond-gan](https://user-images.githubusercontent.com/25279765/28810813-acb1b642-76c6-11e7-94cc-cf5cdeb579d4.PNG)

generator의 경우 input noise ${p_z(z)}$와 ${y}$가 hidden layer에서 결합하고 discriminator의 경우 ${x}$와 ${y}$가 discriminative function에 입력으로 들어가게 되는 것입니다. 이때 two-player min-max의 목적함수는 $${\min_G \max_D = \mathbb{E}_{x\sim {p_{data}(x)}}[\log(D(x|y))] + \mathbb{E}_{z\sim {p_z(z)}} [log(1-D(G(z|y)))]}$$와 같아집니다. 즉 어떠한 데이터가 ${y}$일 때 가짜인 확률과 진짜인 확률을 추정한다고 생각하면 됩니다.
