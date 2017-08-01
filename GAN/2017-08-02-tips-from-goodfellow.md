# NIPS 2016 Tutorial: GAN
### Tips from Goodfellow
<h6 align="right">강병규</h6>

이번에는 2014년 Goodfellow의 논문에 이어 2016년 NIPS에서 저자가 직접 강의한 Tutorial에 대해 알아보고자 합니다. 크게

1. 왜 Generative modeling을 공부하는가?
2. 어떻게 이러한 모델들이 작동하는가? GAN이 다른 generative model에 대해 갖는 이점이 무엇인가?
3. 어떻게 GAN이 작동하는가?
4. GAN을 학습시킬 때 tip과 trick들
5. 연구 분야
6. 이 Tutorial 당시 새롭게 제시된 모델인 Plug and Play Generative Networks에 대한 설명
7. Excercise
8. Solution to Excercise
9. 결론

정도로 이루어져 있는데, 이번 글에서는 4번, GAN을 학습시키는 과정에서 사용할 수 있는 팁들에 대해서만 알아보겠습니다.

## Introduction

 GAN의 퍼포먼스를 향상시킬 수 있는 여러가지 테크닉들이 있습니다. 하지만 이러한 테크닉과 트릭들이 얼마나 효율적인지를 정확히 말하기란 어렵습니다. 어떤 상황에서는 향상이 될지도 모르지만, 또 다른 상황에서는 오히려 방해가 될 수도 있죠. 결국 여기서 소개하는 팁들은 시도해볼만하긴 하지만, 항상 최고의 결과를 가져오는 건 아니라는 걸 아셔야합니다.


## Train with labels

어떤 식으로든, 어떤 형태든 label을 사용하는 것은 모델이 만들어내는 sample의 질을 엄청나게 향상시킬 수 있습니다. Denton et al(2015)이 처음 제시했는데요, 이 논문에서는 class-conditional GAN을 만들었습니다. 일반적인 GAN은 어떤 class인지는 신경쓰지 않고 sample을 만든다는 점에서 class-conditional GAN과 차이가 있습니다. 이렇게 할 경우 성능이 더 좋아진다고 합니다. 또 Slimans et al. (2016)에서는 generator에 class 정보를 주는 것이 아니라 discriminator가 실제 데이터의 class들을 분류하도록 학습시켰더니 성능이 더 좋아졌다고 합니다.

 이러한 trick이 어떻게 가능한지 얘기하가는 어렵습니다. 아마 class의 정보를 포함하는 것이 최적화에 도움이 되는 어떤 정보를 주기때문아닐까요? 또 이 trick의 문제는 sample의 질이라는게 엄청 주관적이라는 것입니다. 어쩌면 사람의 시각 체계가 sample을 볼때 어떠한 biases가 개입한 것일 수도 있단 얘기지요. 만약 이러한 경우리면, 이 trick은 실제 데이터의 분포를 모방하는 더 좋은 모델을 만들어주는 것이 아니라, 그저 사람이 더 즐길 수 있는 오락거리를 만들어주는 것일지도 모릅니다.

중요한 점은 이 trick을 통해 얻은 결과는 똑같은 trick을 사용해서 얻은 결과물과 비교를 해야한다는 것입니다. 즉 trick을 사용하지 않았을 때의 결과와 trick을 사용해서 얻은 결과를 비교해서는 안되는 것입니다. CNN(Convolutional Neural Network)의 경우를 생각해봅시다. 이미지와 관련된 task에서 CNN 모델이 CNN모델이 아닌 network보다 일반적으로 더 좋은 성능을 낸다는 것은 명확합니다. 이때 CNN의 성능을 보기 위해 CNN이 아닌 모델과 비교하는 것은 잘못된 것과 비슷하다고 생각하시면 됩니다.

## One-sided label smoothing

GAN은 discriminator가 두 확률밀도함수, 즉 generator와 실제 데이터의 분포를 추정하면서 작동해야합니다. 그러니까 가짜 데이터와 실제 데이터를 진짜인지, 가짜인지 판단하면서 작동하는 것입니다. 하지만 deep neural net은 옳은 class인지를 식별하는 높은 confident의 output을 만들어내는 것에 취약합니다. 대신 매우 높은 확률 값을 만들어냅니다. (but deep neural nets are prone to producing highly confident outputs that identify the correct class but with too extreme of a probability)

adversarial한 network의 경우 이러한 상황이 더 자주 발생하죠. classifier, 그러니까 discriminator가 선형으로(linearly) 진짜인지 가짜인지를 추론하고, 극단적으로 confident한 예측을 만들어냅니다. discriminator가 부드러운 형태로 확률을 예측하도록 하기 위해서 **One-sided label smoothing** 라는 테크닉을 사용합니다. (Salimans et al. 2016)

discriminator의 cost를 ${J^{(D)}}$라고 하고, ${D}$가 사용하는 parameter를 ${\theta^{(D)}}$, ${G}$의 경우, ${\theta^{(D)}}$라고 합시다. 보통 우리는 discriminator를 $${J^{(D)}(\theta^{(D)}, \theta^{(G)})} = -\frac{1}{2}\mathbb{E}_{x\sim p_{data}}\log D(x) - \frac{1}{2}\mathbb{E}_z \log(1-D(G(z)))$$
를 최소화하도록 학습을 시킵니다.

이 cost를 계산하는 코드를 텐서플로우(Abadi et al., 2015)를 통해 구현한다면
```{.python}
d_on_data = discriminator_logits(data_minibatch)
d_on_samples = discriminator_logits(samples_minibatch)
loss = tf.nn.sigmoid_cross_entropy_with_logits(d_on_data, 1.) + \
 tf.nn.sigmoid_cross_entropy_with_logits(d_on_samples, 0.)
```

과 같습니다.

One-sided label smoothing의 핵심은 실제 데이터에 대한 target 값을 1보다 약간 작은 값, 이를테면 0.9로 해준다는 것입니다. 이를 코드로 나타낸다면 다음과 같을 것입니다.

```{.python}
loss = tf.nn.sigmoid_cross_entropy_with_logits(d_on_data, .9) + \
 tf.nn.sigmoid_cross_entropy_with_logits(d_on_samples, 0.)
```

이렇게 한다면 discriminator의 extraploation을 극대화할 수 있습니다. 즉 어떠한 input에 대해 1에 가까울 정도로 높은 확률값을 예측하도록 학습이 되었다고 합시다. 이러한 경우 페널티를 얻고, 예측값이 좀 더 작아지게 하는 것이죠. 이때 중요한 것은 가짜 sample에 대해서는 smooth하지 않는 것입니다.

우리가 실제 데이터에 대해서 ${1-\alpha}$의 target, 가짜 sample에 대해서는 ${0 + \beta}$의 target을 갖는다고 해봅시다. 그렇다면 이상적인 discriminator는 $${D^{\star}(x) = \frac{(1-\alpha)p_{data}(x) + \beta p_{model}(x)}{p_{data}(x) + p_{model}(x)}} $$ 를 나타낼 것입니다.

이때 $\beta$가 0이라면, 즉 smoothing을 하지 않는다면 $\alpha$가 smoothing하는 것은 그냥 discriminator의 최적 값을 낮춰주는 것밖에는 없습니다. 만약 $\beta$가 0이 아니라면 discriminator의 최적 값은 변하게 되죠. 특히 $p_{data}(x)$가 매우 작고 $p_{model}(x)$가 매우 큰 경우를 생각해봅시다. 이 경우에 ${D^\star}$는 $p_{model}(x)$의 spurious mode에서 최고값을 가지게 될 것입니다. 이렇게 되면 discriminator는 generator의 잘못된 행동을 강화하게 됩니다. 결국 generator는 실제 데이터와 비슷한 sample을 만들거나, 이미 만들었던 sample과 비슷한 sample을 만들게 됩니다.

One-sided label smoothing은 이전부터, 최소한 80년대부터 존재했던 label smoothing technique의 간단한 변형입니다. Szegedy et al.(2015)는 object recognition을 위해 CNN을 사용할 때, label smoothing이 좋은 regularizer라고 말했습니다. 이러한 label smoothing이 regularizer로 잘 작동하는 이유 중 하나는 모델이 training set의 incorrect class를 선택하지 않도록 하고, correct class에 대한 confidence를 낮춰주기 때문입니다.

![latex](https://user-images.githubusercontent.com/25279765/28833933-d95f5190-771b-11e7-93b0-e2ff2d5a6def.png)
> weigth decay(regularization), 이전에 구한 loss와 weight의 크기를 더한다. 이때의 ${\alpha}$는 hyper-parameter로, 얼마나 강하게 regularization을 할지 결정한다.

weight decay와 같은 다른 regularizer들은 regularizer의 coefficient가 지나치게 높게 설정되었다면, misclassification의 가능성이 생깁니다.

Warde-Farley 와 Goodfellow(2016)은 label smoothing이 가짜 데이터, sample에 대한 취약성을 줄이는데 도움이 된다는 것을 보였습니다. 즉 discriminator가 generator의 공격을 효율적으로 대응하는데  label smoothing이 도움이 된다는 이야기입니다.

## Virtual batch normalization

DCGAN이 소개된 이후로, 대부분의 GAN 구조는 어떤 종류든 batch normalization을 가지는 경향을 보이고 있습니다. 이렇게하면 모델을 reparameterizing하게 되고, 이는 각 feature의 평균과 분산이 각각 한 개의 parameter에 의해 결정되도록 합니다. 즉 feature를 추출하기 위해 사용된 모든 layer들의 복잡한 상호작용으로 feature의 평균과 분산을 결정할 필요가 없게 되는 것이죠. 결국 batch normalization의 목적은 모델의 최적화과정을 향상시키는 것에 있습니다.

이러한 데이터의 minibatch에서, feature의 평균을 빼고 표준편차로 나누면 이러한 reparameterization이 가능합니다.

![bn1](https://user-images.githubusercontent.com/25279765/28708644-ef7537f4-73b7-11e7-9b0c-02c4e59b540c.png)

> 일반적인 Neural Networks에서 batch normalization을 적용하는 경우를 생각해봅시다. input이 다음 layer로 들어가고 일련의 연산과 activation function을 거쳐 output이 나오게 됩니다. 이때 batch normalization을 적용한다면, input이 들어가기전에 위의 연산을 통해 input을 변화시키고 그 다음에야 layer의 input으로 전달합니다.

이때 중요한 점은 normalization 연산 또한 모델의 일부이므로, backpropagation 과정에서 항상 normalized된 feature의 gradient를 계산한다는 것에 있습니다. 만약 normalization이 모델의 일부가 아니고, 학습 뒤에야 feature들이 renormalize된다면 영향이 줄어들게 됩니다.

Batch normalization은 매우 유용한 것임은 분명합니다. 하지만 GAN에겐 안타깝게도 몇 개의 부작용이 있습니다. 데이터에서 각각의 minibatch를 사용해 학습 과정에서 normalization을 수행한다면 normalizing한 feature들의 값이 변화하게 됩니다. minibatch의 크기가 작다면, 이러한 fluctuation의 영향이 커지게 되고, 결국 input noise ${z}$보다 만들어내는 이미지에 미치는 영향이 더 커집니다.

![batch](https://user-images.githubusercontent.com/25279765/28708639-ecc4e2ca-73b7-11e7-8f6c-5c02c756839b.PNG)

> 위 아래, 두 개의 minibatch에서 나온 결과물입니다. 이때 보시면 위쪽은 모두 오렌지색, 그리고 아래쪽은 모두 초록색을 가지는 sample이 나온 것을 볼 수 있습니다. 즉 input noise ${z}$보다 normalization의 영향이 더 커지게 된 것입니다.

Salimans et al.(2016)은 이 문제를 해결하기 위한 해결책을 내놓았습니다. 바로 **Reference batch normalization** 인데요, 핵심은 generative network를 두 번 돈다는 것입니다. 한 번은 Reference example의 minibatch를, 그리고는 현재 학습하고 있는 데이터들의 minibatch를 input으로 주는 것이죠. 학습을 시작할 때 Reference example을  뽑아낸 다음, 절대로 바꾸지 않는다는 점을 주의해야합니다.

이때, 각 feature의 평균과 표준편차는 Reference batch에서만 구합니다. 그 다음 이렇게 Reference batch에서만 구한 값을 가지고 normalizing을 수행합니다. 하지만 이렇게 한다면 모델이 Reference batch에 대해서 overfitting될 것입니다. 이런 overfitting문제를 약간 완화시켜주기 위해서 **virtual batch normalization** 을 사용할 수 있습니다. Reference batch에 현재 학습중인 데이터를 포함해서 feature들의 평균과 표준편차를 구하는 것이죠.

Reference batch normalization과 virtual batch normalization 모두 training minibatch의 모든 example들이 서로 독립적으로 처리되고, generator가 만들어내는 모든 sample들이 i.i.d (independent identically distributed)하다는 특징을 가집니다.

## Can one balance G and D?

GAN의 기본적인 가정은 게임이론에서 비롯되었습니다. 그렇다면 아마 대부분의 사람들이 하나가 다른 하나를 압도하지 못하도록 어떻게든 균형을 맞춰줄 필요가 있다고 생각할 것입니다.

GAN은 데이터의 확률밀도함수와 모델, 즉 가짜 데이터의 확률밀도함수의 비율을 추정하면서 작동합니다. 이러한 비율은 discriminator가 최적일때만 가능합니다. 즉 discriminator가 generator를 압도하는 것은 괜찮은 것입니다.  하지만 discriminator가 너무 정확하다면 가끔 generator의 gradient가 사라지는 문제가 생깁니다. 이러한 문제를 해결하는 좋은 방법은 discriminator를 무능력하게 만드는 것이 아니라, gradient가 사라지지 않도록 하는 parameterization을 사용하는 것입니다.

학습 초기과정을 생각해봅시다. 아마 ${G}$가 만든 가짜는 진짜와 큰 차이가 있을 것입니다. 그렇게 되면 ${D}$는 손쉽게 진짜와 가짜를 구분할 수 있게 됩니다. 이런 경우 ${G}$의 vanishing gradient 문제가 발생할 수 있습니다. 이에 대해서는 [이전 글](https://kangbk0120.github.io/articles/2017-07/first-gan)을 읽어보세요.

또 discriminator가 confident해진다면 generator의 gradient가 매우 커질 수도 있게 됩니다. 역시 discriminator를 부정확하게 만드는 것보다, 위에서 설명한 One-sided label smoothing을 적용하는 것이 좋습니다. 두 비율을 예측하기 위해 discriminator가 최적이어야한다는 것은 discriminator를 k(>1)단계에 대해서 학습을 시킨다음, generator를 한번 학습시킨다는 얘기와 같습니다. 하지만 실제로는 이렇게 해도 뚜렷한 향상이 생기지 않습니다.

generator와 discriminator의 균형을 맞추기위해 모델의 크기를 바꿀 수도 있습니다. 일반적으로 discriminator가 generator보다 더 깊고, 더 많은 filter를 layer에 가집니다. ([예시1](https://github.com/YBIGTA/Deep_learning/blob/master/GAN/2017-07-29-GAN-tutorial-2-MNIST.markdown) [예시2](https://angrypark.github.io/GAN-tutorial-1/))

이렇게 하는 이유는 discriminator가 실제 데이터의 분포와 ${G}$가 만드는 가짜 데이터의 분포를 비교, 추정해야하기 때문입니다. 하지만 이렇게하면 mode collapse라는 문제가 생깁니다. generator가 현재 학습 방법에서 모든 capacity를 사용하지 않기 때문에, 결국 구현하는 입장에서는 generator의 크기를 늘리는 것이 얼마나 이득이 될지 모릅니다. 이러한 문제를 해결할 수 있다면, generator의 크기는 아마 늘어날 수 있을 것입니다. 그러나 discriminator의 크기 또한 비례해서 늘어날지는 알 수 없습니다.

###### mode collapse?

**Helvetica scenario** 라고도 합니다. generator가 서로 다른 ${z}$들을 하나의 output point로 매핑할 때 발생하는 문제입니다. 일반적으로 완전한 mode collapse는 잘 일어나지 않고, 부분적인 mode collapse가 주로 일어납니다. 이때 부분적이라는 뜻은 같은 색이나 같은 텍스처를 갖는 이미지를 여러장 만든다거나 하는 경우를 생각하면 될 것같습니다.

![mode_collapse](https://user-images.githubusercontent.com/25279765/28835648-1559130c-7721-11e7-95ec-e08499039369.PNG)

>위의 사진을 보면 첫번째 줄에는 실제 데이터의 분포 ${p_{data}}$를 볼 수 있습니다. 이때 ${p_{data}}$는 2차원에서 Gaussian의 mixture입니다. 그 밑줄에는 GAN의 학습이 진행되면서 만들어지는 분포들의 사진이 있습니다. training set에 있는 모든 mode를 포함하는 분포로 수렴하지 않고, 한번에 한개의 mode만 만들어내는 모습을 볼 수 있습니다. 이는 discriminator가 각각을 reject하기 때문입니다. (Metz et al.(2016))

## Conclusion

여기서 Goodfellow가 제안하는 효과적인 테크닉들을 정리해보자면

1. label을 가지고 학습을 시키기
2. One-sided label smoothing
3. Virtual batch normalization

정도로 정리할 수 있을 것 같습니다.

분명 구현 과정에서 시도해볼 만한 부분들입니다. 그러나 다른 딥러닝 패러다임에서처럼 이러한 테크닉들이 반드시 성능향상을 가져다주는 것이 아님을 주의해야 합니다.

---
## Reference

Salimans *et al*. "Imporve Technique for Training GANs", 2016
