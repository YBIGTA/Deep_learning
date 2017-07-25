
# Review of 
## ROBUST CONVOLUTIONAL NEURAL NETWORKS UNDER ADVERSARIAL NOISE

https://arxiv.org/pdf/1511.06306.pdf

## 요약

여러 연구 [CVPR 2017 https://arxiv.org/pdf/1610.08401.pdf] 들을 보면 CNN은 작은 perturbation ( adversarial examples )에 취약합니다. 위 논문은 adversarial noise에 robust한 CNN을 모델을 제안하였습니다. 

## 취약한 CNN (이미지 공격)

### 1. Input Noise

Image에 random noise ( 정규 분포 ) 를 추가함. 

$$ X_{ijk} = x_{ijk} + N(\mu x_{ijk},\sigma^2_N) $$

### 2. Universal Adversarial Perturbations - https://arxiv.org/pdf/1610.08401.pdf

Perturbation은 '섭동'이라는 뜻인데, 천문학 기준으로는 원래의 궤도에서 벗어나게 하는 힘을 의미한다고 합니다. 

해당 논문에서는, 이미지 분류를 제대로 하지 못하게 하는 방해 요소라는 의미로 이해하면 좋겠습니다

![image](https://pbs.twimg.com/media/CwFOOn-WcAADvRv.jpg)

https://github.com/LTS4/universal/blob/master/python/universal_pert.py

### Calc Perturbation

최소한의 이미지 vector 이동을 통한 예측 오류 생성


```python
def universal_perturbation(dataset, f, grads, delta=0.2, max_iter_uni = np.inf, xi=10, p=np.inf, num_classes=10, overshoot=0.02, max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax)
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """

    v = 0 # image 이동 vector
    fooling_rate = 0.0
    num_images = np.shape(dataset)[0]

    itr = 0
    while fooling_rate < 1-delta and itr < max_iter_uni: # fooling rate가 어느정도 이상이 되거나, 많은 iteration을 돌았을 때
        
        np.random.shuffle(dataset) # 데이터 set 섞기
        
        
        #################################### 시작 ##########################################
        
        # Pertubation 계산
        for k in range(0, num_images):
            cur_img = dataset[k:(k+1), :, :, :]
            
            # v가 image에 영향을 끼치지 못할 정도로 작은 경우, v value 업데이트
            if int(np.argmax(np.array(f(cur_img)).flatten())) == int(np.argmax(np.array(f(cur_img+v)).flatten())):

                # Pertubation 계산
                dr,iter,_,_ = deepfool(cur_img + v, f, grads, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter_df)

                # v value 업데이트
                if iter < max_iter_df-1:
                    v = v + dr

                    # Project on l_p ball
                    v = proj_lp(v, xi, p)

        #################################### 끝 ##########################################
                    
        itr = itr + 1

        # Perturb the dataset with computed perturbation
        dataset_perturbed = dataset + v

        est_labels_orig = np.zeros((num_images))
        est_labels_pert = np.zeros((num_images))

        batch_size = 100
        num_batches = np.int(np.ceil(np.float(num_images) / np.float(batch_size)))

        # Compute the estimated labels in batches
        for ii in range(0, num_batches):
            m = (ii * batch_size)
            M = min((ii+1)*batch_size, num_images)
            est_labels_orig[m:M] = np.argmax(f(dataset[m:M, :, :, :]), axis=1).flatten()
            est_labels_pert[m:M] = np.argmax(f(dataset_perturbed[m:M, :, :, :]), axis=1).flatten()

        # Compute the fooling rate
        fooling_rate = float(np.sum(est_labels_pert != est_labels_orig) / float(num_images))
        print('FOOLING RATE = ', fooling_rate)

    return v
```


```python
def deepfool(image, f, grads, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
       :param grads: gradient functions with respect to input (as many gradients as classes).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 10)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    f_image = np.array(f(image)).flatten()
    I = (np.array(f_image)).flatten().argsort()[::-1]

    I = I[0:num_classes]
    label = I[0] # model의 데이터에 대한 예측 label

    input_shape = image.shape
    pert_image = image

    f_i = np.array(f(pert_image)).flatten()
    k_i = int(np.argmax(f_i)) # label이랑 다를게 없는 것 같은데...?

    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    #################################### 시작 ##########################################
    while k_i == label and loop_i < max_iter: # 예측되는 label이 달라졌거나, iteration을 많이 돌렸으면 탈출!

        pert = np.inf
        gradients = np.asarray(grads(pert_image,I)) # input과 실제 label에 따른 변경될 gradient 계산

        for k in range(1, num_classes):

            # set new w_k and new f_k
            w_k = gradients[k, :, :, :, :] - gradients[0, :, :, :, :]
            f_k = f_i[I[k]] - f_i[I[0]]
            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        r_i =  pert * w / np.linalg.norm(w)
        r_tot = r_tot + r_i

        # perturbation 추가한 이미지
        pert_image = image + (1+overshoot)*r_tot
        loop_i += 1

        # label 계산을 다시 함
        f_i = np.array(f(pert_image)).flatten()
        k_i = int(np.argmax(f_i))
    #################################### 끝 ##########################################
    
    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, k_i, pert_image
```

### Perturbation Result

![image](http://i.imgur.com/T6fqjvP.png)

Universal Adversarial Pert.의 Remarkable 한 점은, 기존의 공격들은 공격된 이미지들을 포함하여 training 시키면 모델들의 robustness를 강화시킬수 있었으나 - UAP의 공격된 이미지는 training 시켜도 robustness가 강화되지 못했다는점!

## 이미지 방어 전략

논문 - ROBUST CONVOLUTIONAL NEURAL NETWORKS UNDER ADVERSARIAL NOISE
흠... 별건 없고 trained 된 모델을 feedfowarding할 때 input에 noise를 주고 뭔가 layer마다 stochastic한 성질은 주는건가? ㅎㅎ

https://github.com/jhjin/stochastic-cnn/tree/master/demo

### Input Noise Model

위의 Input Noise를 줌

### 충격! 논문이 정말 별거 없었다..

Input에다가 Noise를 주면 뒤의 모든 layer들은 stochastic 해짐... 변하는건 없음!

와... 논문의 Contribution이 feedfoward할 때 Input Noise만 준 것 뿐... 성능은 ㄱㅊㄱㅊ

### Result

![image](https://github.com/jhjin/stochastic-cnn/raw/master/demo/visualization.jpg)

# Image 방어 분야는 한-참 갈길이 멀다...
