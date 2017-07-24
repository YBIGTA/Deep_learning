YBIGTA 10기 노혜미

# skip-gram with tensorflow

* 해당 코드는 오픈 소스임을 밝힌다. 

https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py

- 해당 코드에 좀 더 자세하게 설명을 달아뒀다. 혹은 코드에 쓰인 용어에 대한 설명이다.
- 아직 skip-gram을 잘 모른다면? 아니면 word2vec을 직관적으로 이해하고 싶다면?

https://github.com/YBIGTA/Deep_learning/blob/master/RNN/nlp/%EC%9D%B4%EB%A1%A0/%EB%8B%A8%EC%96%B4%2C%20%EB%AC%B8%EC%9E%A5%EC%9D%84%20vector%EB%A1%9C%20%ED%91%9C%ED%98%84%ED%95%98%EA%B8%B0.md




```python
# -*- coding: utf-8 -*-

# 절대 임포트 설정

from __future__ import absolute_import

from __future__ import print_function


# 필요한 라이브러리들을 임포트
import pandas as pd

import collections

import math

import os

import random

import zipfile

import numpy as np

from six.moves import urllib

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf
```

*step1은 크롤링이나 캐글에서 다운로드 받을 경우 필요 없는 과정*


```python
# Step 1-1: 필요한 데이터를 다운로드한다.

# 해당 사이트에서 필요한 파일을 다운로드 받는 듯. 
url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):

  """파일이 존재하지 않으면 다운로드하고 사이즈가 적절한지 체크한다."""

    if not os.path.exists(filename):

    filename, _ = urllib.request.urlretrieve(url + filename, filename)

  statinfo = os.stat(filename)

    if statinfo.st_size == expected_bytes:

    print('Found and verified', filename)

    else:

    print(statinfo.st_size)

    raise Exception(

        'Failed to verify ' + filename + '. Can you get to it with a browser?')

    return filename

filename = maybe_download('text8.zip', 31344016)



# 문자열로 데이터를 읽는다

def read_data(filename):

  """zip파일 압축을 해제하고 단어들의 리스트를 읽는다."""

  with zipfile.ZipFile(filename) as f:

    data = f.read(f.namelist()[0]).split()

    return data



words = read_data(filename)

print('Data size', len(words)
```

대신 csv 파일인 뉴스 레딧을 이용해보자. (kaggle에 다양한 데이터가 있으니 처리하기 쉽고 맘에 드는 걸 하나 고르자.)


```python
# Step 1-2: 필요한 csv파일을 불러온다.

data = pd.read_csv('RedditNews.csv')
News = data['News']
sentences = News.tolist()
sentences[0]
```




    'A 117-year-old woman in Mexico City finally received her birth certificate, and died a few hours later. Trinidad Alvarez Lira had waited years for proof that she had been born in 1898.'




```python
# 단어 단위로 쪼개준다.
words = []
for sentence in sentences:
    words += sentence.split(' ')
for word in words:
    word = word.strip('"')
    word = word.strip("'.!:")
```

## step2 보충 설명 
- UNK 토큰?

unknown token의 줄임말.

train에서 빈도 수가 거의 없는 단어를 처리하기 위한 토큰 혹은 트레인에는 없지만 테스트에는 있는 단어를 처리하기 위한 토큰 


- list.extend: appending elements from the iterable.

x = [1, 2, 3]

x.extend([4, 5])

[1, 2, 3, 4, 5]


- Counter(words).most_common(10)

[('the', 1143), ('and', 966), ('to', 762), ('of', 669), ('i', 631),
 ('you', 554),  ('a', 546), ('my', 514), ('hamlet', 471), ('in', 451)]

## Skip-gram 모델의 코드 이해를 돕기 위해...


예를 들어, 아래와 같은 데이터셋이 주어졌다고 가정해보자.

the quick brown fox jumped over the lazy dog 

이번 구현에서는 간단하게, 콘텍스트를 타겟 단어의 왼쪽과 오른쪽 단어들의 윈도우로 정의한다. 

윈도우 사이즈를 1로하면 (context, target) 쌍으로 구성된 아래와 같은 데이터셋을 얻을 수 있다.

([the, brown], quick), ([quick fox], brown), ([brown, jumped], fox), …

‘quick’이라는 타겟단어로부터 콘텍스트 ‘the’와 ‘brown’을 예측하는 것이다. 

따라서 우리 데이터셋은 아래와 같은 (input, output) 쌍으로 표현할 수 있다.

(quick, the), (quick, brown), (brown, quick), (brown, fox), …


```python
# Step 2: dictionary를 만들고 UNK 토큰을 이용해서 rare words를 교체(replace)한다.

vocabulary_size = 50000 # 사용할 빈발 단어의 수


def build_dataset(words):

    count = [['UNK', -1]]

    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    # 등장 빈도가 상위 5000개인 단어만 넣는다. 그렇지 못하면 모두 UNK처리.
    
    dictionary = dict()

    for word, _ in count:

        dictionary[word] = len(dictionary)
    # len이 계속 증가하므로 결과적으로 index의 효과


    data = list()

    unk_count = 0

    for word in words:

        if word in dictionary:

            index = dictionary[word]

        else:

            index = 0  # dictionary['UNK']

            unk_count += 1

        data.append(index)
    # data에 word index를 추가해준다. 이때 unk토큰일 경우 0으로 추가해준다.    

    count[0][1] = unk_count

    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    return data, count, dictionary, reverse_dictionary



data, count, dictionary, reverse_dictionary = build_dataset(words)

del words  # Hint to reduce memory.

print('Most common words (+UNK)', count[:5])

print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])



data_index = 0
```

    Most common words (+UNK) [['UNK', 0], ('the', 39411), ('to', 37378), ('of', 33279), ('in', 32235)]
    Sample data [36, 72657, 184, 4, 244, 791, 1192, 1307, 91, 1470] ['A', '117-year-old', 'woman', 'in', 'Mexico', 'City', 'finally', 'received', 'her', 'birth']


## skip window, num skips?

바로 예시를 들어보겠다.

skip window = 1, num skips = 2이고 "So I drink beer on desk"가 있다.

그리고 target word를 drink라고 해보자.

skip window가 target기준 양 방향으로 얼마나 많은 맥락을 고려할지니까 

skip window가 1이라면 (context, target)으로 ([I, beer], drink)이런 결과를 얻을 수 있다.

만약 skip window 2라면 ([So, I, beer, on] drink)이런 식으로 결과를 얻을 것이다.

만약 ([So, I, beer, on] drink)에서 num skips = 2이면 랜덤하게 (I, drink), (on, drink) 이렇게 2개의 pair가 만들어 진다. 

num skips = 4 라면(so, drink), (I, drink), (beer, drink), (on, drink) 이렇게 만들어진다!

num skips가 skip window의 2배이면 모든 맥락을 고려하기 때문에 랜덤하게 뽑을 필요가 없어진다.



```python
# Step 3: skip-gram model을 위한 트레이닝 데이터(batch)를 생성하기 위한 함수.

# batch: 일정 갯수의 데이터.

def generate_batch(batch_size, num_skips, skip_window):
    
# skip_window : 왼쪽과 오른쪽으로 얼마나 많은 단어를 고려할지를 결정.
# num_skips: context window 내에서 (target, context) pair를 얼마나 생성할 지.
# 보충 아래

    global data_index

    assert batch_size % num_skips == 0

    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    # batch_size만큼의 열을 가진 batch 행vector 생성. ex) [1,2,3,4,5,6,7,8]

    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    # batch_size 만큼의 행을 가진 labels 열vector 생성 ex)[[1],
    #                                                   [2]]

    span = 2 * skip_window + 1 # [ skip_window target skip_window ]

    buffer = collections.deque(maxlen=span)
    # deque: 양 방향 큐. 양쪽 방향에서 데이터를 추가하고 삭제할 수 있음.
    for _ in range(span): 

        buffer.append(data[data_index])
        #위에서 data_index는 0으로 정의 됐음. 

        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):

        target = skip_window  # target label at the center of the buffer

        targets_to_avoid = [ skip_window ]

        for j in range(num_skips):

            while target in targets_to_avoid:

                target = random.randint(0, span - 1)

            targets_to_avoid.append(target)

            batch[i * num_skips + j] = buffer[skip_window]

            labels[i * num_skips + j, 0] = buffer[target]

            buffer.append(data[data_index])

            data_index = (data_index + 1) % len(data)

    return batch, labels



batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

for i in range(8):

    print(batch[i], reverse_dictionary[batch[i]],

      '->', labels[i, 0], reverse_dictionary[labels[i, 0]])
```

    72657 117-year-old -> 184 woman
    184 woman -> 72657 117-year-old
    4 in -> 184 woman
    244 Mexico -> 791 City
    791 City -> 1192 finally
    1192 finally -> 791 City
    1307 received -> 1192 finally
    91 her -> 1470 birth



```python
batch
```




    array([72657,   184,     4,   244,   791,  1192,  1307,    91])




```python
labels
```




    array([[  184],
           [72657],
           [  184],
           [  791],
           [ 1192],
           [  791],
           [ 1192],
           [ 1470]])




```python
sentences[0]
```




    'A 117-year-old woman in Mexico City finally received her birth certificate, and died a few hours later. Trinidad Alvarez Lira had waited years for proof that she had been born in 1898.'



117-year-old woman 의 맥락으로는 woman이, Mexico의 맥락으로는 City가 선택됐다고 이해하면 된다. 

## tf.nn.embedding_lookup?

tf.nn.embedding_lookup(params,ids) 예시

params = tf.constant([10,20,30,40])

ids = tf.constant([1,1,3])

tf.nn.embedding_lookup(params,ids).eval()

[20 20 40]

행벡터나 열벡터 라면 index는 각 열이나 행이 되고 매트릭스라면 index는 각 행이 된다!

아래 예시랑 비슷한 개념이다...!


```python
matrix = np.random.random([10, 5])  # 64-dimensional embeddings
ids = np.array([0, 2, 4, 6])
print (matrix)
print('-----------')
print (matrix[ids])  # prints a matrix of shape [4, 64]
```

    [[ 0.69790448  0.80412738  0.41803897  0.26215691  0.9590197 ]
     [ 0.06496029  0.22054213  0.34001469  0.00508572  0.3686428 ]
     [ 0.46248305  0.84020074  0.9926538   0.68563721  0.11789896]
     [ 0.85309971  0.80802479  0.83194076  0.13241919  0.95143293]
     [ 0.37420272  0.75321878  0.21514199  0.85283033  0.45458437]
     [ 0.19402967  0.59125361  0.69986521  0.51820879  0.72360693]
     [ 0.42823774  0.74959653  0.90853251  0.12835431  0.80546921]
     [ 0.93690952  0.24667782  0.6866535   0.38764635  0.06023061]
     [ 0.20540125  0.81291631  0.7122529   0.75812136  0.76815115]
     [ 0.3066154   0.10670046  0.70190714  0.09828408  0.40535762]]
    -----------
    [[ 0.69790448  0.80412738  0.41803897  0.26215691  0.9590197 ]
     [ 0.46248305  0.84020074  0.9926538   0.68563721  0.11789896]
     [ 0.37420272  0.75321878  0.21514199  0.85283033  0.45458437]
     [ 0.42823774  0.74959653  0.90853251  0.12835431  0.80546921]]


# nce loss?

- 원래 standard neural network는 multinomial (multi-class) classifier였다. 그래서 이러한 뉴럴 넷은 보통 확률을 필요로 하는

cross-entropy cost function으로 트레인 했다. 즉, 모든 천차만별의 결과를 soft max를 통해 0~1의 값으로 정규화(normalize)할 필요

가 있었다. 하지만 soft max는 큰 output layer에 적용하는데 좀 무리였다고 한다...


- 해결책

==> *nce loss(noise-contrastive estimation)*

: 잡것(?)과 대조하여 추정하는 방법

a multinomial classification -> a binary classification

soft max를 사용하는 대신 binary logistic regression를 사용하기로 한 것이다.

classifier에 center word와 context word로 되어있는 true pair와 center word와 랜던 단어로 되어있는 false pair를 트레이닝 시킨다.

true pair와 false pair를 구분하는 것을 배워서 결국에는 word vectors를 학습하는거라고 한다.

중요한 점은 일반적인 training technique처럼 다음 단어를 예측하는 것이 아니라 단지 이 pair가 good이냐 bad이냐를 판단하면 되는 것이다!

ex) target: 'dog', context: 'bark'

true pair: ('dog', 'bark')

false pair: ('dog', 'fly'), ('dog', 'study')...


```python
# Step 4: skip-gram model 만들고 학습시킨다.


batch_size = 128

embedding_size = 128  # embedding vector의 크기.

skip_window = 1       # 윈도우 크기 : 왼쪽과 오른쪽으로 얼마나 많은 단어를 고려할지를 결정.

num_skips = 2         # context window 내에서 (target, context) pair를 얼마나 생성할 지.



# sample에 대한 validation set은 원래 랜덤하게 선택해야한다. 하지만 여기서는 validation samples을 

# 가장 자주 생성되고 낮은 숫자의 ID를 가진 단어로 제한한다.

valid_size = 16     # validation 사이즈, 유사성을 평가할 단어 집합 크기.

valid_window = 100  # 분포의 앞부분(head of the distribution)에서만 validation sample을 선택한다.

valid_examples = np.random.choice(valid_window, valid_size, replace=False)
# array([88, 72, 22,  2,  6, 25, 42,  7, 85, 12, 23, 46, 29, 89, 15, 26])

num_sampled = 64    # sample에 대한 negative examples의 개수.

# Negative Sampling
# target word와 그에 대한 context words가 있을 때, target word와 context word를 묶은 true pair를 만들고 target word와 관계없는 
# random word를 묶은 negative example들 k개 생성해서 훈련.
# ex) target:'dog', context word: 'bark' -> negative example: ('dog', 'fly')


graph = tf.Graph()
# tf.Graph(): 그룹으로 실행되는 op들의 집합.

with graph.as_default():

# 트레이닝을 위한 인풋 데이터들

    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])

    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
    
    # tf.placeholder: traing할 때 feed_dict로 값을 먹이는 변수
    # tf.constant: 변하지 않는 값.

# Ops and variables pinned to the CPU because of missing GPU implementation
# embedding_lookup이 GPU implementation이 구현이 안되어 있어서 CPU로 해야함.
# default가 GPU라서 명시적으로 CPU라고 지정해줌.
    with tf.device('/cpu:0'):

    # embedding vectors 행렬을 랜덤값으로 초기화

        embeddings = tf.Variable(

            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            # tf.random_uniform: 각각의 구간에서 동일한 확률로 표현되는 분포로 사각형 모양
            # 최소 -1값과 최대 1값으로 50000(voca_size)x128(emb_size)행렬을 만들겠다는 뜻이다. => 초기화


    # 행렬에 트레이닝 데이터를 지정

        embed = tf.nn.embedding_lookup(embeddings, train_inputs)
        # 밑에 보충설명
        # 전체 embedding matrix에서 train_inputs이 가리키는 임베딩 벡터만을 추출


    # NCE loss를 위한 변수들을 선언

        nce_weights = tf.Variable(

            tf.truncated_normal([vocabulary_size, embedding_size],

                                 stddev=1.0 / math.sqrt(embedding_size)))
        # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
        # 잘린 정규분포. 정규분포의 일부분을 사용하고 싶을 때 씀.
        # 초기화를 할 때 0 주위의 값으로 초기화하는데 normal은 드물지만 꽤 큰 값들이 나올 수 있음. 그런 경우를 방지하기 위해서 씀.

        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
        # tf.zeros: shape만큼의 0값을 가진 vector나 matrix를 만들어줌.
        # 데이터가 치우져 있지 않다고 가정하고 0으로 둠.

# batch의 average NCE loss를 계산한다.

# tf.nce_loss 함수는 loss를 평가(evaluate)할 때마다 negative labels을 가진 새로운 샘플을 자동적으로 생성한다.

    loss = tf.reduce_mean( 
        # tf.reduce_mean: 평균 내주는 함수.

        tf.nn.nce_loss(weights=nce_weights,

                         biases=nce_biases,

                         labels=train_labels,

                         inputs=embed,

                         num_sampled=num_sampled, # 샘플링할 단어 수

                         num_classes=vocabulary_size)) # 맞춰야 할 단어 갯수(?)



# SGD optimizer를 생성한다.

    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    


# minibatch examples과 모든 embeddings에 대해 cosine similarity를 계산한다.

    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    # norm: 벡터 크기.

    normalized_embeddings = embeddings / norm
    # -1.0~1.0의 uniform 분포 값을 가지는 embeddings이기 때문에 각 행의 크기(열 아님)는 다르다. 
    # 따라서 그 크기로 나눠줘서 크기를 1로 맞춰줌.
    
    valid_embeddings = tf.nn.embedding_lookup(

      normalized_embeddings, valid_dataset)
    # normalized embeddings와 array([88, 72, 22,  2,  6, 25, 42,  7, 85, 12, 23, 46, 29, 89, 15, 26])이렇게 생긴 
    # vaild_dataset을 embedding lookup 해줌. 즉 normalized embedding에서 해당 row를 보겠다는 뜻.

    similarity = tf.matmul(

      valid_embeddings, normalized_embeddings, transpose_b=True)
    # tf.matmul: 행렬끼리 곱해주는 함수.
    #(16,128) * (128,50000) = 16*50000
    # ==> normalized_embeddings를 transpose했기 때문.
    # 유사도를 볼 단어들의 임베딩과 모든 다른 단어들의 임베딩을 곱해준다. 값이 클 수록 유사한 단어라는 뜻!
    
```


```python
# Step 5: 트레이닝을 시작한다.

num_steps = 100001



with tf.Session(graph=graph) as sess:

# 트레이닝을 시작하기 전에 모든 변수들을 초기화한다.

    sess.run(tf.global_variables_initializer())

    print("Initialized")

    
    average_loss = 0

    
    for step in xrange(num_steps):

        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)

        feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
    # train_inputs에 batch_inputs을, train_labels에 batch_labels를 값으로 준다.


    # optimizer op을 평가(evaluating)하면서 한 스텝 업데이트를 진행한다.

        _, loss_val = sess.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += loss_val



    if step % 2000 == 0:

        if step > 0:

            average_loss /= 2000

    # 평균 손실(average loss)은 지난 2000 배치의 손실(loss)로부터 측정된다.

        print("Average loss at step ", step, ": ", average_loss)

        average_loss = 0
    # 2000번을 기준으로 그 때의 average loss계산



    # Note that this is expensive (~20% slowdown if computed every 500 steps)

    if step % 10000 == 0:

        sim = similarity.eval()

        for i in xrange(valid_size):

            valid_word = reverse_dictionary[valid_examples[i]] # 유사성을 평가할 단어, 얘랑 누구랑 유사한가?

            top_k = 8 # nearest neighbors의 개수

            nearest = (-sim[i, :]).argsort()[1:top_k+1]

            log_str = "Nearest to %s:" % valid_word

            for k in xrange(top_k):

                close_word = reverse_dictionary[nearest[k]]

                log_str = "%s %s," % (log_str, close_word)

            print(log_str)

    final_embeddings = normalized_embeddings.eval()
```

    Initialized
    Average loss at step  100000 :  4812.81963046
    Nearest to may: could, will, should, would, must, might, can, GM/GE,
    Nearest to this: sights, Copenhagen, damn, Ezzouek,, heads., Accuser, quarantine", hearing,
    Nearest to being: multi-million-dollar, Sick-Man, (after, support;, were, sponsors, suggestion, captive,
    Nearest to Israel: Russia, acid, sabotaging, US, NATO, China, U.S., lord,
    Nearest to people: men, aggravated, children, civilians, 6000km, nearly, infiltrators', b'PETA,
    Nearest to Russian: Israeli, German, Australian, retreat,, Turkish, Ukranian, Egyptian, Italian,
    Nearest to Korea: Korea's, levies, China, b'Iran, Russia, Strauss, Wouldn't, Iran,
    Nearest to not: soon, airport",, developments.", b'Mapping, banks', Don\'t, Nasa:, reports,,
    Nearest to up: tremors, pressed, metal, recklessly,, severe, (91.5, people,, simplified,
    Nearest to one: insensitive, consumers."', Bolivias, collapsed., fray, show,, idea, Saud,,
    Nearest to but: self-reliance,, Sawers, 819,100%, Lhasa', ..., Offer", 5-cent, five-mile,
    Nearest to British: Israeli, Australian, pacts, Turkish, mercury, Indian, Italian, interior,
    Nearest to China: Russia, Germany, Pakistan, North, Belgians,, 63F,, Sharia, myself.",
    Nearest to were: are, have, being, statuses, was, Levels', squad', be,
    Nearest to Saudi: Wake,, length, ISIS,, incomprehensible, dreamed, believed., dust, general,,
    Nearest to its: their, Bus, DIE, planning,, Qusra, shrink, brother, Galaxy,



```python
# Step 6: embeddings을 시각화한다.



def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):

    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

    plt.figure(figsize=(18, 18))  #in inches

    for i, label in enumerate(labels):

        x, y = low_dim_embs[i,:]
        
        plt.scatter(x, y)

        plt.annotate(label,

                 xy=(x, y),

                 xytext=(5, 2),

                 textcoords='offset points',

                 ha='right',

                 va='bottom')
        
        plt.savefig(filename)

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt        
        
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

plot_only = 500

low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only,:])

labels = [reverse_dictionary[i] for i in xrange(plot_only)]

plot_with_labels(low_dim_embs, labels)

```

### 시각화 결과





![word-embedding](http://i.imgur.com/6u0IujA.png)
'뭐 어쩌라는 거지'라는 생각이 들 것이다. 

일부를 확대해서 결과를 확인해보자!

![조동사](http://i.imgur.com/j8IjsZz.png)

조동사끼리 뭉쳐있는 것을 확인할 수 있다.

다른 결과를 하나만 더 보자.

![국민](http://i.imgur.com/kWDUcHz.png)

이번에는 국민을 지칭하는 단어끼리 꽤 잘 뭉쳐 있다. 

(잘 찾아보면 Korean도 있다.)

이렇듯 word2vec은 문맥을 고려해주기 때문에 단어간의 유사도를 볼 때 매우 유용하다!

### reference

- https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/examples/tutorials/word2vec/word2vec_basic.py


- http://solarisailab.com/archives/374

- http://khanrc.tistory.com/entry/TensorFlow-7-word2vec-Implementation

  ​


