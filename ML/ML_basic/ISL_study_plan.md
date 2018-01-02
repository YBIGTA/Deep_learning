An Introduction to Statistical Learning( ISL스터디 )
===================

![image1] (https://user-images.githubusercontent.com/31824102/34483251-91784b44-f000-11e7-948e-33812f81956b.PNG)

[TOC]

## 1.개요
---
 - ML 이론서 중 유명하고, 인기있는 그리고 비교적 입문과정인 이론서로 알고 있습니다. 원래는 ESL(elementary to statistical learning)이 원조이고 좀더 깊은 내용을 다루고 있는 것으로 알고 있는데, 배탈이 나지 않기 위해 물 탄 버전인 ISL부터 함께 공부하고자 합니다
 - 따라서 본 스터디는 ISL을 통하여 ML이론에 대해 발을 담궈보는것을 목표로 하고, 최종적으로는 ESL을 함께 공부하며 ML이론에 대한 기반을 탄탄하게 세워보자는데 목표가 있습니다. (방학 내에는 절대 다 못봅니다)
 - R을 통한 단순한 구현(사실 library..)코드가 함께 있는 것으로 알고 있는데, 좀더 intensive하게 나가자면 python을 통해 직접 구현해보는것? 등 세부적인 방향을 논의할 수 있습니다.
 1. 장점: 비교적 입문서 입니다. 수식도 깊게 쓰여져 있지 않고, 초심자를 대상으로 씌여졌습니다(고 들었습니다).
 2. 단점: ESL의 물탄 버젼 입니다. 제가 알기로 ESL이 원조이고, 해당 저서에서 깊은 부분과 수식을 몇개 쳐내고 쉽게 씌여진게 ISL이라고 합니다. 따라서 ISL->ESL의 먼 여정을 떠나야 합니다. 그런만큼 속도도 비교적 빠르게 나가기를 희망합니다.
 3. 단점2 : 물탄버젼임에도 불구하고 어려울 수 있습니다....따라서 빠른 진도로 인해 상당한 노력을 요할 수 있습니다. 또한 발제자가 아니더라도 모두가 해당 부분을 동일하게 공부하고, 충분히 공유한 후 발제가 이뤄지길 희망합니다.


## 2.순서
---
내용을 참고하기 쉽게, 세부목차까지 가져왔습니다. 총 10chapter입니다.

PREFACE VII
1 INTRODUCTION 1
2 STATISTICAL LEARNING 15
2.1 WHAT IS STATISTICAL LEARNING? . . . . . . . . . . . . . . . . . 15
2.2 ASSESSING MODEL ACCURACY . . . . . . . . . . . . . . . . . . . 29
2.3 LAB: INTRODUCTION TO R . . . . . . . . . . . . . . . . . . . . . 42
2.4 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 52
IX
X CONTENTS
3 LINEAR REGRESSION 59
3.1 SIMPLE LINEAR REGRESSION . . . . . . . . . . . . . . . . . . . 61
3.2 MULTIPLE LINEAR REGRESSION . . . . . . . . . . . . . . . . . . 71
3.3 OTHER CONSIDERATIONS IN THE REGRESSION MODEL . . . . . . . . 82
3.4 THE MARKETING PLAN . . . . . . . . . . . . . . . . . . . . . . 102
3.5 COMPARISON OF LINEAR REGRESSION WITH K-NEAREST
NEIGHBORS . . . . . . . . . . . . . . . . . . . . . . . . . . . . 104
3.6 LAB: LINEAR REGRESSION . . . . . . . . . . . . . . . . . . . . . 109
3.7 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 120
4 CLASSIFICATION 127
4.1 AN OVERVIEW OF CLASSIFICATION . . . . . . . . . . . . . . . . . 128
4.2 WHY NOT LINEAR REGRESSION? . . . . . . . . . . . . . . . . . 129
4.3 LOGISTIC REGRESSION . . . . . . . . . . . . . . . . . . . . . . . 130
4.4 LINEAR DISCRIMINANT ANALYSIS . . . . . . . . . . . . . . . . . 138
4.5 A COMPARISON OF CLASSIFICATION METHODS . . . . . . . . . . . 151
4.6 LAB: LOGISTIC REGRESSION, LDA, QDA, AND KNN . . . . . . 154
4.7 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 168
5 RESAMPLING METHODS 175
5.1 CROSS-VALIDATION . . . . . . . . . . . . . . . . . . . . . . . . 176
5.2 THE BOOTSTRAP . . . . . . . . . . . . . . . . . . . . . . . . . 187
5.3 LAB: CROSS-VALIDATION AND THE BOOTSTRAP . . . . . . . . . . . 190
5.4 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 197
6 LINEAR MODEL SELECTION AND REGULARIZATION 203
6.1 SUBSET SELECTION . . . . . . . . . . . . . . . . . . . . . . . . 205
6.2 SHRINKAGE METHODS . . . . . . . . . . . . . . . . . . . . . . . 214
6.3 DIMENSION REDUCTION METHODS . . . . . . . . . . . . . . . . 228
6.4 CONSIDERATIONS IN HIGH DIMENSIONS . . . . . . . . . . . . . . 238
6.5 LAB 1: SUBSET SELECTION METHODS . . . . . . . . . . . . . . . 244
XII CONTENTS
6.6 LAB 2: RIDGE REGRESSION AND THE LASSO . . . . . . . . . . . . 251
6.6.1 RIDGE REGRESSION . . . . . . . . . . . . . . . . . . . . 251
6.7 LAB 3: PCR AND PLS REGRESSION . . . . . . . . . . . . . . . 256
6.8 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 259
7 MOVING BEYOND LINEARITY 265
7.1 POLYNOMIAL REGRESSION . . . . . . . . . . . . . . . . . . . . . 266
7.2 STEP FUNCTIONS . . . . . . . . . . . . . . . . . . . . . . . . . 268
7.3 BASIS FUNCTIONS . . . . . . . . . . . . . . . . . . . . . . . . . 270
7.4 REGRESSION SPLINES . . . . . . . . . . . . . . . . . . . . . . . 271
7.5 SMOOTHING SPLINES . . . . . . . . . . . . . . . . . . . . . . . 277
7.6 LOCAL REGRESSION . . . . . . . . . . . . . . . . . . . . . . . . 280
7.7 GENERALIZED ADDITIVE MODELS . . . . . . . . . . . . . . . . . 282
7.8 LAB: NON-LINEAR MODELING . . . . . . . . . . . . . . . . . . . 287
7.9 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 297
8 TREE-BASED METHODS 303
8.1 THE BASICS OF DECISION TREES . . . . . . . . . . . . . . . . . 303
8.2 BAGGING, RANDOM FORESTS, BOOSTING . . . . . . . . . . . . . 316
8.3 LAB: DECISION TREES . . . . . . . . . . . . . . . . . . . . . . . 324
8.4 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 332
9 SUPPORT VECTOR MACHINES 337
9.1 MAXIMAL MARGIN CLASSIFIER . . . . . . . . . . . . . . . . . . . 338
9.2 SUPPORT VECTOR CLASSIFIERS . . . . . . . . . . . . . . . . . . . 344
9.3 SUPPORT VECTOR MACHINES . . . . . . . . . . . . . . . . . . . 349
9.4 SVMS WITH MORE THAN TWO CLASSES . . . . . . . . . . . . . . 355
9.5 RELATIONSHIP TO LOGISTIC REGRESSION . . . . . . . . . . . . . . 356
9.6 LAB: SUPPORT VECTOR MACHINES . . . . . . . . . . . . . . . . 359
9.7 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 368
10 UNSUPERVISED LEARNING 373
10.1 THE CHALLENGE OF UNSUPERVISED LEARNING . . . . . . . . . . . 373
10.2 PRINCIPAL COMPONENTS ANALYSIS . . . . . . . . . . . . . . . . 374
10.3 CLUSTERING METHODS . . . . . . . . . . . . . . . . . . . . . . . 385
10.4 LAB 1: PRINCIPAL COMPONENTS ANALYSIS . . . . . . . . . . . . 401
XIV CONTENTS
10.5 LAB 2: CLUSTERING . . . . . . . . . . . . . . . . . . . . . . . . 404
10.6 LAB 3: NCI60 DATA EXAMPLE . . . . . . . . . . . . . . . . . 407
10.7 EXERCISES . . . . . . . . . . . . . . . . . . . . . . . . . . . . 413

갈 길이 멉니다 하하..그러나 하나같이 도움되는 내용들입니다.
방학 목표는 ch.9까지 입니다. 1,2과는 사실상 intro이기에 아주 간략하게 다루고 넘어갈 것입니다.
 
##3. 스터디 진행
----

•1주차: 1,2장, 3장
•2주차: 3~4장
•3주차: 5~6장
•4주차: 7장
•5주차: 8장
•6주차: 9장
•7주차: 밀린 부분, 혹은 10장

 -- 한명씩 돌아가며 발제를 맡습니다. 그러나 발제는 말그대로 발제이고, 모두가 다함께 해당 부분을 공부합니다.
 -- 발제가 끝나면, 다함께 공유하자는 취지에서 적극적인 피드백을 합니다. 사실상 발제가 주가 아니라 토의가 주입니다.
 -- 여력이 된다면 좀더 수식적인 이해나 numpy등을 통한 구현도 생각할 수 있을 것 같습니다.

<예상되는 한계>
 - 상당히 긴 여정입니다. 게다가 양에 비해 속도도 빠르기 때문에 도중에 지치기 쉽습니다. 무엇보다 한 챕터라도 못읽었을 시 뒷부분을 이해하지 못하여 낙오될 가능성이 다분합니다.
 - 이론적 기반이 부족합니다(저는). 따라서 집단지성의 힘을 빌어 모두가 함께 생각하고, 구글링하고, 서로가 서로를 가르쳐주어야합니다. 공유지의 비극이 일어날 경우, 모두가 잘못된 이론을 이해하는 참사가 벌어질 수 있습니다.


####저는 ISL을 공부하고 싶은 사람일 뿐 매우 모르기에,  함께 공부하고싶은 분들의 의견을 함께 반영하여 수정하면 좋을것 같습니답


