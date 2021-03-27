## Task Description

Text Sentiment Classification

- 本次作业为 Twitter 上收集到的推文，每则推文都会被标注为正面或负面
- 除了 labeled data 以外，我们还额外提供了 120 万笔左右的 unlabeled data
- labeled training data ：20万
- unlabeled training data ：120万
- testing data ：20万（10 万 public，10 万 private）

- 利用 Word Embedding 来代表每一个单字，并由 RNN model 得到一个代表该句的 vector
- 或可直接用 bag of words (BOW) 的方式获得代表该句的 vector

- 1-of-N encoding Issue 
  - 缺少字与字之间的关联性 (当然你可以相信 NN 很强大他会自己想办法)
  - 很吃内存，200000(data)\*30(length)*20000(vocab size) \*4(Byte) = 4.8\*10^11 = 480 GB

- 用一些方法 pretrain 出 word embedding (e.g., skip-gram, CBOW. )然后跟 model 的其他部分一起 train

- Self-Training: 把 train 好的 model 对 unlabel data 做预测，并将这些预测后的值转成该笔unlabel data 的 label，并加入这些新的 data 做 training。你可以调整不同的 threshold，或是多次取样来得到比较有信心的 data。e.g., 设定 pos_threshold = 0.8，只有 prediction > 0.8 的 data 会被标上 1


Kaggle competition: https://www.kaggle.com/c/ml2020spring-hw4

##  Report

1. (1%) 请说明你实作的 RNN 的模型架构、word embedding 方法、训练过程 (learning curve) 和准确率为何？ (尽量是过 public strong baseline 的 model)
2. (2%) 请比较 BOW + DNN 与 RNN 两种不同 model 对于 "today is a good day, but it is hot" 与 "today is hot, but it is a good day" 这两句的分数 (过 softmax 后的数值)，并讨论造成差异的原因。 
3. (1%) 请叙述你如何 improve performance（preprocess、embedding、架构等等），并解释为何这些做法可以使模型进步，并列出准确率与 improve 前的差异。（semi-supervised 的部分请在下题回答）
4. (2%) 请描述你的semi-supervised方法是如何标记label，并比较有无semi-supervised training对准确率的影响并试着探讨原因（因为 semi-supervise learning 在 labeled training data 数量较少时，比较能够发挥作用，所以在实作本题时，建议把有 label 的training data从 20 万笔减少到 2 万笔以下，在这样的实验设定下，比较容易观察到semi-supervise learning所带来的帮助）

## Kaggle


