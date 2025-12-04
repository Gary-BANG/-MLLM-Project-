### Chapter One
**NLP**: Natural Language Processing  
**CWS**: Chinese Word Segmentation  
**POS Tagging**: Part-Of-Speech Tagging 词性标注  
**NER**: Name Entity Recognition 实体识别  
**Text Summarization**: 文本识别  
**MT**: Machine Translation 机器翻译  
**VSM**: Vector Space Model 向量空间模型  
**N-gram**: a model to estimate the probability of the nth word based on the former (n-1)th words
**Word2Vec**: one of the Word Embedding technologies  
`    `**CBOW**: Continuous Bag of Words, estimate the most possible words based on the VSM of	words above and below  
`    `**Skip-Gram**: Estimate the VSM of words above and below based on the word you choose
**ELMo**: Embedding from Language Models  
**Word Embedding**: 词向量  

### Chapter Two
**AM**: Attention Mechanism 注意力机制  
**FNN**: Feedforward Neural Network 全连接神经网络  
**CNN**: Convolutional Neural Network 卷积神经网络  
**RNN**: Recurrent Neural Network 循环神经网络  
Three Key Elements: **Query** 查询值, **Key** 键值, **Value** 真值  
###### Example:   
` `We have a dictionary:
```
{
    "apple":10,
    "banana":5,
    "chair":2
}
```
` `if we look for **"apple"**, then:  
`  `**Query** = "apple"  
`  `**Key** = "apple"  
`  `**Value** = 10  
` `if we look for **"fruit"**, then there is a **soft matching 软匹配**:  
We give each key a **matching weight 匹配权重**:  
```
{
    "apple":0.6,
    "banana":0.4,
    "chair":0
}
```
`  `That means you think apple is more like "fruit" and chair is not related to "fruit".  
`  `**Query** = "fruit"  
`  `**Key** = "apple" "banana" "chair"  
` `Thus, we can get the value:  
$$
value = 0.6*10+0.4*5+0*2=8
$$
` `We give each key a **Attention Weight** (i.e. 0.6, 0.4, 0), but how can we obtain the **Attention Weight**?  
`  `Actually, we have **Word Embedding** to represent the meaning of the word and we can use dot product to measure the relativity of two words:  
$$
{
    v \cdot w = \sum_i v_i w_i
}
$$
`  `If the result is larger than 0, then have similar meaning. Otherwise, they do not have similar meaning. Thus, assume the **Query** is "fruit", whose **Word Embedding** is $q$ and our corresponding **Word Embeddings** of **Key** is $K=[v_{apple}, v_{banana}, v_{chair}]$. And we can estimate the similiarity of **Query** and each **key**:  
$$
{
    x = q K^T \Rightarrow softmax(x)_i = \frac {e^{x_i}} {\sum_j e^{x_j}}
}
$$  
` `Thus the total formula for **Attention Mechanism** is:  
$$
1.\ One\ query: \ \ attention(q,K,v)=softmax(qK^T)v\\  
2.\ Many\ queries:\ \ attention(q,K,V)=softmax(QK^T)V  
$$
` `【注】If the dimension of **Query** and **Key** (i.e. $d_k$) is large, then their dot product will be large as well. After softmaxing 归一化: The differences will be expanded and there exists many values that are close to 0 and 1. This will cause the gradients unstable and harm the training stability.    --梯度会变得非常不稳定（训练会震荡或不收敛）
```
'''
注意力计算函数
Args:
    query: 查询矩阵 Q
    key:   键矩阵 K
    value: 值矩阵 V
    dropout: dropout 层（可选）
    query的形状通常是：(batch_size, num_heads, seq_len_q, d_k)
    key的形状通常是： (batch_size, num_heads, seq_len_k, d_k)
'''
def attention(query, key, value, dropout=None):
    # 获取 Key 的向量维度 d_k
    # d_k = Q 和 K 的最后一维大小
    d_k = query.size(-1) # 获取Q最后一维的大小
    
    # QK^T，计算 Query 和 Key 的相似度矩阵
    # key.transpose(-2, -1) 表示转置矩阵K
    # torch.matual 表示矩阵乘法
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 对相似度进行 softmax，得到注意力权重
    p_attn = scores.softmax(dim=-1)

    # 如果使用 dropout，就对注意力结果进行 dropout
    if dropout is not None:
        p_attn = dropout(p_attn)

    # 用注意力权重对 Value 做加权求和
    return torch.matmul(p_attn, value), p_attn
```