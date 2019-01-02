# My Takeaway of these blogs

## 1. Understanding Hinton’s Capsule Networks. Part I: Intuition.

https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b

`1. CNN's drawbacks`

<img src="https://ws1.sinaimg.cn/large/006tNbRwly1fys4odplahj310y0u0n2n.jpg" width="250px"/>

As this author said, the limit of CNN is orientational and relative spatial relationships between elements.

max pooling is losing information.

Internal data representation of a convolutional neural network does not take into account important spatial hierarchies between simple and complex objects.

Hinton argues that brains, in fact, do the opposite of rendering. He calls it inverse graphics: from visual information received by eyes, they deconstruct a hierarchical representation of the world around us and try to match it with already learned patterns and relationships stored in the brain. This is how recognition happens. And the key idea is that representation of objects in the brain does not depend on view angle.

When these relationships are built into internal representation of data, it becomes very easy for a model to understand that the thing that it sees is just another view of something that it has seen before.

“dynamic routing between capsules”.



---

## 2. CapsNet

https://www.youtube.com/watch?v=pPN8d0E3900
