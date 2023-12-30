# adaptive-trading-agent

A challenging problem in computational intelligence lies in effectively designing models that have no predefined output label. 

The market is one of these situations. It has a clear goal (make line go up) and clear input data (previous price/volume data), but there is no clear strategy that is the best.

Most (naive) trading strategies simplify their algorithms relying on a predict-the-next-price structure. In this situation, there is a clear label (the actual next price) that the model can be trained to predict. However, these strategies are incapable of beating out a simple linear regression, and it should be obvious that this strategy will have mediocre results.

More generally, any method that is predefined by the human can easily be transferred to a computational agent by labelling the data in the way the human wants it to be. Essentially telling the agent, "trade like this." But this clearly is limited to the imagination and intellect of its human.

So then the question becomes, how can an artificial system come up with its own trading system where it is unclear to the user what the best strategy is? In this repo we will explore different methods of answering this question.