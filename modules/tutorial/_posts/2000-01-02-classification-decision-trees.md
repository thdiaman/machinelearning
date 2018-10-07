---
title: Classification with Decision Trees
published: true
---

# Classification with Decision Trees

## Algorithm Description
Check <a target="_blank" href="{{site.baseurl}}/presentations/DecisionTrees.pdf">here</a>
for a presentation on the algorithm and the relevant info.

## Example 1 - Categorical Data
### Dataset
This example uses the tennis dataset, which can be downloaded
<a target="_blank" href="{{site.dataurl}}/DecisionTrees/data.csv">here</a>.
Save the dataset in a folder and let's start coding.

### Solution
Click <a target="_blank" href="{{site.dataurl}}/DecisionTrees/decision_tree.py">here</a>
to download the solution to this example problem.

## Example 2 - Numerical Data
### Dataset
The dataset of this example can be downloaded
<a target="_blank" href="{{site.dataurl}}/DecisionTrees/treedata.csv">here</a>.
Save the dataset in a folder and let's start coding.

### Solution
Click <a target="_blank" href="{{site.dataurl}}/DecisionTrees/decision_tree_full.py">here</a>
to download the solution to this example problem.

## Exercise
Can we perform the same analysis for the iris dataset? 

The iris dataset can be loaded as a pandas dataframe using the commands

```
iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
```

To make the problem harder, use only the first two columns of the dataset.
Try this before you check the solution
(<a target="_blank" href="{{site.dataurl}}/DecisionTrees/decision_tree_exercise.py">here</a>)
