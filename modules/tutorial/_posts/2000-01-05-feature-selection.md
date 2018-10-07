---
title: Feature Selection
published: true
---

# Feature Selection

## Algorithm Description
Check <a target="_blank" href="{{site.baseurl}}/presentations/FeatureSelection.pdf">here</a>
for a presentation on the algorithm and the relevant info.

## Example 1 - Feature Selection using Correlation
### Dataset
The dataset of this example is the iris dataset, which can be
loaded as a pandas dataframe using the commands:

```
iris = datasets.load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])
```

### Solution
Click <a target="_blank" href="{{site.dataurl}}/FeatureSelection/feature_selection.py">here</a>
to download the solution to this example problem.

## Example 2 - Feature Selection using Mutual Information
### Dataset
The dataset of this example is the iris dataset, which can be loaded as above.

### Solution
Click <a target="_blank" href="{{site.dataurl}}/FeatureSelection/feature_selection_mi.py">here</a>
to download the solution to this example problem.

## Exercise
Can we perform the same analysis using the chi squared statistic?
Apply it again on the iris dataset, which can be loaded as above.

Try this before you check the solution
(<a target="_blank" href="{{site.dataurl}}/FeatureSelection/feature_selection_chi.py">here</a>).
