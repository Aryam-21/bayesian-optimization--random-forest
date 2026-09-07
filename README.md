
# Bayesian Optimization vs Random Search for Random Forest

## Overview

This project investigates a practical question in hyperparameter optimization:

> **Which method finds a strong Random Forest configuration sooner: Random Search or Bayesian Optimization?**

The experiment uses the **Breast Cancer Wisconsin dataset** and a `RandomForestClassifier`. Two hyperparameter optimization strategies are compared under the same experimental conditions:

- Random Search
- Bayesian Optimization

Both methods are given the same search space, training data, cross-validation procedure, evaluation metric, random seed, and optimization budget.

The goal is not only to identify the best-performing configuration, but also to study **how quickly each optimization method discovers a strong configuration**.

---

## Objectives

The experiment has four main objectives:

1. Establish a baseline Random Forest model using default hyperparameters.
2. Optimize Random Forest hyperparameters using Random Search.
3. Optimize the same hyperparameters using Bayesian Optimization.
4. Compare the two methods based on:
   - Cross-validation accuracy
   - Test accuracy
   - Runtime
   - Convergence behavior

The central question is:

> **Which optimization strategy reaches a strong configuration sooner?**

---

## Dataset

The experiment uses the **Breast Cancer Wisconsin classification dataset** provided through `scikit-learn`.

Dataset characteristics:

| Property | Value |
|---|---:|
| Number of samples | 569 |
| Number of features | 30 |
| Number of classes | 2 |
| Training samples | 455 |
| Test samples | 114 |

The data is divided using a **stratified 80/20 train-test split**.

The test set is kept completely separate during hyperparameter optimization.

### Data Split

```text
Full Dataset
    │
    ├── 80% Training Data
    │       │
    │       └── 5-Fold Cross-Validation
    │              └── Hyperparameter Optimization
    │
    └── 20% Test Data
            │
            └── Final evaluation only
# Results Feedback

The experimental results show a clear difference between the two optimization strategies.

## 1. Improvement over the Baseline

The baseline Random Forest achieved a 5-fold cross-validation accuracy of **0.9538**.

Both optimization methods improved upon this baseline:

- Random Search increased CV accuracy to **0.9604**.
- Bayesian Optimization increased CV accuracy to **0.9626**.

This represents an improvement of approximately:

| Method | CV Accuracy | Improvement over Baseline |
|---|---:|---:|
| Baseline | 0.9538 | — |
| Random Search | 0.9604 | +0.0066 |
| Bayesian Optimization | 0.9626 | +0.0088 |

Therefore, Bayesian Optimization found the configuration with the highest estimated cross-validation performance.

---

## 2. Convergence Feedback

The convergence plot provides the most important evidence for answering the main question of this experiment: **which method finds a strong configuration sooner?**

Bayesian Optimization reached its final best CV accuracy of **0.9626 by evaluation 3**. After this point, the best-so-far score remained approximately unchanged.

Random Search reached its final best CV accuracy of **0.9604 by evaluation 9** and also remained unchanged for the remaining evaluations.

| Method | Evaluation at Final Best | Final Best CV Accuracy |
|---|---:|---:|
| Random Search | 9 | 0.9604 |
| Bayesian Optimization | 3 | **0.9626** |

This suggests that Bayesian Optimization was more efficient **in terms of the number of evaluations required to find a strong configuration**.

The behavior is consistent with the purpose of Bayesian optimization: previous evaluations are used to guide subsequent evaluations toward promising regions instead of selecting configurations independently.

---

## 3. Runtime Feedback

The runtime results tell a different story.

| Method | Runtime |
|---|---:|
| Random Search | **77.82 s** |
| Bayesian Optimization | 98.66 s |

Random Search completed all 20 evaluations approximately **20.84 seconds faster** than Bayesian Optimization.

Therefore, the word "efficient" needs to be interpreted carefully.

- Bayesian Optimization was more efficient in **number of evaluations**.
- Random Search was more efficient in **total wall-clock runtime**.
- Bayesian Optimization achieved the **highest CV accuracy**.

This demonstrates why convergence and runtime should both be considered rather than relying on only one measure.

---

## 4. Test Set Feedback

The final test results were:

| Method | Best CV Accuracy | Test Accuracy |
|---|---:|---:|
| Baseline | 0.9538 | 0.9561 |
| Random Search | 0.9604 | **0.9561** |
| Bayesian Optimization | **0.9626** | 0.9474 |

Random Search matched the baseline test accuracy at **0.9561**.

Bayesian Optimization achieved the highest CV accuracy, but its test accuracy was **0.9474**, which was lower than both the baseline and Random Search.


---

# Overall Interpretation

The results provide evidence that Bayesian Optimization was able to locate a stronger configuration with fewer evaluations.

The Bayesian optimizer reached a CV accuracy of **0.9626 after only 3 evaluations**, while Random Search required approximately **9 evaluations** to reach its final best score of **0.9604**.

However, Bayesian Optimization required more time to complete the full 20-evaluation budget and produced lower test accuracy in this particular experiment.

Therefore, the results should not be interpreted as showing that one optimization method is universally superior. Instead, they demonstrate a trade-off:

> **Bayesian Optimization was more efficient in terms of evaluations and achieved the highest CV score, while Random Search was faster in total runtime and achieved higher test accuracy in this particular run.**

---

# Conclusion

This experiment compared Random Search and Bayesian Optimization for tuning a Random Forest classifier on the Breast Cancer Wisconsin dataset. Both methods were evaluated under the same 80/20 stratified train-test split, 5-fold cross-validation, accuracy metric, random seed, search space, and 20-evaluation budget.

The results show that Bayesian Optimization found the strongest cross-validation configuration, achieving **0.9626** compared with **0.9604** for Random Search. More importantly, the convergence results show that Bayesian Optimization reached its final best configuration by approximately **evaluation 3**, while Random Search reached its final best at approximately **evaluation 9**. Based on the number of evaluations required to find a strong configuration, Bayesian Optimization was more efficient in this experiment.

However, Random Search completed the full optimization faster (**77.82 s** versus **98.66 s**) and achieved higher test accuracy (**0.9561** versus **0.9474** for Bayesian Optimization).

Overall, the experiment suggests that Bayesian Optimization can use previous evaluations to reach promising configurations with fewer trials, but it does not prove that Bayesian Optimization is universally better than Random Search. The conclusion is limited to this dataset, model, search space, evaluation budget, and experimental setup. Multiple datasets, repeated runs, or different budgets would be needed to make a broader claim.