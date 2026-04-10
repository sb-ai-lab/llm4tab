from sklearn import metrics


def score(pipeline, X_test, y_test, score="f1"):
    y_pred = pipeline.predict(X_test)

    if score == "acc":
        return score_acc(y_test, y_pred)

    elif score == "f1":
        return score_f1(y_test, y_pred)

def score_acc(y_test, y_pred):
    return metrics.balanced_accuracy_score(y_test, y_pred)


def score_f1(y_test, y_pred):
    return metrics.f1_score(y_test, y_pred, average="macro")

def score_roc_auc(y_test, y_pred):
    return metrics.roc_auc_score(y_test, y_pred)