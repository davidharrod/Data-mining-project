from sklearn import model_selection
from sklearn import ensemble
from sklearn import metrics
import pandas as pd
import data_utils
import matplotlib.pyplot as plt


def _calculate_auc(model, X, y):
    # Use test dataset
    y_score = model.predict_proba(X)[:, 1]
    fpr, tpr, threshold = metrics.roc_curve(y, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    return fpr, tpr, roc_auc


def _plot_confusion_matrix(model, X, y):
    y_pred = predict(model, X)
    confusion_matrix = metrics.confusion_matrix(y, y_pred)
    display = metrics.ConfusionMatrixDisplay(confusion_matrix)
    display.plot(include_values=True)
    plt.plot()
    plt.show()


def train(X, y):
    model = ensemble.GradientBoostingClassifier(random_state=10)
    model.fit(X, y)
    return model


def predict(model, X):
    return model.predict(X)


def test(model, X, y):
    y_pred = predict(model, X)
    print("模型的准确率为：\n", metrics.accuracy_score(y, y_pred))
    print("模型的评估报告：\n", metrics.classification_report(y, y_pred))
    fpr, tpr, roc_auc = _calculate_auc(model, X, y)
    _plot_confusion_matrix(model, X, y)
    plot_roc(fpr, tpr, roc_auc)
    rank_features(model, X)
    return None


def plot_roc(fpr, tpr, roc_auc):
    # 绘制面积图
    plt.stackplot(fpr, tpr, color='steelblue', alpha=0.5, edgecolor='black')
    # 添加边际线
    plt.plot(fpr, tpr, color='black', lw=1)
    # 添加对角线
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    # 添加文本信息
    plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % roc_auc)
    # 添加x轴与y轴标签
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    # 显示图形
    plt.show()


def rank_features(model, X):
    importance = pd.Series(model.feature_importances_, index=X.columns)
    importance.sort_values().plot(kind="barh")
    plt.show()


# Test codes.
# path = "./data/data.csv"
# X_train, X_test, y_train, y_test = data_utils.load_data(path)
# model = train(X_train, y_train)
# # Test on training set.
# test(model, X_train, y_train)
# # Test on testing set.
# test(model, X_test, y_test)
