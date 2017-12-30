import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.metrics import roc_auc_score


def seaborn_scatterplot_df(df):
    '''
    Super simple function that calls up the sns pairplot using only a
    small amount of the data.
    Creates a scatterplot using the seaborn style

    parameters:
    --------------------------------
    df : pandas dataframe to create plot of

    returns:
    --------------------------------
    none
    '''
    sns.set(style="ticks")
    sns.pairplot(df.head(5000))
    plt.show()


def create_corr_mat(df, title):
    """
    Takes in a dataframe and creates a correlation matrix, great for
    visuallizing highly correlated vars.

    parameters:
    --------------------------------
    df : dataframe to create a matrix of
    title : title for the plot

    returns:
    --------------------------------
    none
    """
    sns.set(style="white")

    # Generate a large random dataset
    # Compute the correlation matrix
    corr = df.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.title(title)
    plt.show()


def Violin_Plot(df, scaled=False):
    """
    Takes in a dataframe and creates a violin plot of all features.

    parameters:
    --------------------------------
    df : dataframe of features to create violin plots for
    scaled : If true the y-axis will cut off at +/- 5 std-deviations

    returns:
    --------------------------------
    none
    """

    fig, axes = plt.subplots(figsize=(11, 9))
    sns.violinplot(data=df, ax=axes, scale='width', palette="Set3",
                   cut=0, fontsize=12)
    axes.set_title('Violin Plot')
    axes.yaxis.grid(True)
    for tick in axes.get_xticklabels():
        tick.set_rotation(45)
        tick.set_horizontalalignment("right")
    axes.set_xlabel('Features')
    axes.set_ylabel('Range')
    if scaled:
        axes.set_ylim(-5, 5)

    plt.show()


def scree_plot(pca, title=None):
    """
    Creates a scree_plot from a fitted pca model

    parameters:
    --------------------------------
    pca : fitted pca model
    title : title if desired for plot

    returns:
    --------------------------------
    none
    """
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(5, 3), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
           color=[(0.949, 0.718, 0.004),
                  (0.898, 0.49, 0.016),
                  (0.863, 0, 0.188),
                  (0.694, 0, 0.345),
                  (0.486, 0.216, 0.541),
                  (0.204, 0.396, 0.667),
                  (0.035, 0.635, 0.459),
                  (0.486, 0.722, 0.329),
                  ])

    for i in range(num_components):
        ax.annotate(r"%s%%" % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]),
                    va="bottom", ha="center", fontsize=6)
    ax.set_xticklabels(ind, fontsize=6)
    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=0)

    ax.set_xlabel("Principal Component", fontsize=6)
    ax.set_ylabel("Variance Explained (%)", fontsize=6)

    if title is not None:
        plt.title(title, fontsize=6)


def plot_embedding(X, y, title=None):
    """
    Creates a pyplot object showing digits projected onto 2-dimensional
    feature space. PCA should be performed on the feature matrix before
    passing it to plot_embedding.

    parameters:
    --------------------------------
    X : decomposed feature matrix
    y : target labels (digits)
    title : title for plot if desired

    returns:
    --------------------------------
    none
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(5, 3), dpi=250)
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(int(y[i])),
                 color=plt.cm.Set1(int(y[i])),
                 fontdict={'weight': 'bold', 'size': 6})
        plt.xticks([]), plt.yticks([])
    plt.ylim([-0.1, 1.1])
    plt.xlim([-0.1, 1.1])

    if title is not None:
        plt.title(title, fontsize=16)


def partial_dependency_plots(X, y, classifier):
    """
    Creates partial dependency plots for all features in dataframe X.

    parameters:
    --------------------------------
    X : Pandas dataframe
    y : target labels (digits)
    classifier : a scikit learn classifier such as random forest

    returns:
    --------------------------------
    none
    """
    names = X.columns.values
    features = list(range(0, len(names)))
    gbrt = classifier.fit(X, y)
    fig, axs = plot_partial_dependence(gbrt, X, features, names, n_jobs=-1,
                                       grid_resolution=50, figsize=(18, 24))
    plt.show()


def roc_curve(X, y, classifier_list):
    """
    Creates ROC curves for all curves in the classifier list,
    reports AUC in legend.

    parameters:
    --------------------------------
    X : Pandas dataframe
    y : target labels (digits)
    classifier_list : a list of scikit learn classifiers such as random forest

    returns:
    --------------------------------
    none
    """
    for classifier in classifier_list:
        classifier.fit(X, y)
        probabilities = classifier.predict_proba(X)[:, 1]
        thresholds = np.sort(probabilities)
        tprs = []
        fprs = []

        num_positive_cases = sum(y)
        num_negative_cases = len(y) - num_positive_cases
        i = 0
        for threshold in thresholds:
            # With this threshold, give the prediction of each instance
            predicted_positive = probabilities >= threshold

            # Calculate the number of correctly predicted positive cases
            true_positives = np.sum(predicted_positive * y)

            # Calculate the number of incorrectly predicted positive cases
            false_positives = np.sum(predicted_positive) - true_positives

            # Calculate the True Positive Rate
            tpr = true_positives / float(num_positive_cases)

            # Calculate the False Positive Rate
            fpr = false_positives / float(num_negative_cases)

            fprs.append(fpr)
            tprs.append(tpr)
            i += 1
        AUC = roc_auc_score(y, probabilities)
        plt.plot(fprs, tprs, label="""{0}, AUC = {1:0.2f}"""
                 .format(classifier.__class__.__name__, AUC))
    plt.plot([0, 1], [0, 1], color='r')
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Plot")


def feature_importance_plot(X, y, classifier):
    """
    Creates a horizntile feature importance plot based on the classifier. The
    classifir should be a scikit learn tree method.

    parameters:
    --------------------------------
    X : Pandas dataframe
    y : target labels (digits)
    classifier : a scikit learn classifier such as random forest

    returns:
    --------------------------------
    none
    """
    classifier.fit(X, y)
    feature_importance = classifier.feature_importances_
    feature_importance = 100.0 * (feature_importance /
                                  feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X.columns[sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()
