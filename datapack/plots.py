import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from .dataset import Dataset

#Plots the ROC curve for the given dataset (dt) using the given numerical attribute (att) as predictor variable.
def plot_roc(dt, att):
    if isinstance(dt,Dataset):
        try:
            results = dt.fpr_tpr(att)
            auc = dt.roc_auc(att)
        except:
            raise NameError("Error while calculating the ROC curve.")
        FPR = results[0]
        TPR = results[1]
        fig, ax = plt.subplots()
        plt.plot(FPR, TPR, 'b')
        plt.title('ROC curve')
        plt.fill_between(FPR, TPR, color='blue', alpha=0.3)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.text(0.9, 0.1, "AUC:" + str(round(auc,2)), ha='right', va='bottom', transform=ax.transAxes, fontsize="x-large", fontweight="bold", color="darkblue")
        plt.show()
    else:
        raise NameError("First parameter must be a valid Dataset.")

#Plots the correlation matrix of the given dataset (dt).
def plot_correlation(dt,method="pearson"):
    if isinstance(dt,Dataset):
        try:
            result = dt.correlation_att(method=method)
        except:
            raise NameError("Error while calculating the pairwise correlation matrix.")
        cor = pd.DataFrame(result)
        mask = cor.isnull()
        ax = plt.axes()
        s = sns.heatmap(cor, linewidth=0.5, annot=True, cmap="RdBu",xticklabels=cor.columns,yticklabels=cor.columns,mask=mask,ax=ax)
        s.set_facecolor('lightgray')
        ax.set_title("Correlation matrix ("+method+")")
        plt.show()
    else:
        raise NameError("First parameter must be a valid Dataset.")

#Plots the normalized mutual information matrix of the given dataset (dt).
def plot_norm_mutual_info(dt):
    if isinstance(dt,Dataset):
        try:
            result = dt.norm_mutual_info_att()
        except:
            raise NameError("Error while calculating the pairwise normalized mutual information matrix.")
        cor = pd.DataFrame(result)
        mask = cor.isnull()
        ax = plt.axes()
        s = sns.heatmap(cor, linewidth=0.5, annot=True, cmap="RdBu",xticklabels=cor.columns,yticklabels=cor.columns,mask=mask, ax=ax)
        s.set_facecolor('lightgray')
        ax.set_title('Normalized Mutual Information matrix')
        plt.show()
    else:
        raise NameError("First parameter must be a valid Dataset.")
