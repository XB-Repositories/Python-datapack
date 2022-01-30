import pandas as pd
import numpy as np
from IPython.display import display
from .attributes import Attribute, Numerical, Boolean, String, Categorical

### DATASET CLASS. The individual attributes of the data set are collected in a dictionary.
class Dataset():
    
    ###################CONSTRUCTOR FUNCTION####################
    def __init__(self,*args):
        self.attributes = None
        self.att_class = None
        self.length = 0
        if len(args)==1 or len(args)==2:
            if isinstance(args[0],pd.DataFrame):
                self.attributes = dict()
                for key in args[0].columns:
                    value = args[0][key]
                    if not isinstance(value, pd.Series):
                        self.attributes = None
                        raise NameError("Input pd.DataFrame can't have duplicated column names.")
                    key = str(key)
                    if value.dtype == int or value.dtype == float:
                        self.attributes[key] = Numerical(value)
                    elif value.dtype == bool:
                        self.attributes[key] = Boolean(value)
                    else:
                        try:
                            self.attributes[key] = String(value)
                        except:
                            self.attributes = None
                            raise NameError("Attribute values must be numerical, bool, str. Each attribute can only contain one data type.")
            else:
                raise NameError("Attributes must be collected in a pd.DataFrame, where each column represents a different attribute.")
            if len(args)==2:
                name = str(args[1])
                if name in self.attributes:
                    self.att_class = name
                else:
                    raise NameError("Class name must match an existing attribute name.")
            self.length = len(args[0].index)
        elif len(args) != 0:
            raise NameError("Incorrect number of parameters.")

    ###################SETTERS####################
    
    #Initializes the data set according to the input pd.DataFrame (att).
    #The name of the class variable can be specified through a parameter (c).
    def set_data(self,att,c=None):
        self.attributes = None
        self.att_class = None
        self.length = 0
        if isinstance(att,pd.DataFrame):
            self.attributes = dict()
            for key in att.columns:
                value = att[key]
                if not isinstance(value, pd.Series):
                    self.attributes = None
                    raise NameError("Input pd.DataFrame can't have duplicated column names.")
                key = str(key)
                if value.dtype == int or value.dtype == float:
                    self.attributes[key] = Numerical(value)
                elif value.dtype == bool:
                    self.attributes[key] = Boolean(value)
                else:
                    try:
                        self.attributes[key] = String(value)
                    except:
                        self.attributes = None
                        raise NameError("Attribute values must be numerical, bool, str. Each attribute can only contain one data type.")
        else:
            raise NameError("Attributes must be collected in a pd.DataFrame, where each column represents a different attribute.")
        if c!=None:
            name = str(c)
            if name in self.attributes:
                self.att_class = name
            else:
                raise NameError("Class name must match an existing attribute name.")
        self.length = len(att.index)

    #Initialize only the class name (c).
    def set_class(self,c):
        name = str(c)
        if name in self.attributes:
            self.att_class = name
        else:
            raise NameError("Class name must match an existing attribute name.")

   ###################GETTERS####################

    #Returns the specified attribute (att).
    def get_attribute(self,att):
        name = str(att)
        if name in self.attributes.keys():
            return self.attributes[name]
        else:
            raise NameError("Attribute not found.")

    #Returns the name of the class attribute.
    def get_class_name(self):
        return self.att_class

    #Returns the number of instances in the dataset.
    def get_number_instances(self):
        return self.length

    #############################################

    #Updates the specified attribute (att) of the instance in the specified index (ind).
    def update_instance(self,ind,att,val):
        name = str(att)
        if name in self.attributes.keys():
            self.attributes[att].update_value(ind,val)
        else:
            raise NameError("Attribute not found.")

    #Adds a single attribute (key: att) to an already existing dataset.
    def add_attribute(self,key,att):
        if isinstance(att,(np.ndarray,list)):
            att = pd.Series(att)
        if not isinstance(att,pd.Series):
            raise NameError("The attribute data must be a pd.Series, a np.array or a list.")
        if self.attributes==None:
            raise NameError("The dataset is not initialized yet. Used the set_data function.")
        if key in self.attributes.keys():
            raise NameError("The attribute already exists.")
        if self.length != len(att):
            raise NameError("The length of the new attribute must be the same as the number of instances in the dataset.")
        if att.dtype == int or att.dtype == float:
            self.attributes[key] = Numerical(att)
        elif att.dtype == bool:
            self.attributes[key] = Boolean(att)
        else:
            try:
                self.attributes[key] = String(att)
            except:
                raise NameError("Attribute values must be numerical, bool, str. The attribute can only contain one data type.")

    #Removes an attribute (att) from an already existing dataset.
    def remove_attribute(self,att):
        name = str(att)
        if name not in self.attributes.keys():
            raise NameError("Attribute not found.")
        del self.attributes[name]
        if name == self.att_class:
            self.att_class = None
        if len(self.attributes)==0:
            self.length = 0
        

    #Converts an attribute (att) to categorical.
    #The possible values of the categorical attribute can be specified through a parameter (values).
    def to_categorical_attribute(self,att,values=None):
        if att in self.attributes.keys():
            if(not isinstance(self.attributes[att],Numerical)):
                self.attributes[att] = self.attributes[att].to_categorical(values=values)
            else:
                raise NameError("Numerical attributes can't be converted to categorical.")
        else:
            raise NameError("Attribute not found.")
    
    #Reads a csv file and stores the data in the data set.
    def from_csv(self,file,header=True,sep=",",c=None):
        try:
            if header:
                df = pd.read_csv(file,header=0,sep=sep)
            else:
                df = pd.read_csv(file,header=None,sep=sep)
        except:
            raise NameError("Csv file can't be read with the given parameters. Format: input file (string), header (boolean), separator (string), class attribute name.")
        self.set_data(df,c)
    
    #Writes the data set into a csv file.
    def to_csv(self,file,header=True,sep=","):
        df = pd.DataFrame(dict(zip([key for key,_ in self.attributes.items()],[value.data for _,value in self.attributes.items()])))
        try:
            df.to_csv(file,index=False,header=header,sep=sep)
        except:
            raise NameError("Csv file can't be written with the given parameters. Format: output file (string), header (boolean), separator (string).")

    #Converts the dataset object into an equivalent dataframe.
    def to_dataframe(self):
        return pd.DataFrame({key: list(value.data) for key,value in self.attributes.items()})
    
    #Prints the data set in a readable format.
    def print_dataset(self):
        display(self.to_dataframe())
        print("Dataset class: " + str(self.att_class))

    #Returns the mean of the specified attribute (att).
    def mean(self,att):
        name = str(att)
        if name in self.attributes.keys():
            if(isinstance(self.attributes[att],Numerical)):
                return self.attributes[name].mean()
            else:
                raise NameError("Can't compute the mean of a non-numerical attribute.")
        else:
            raise NameError("Attribute not found.")

    #Returns the means of all the numerical attributes in the data set.
    def mean_att(self):
        return dict(zip([key for key, _ in self.attributes.items()],[value.mean() if isinstance(value,Numerical) else np.NaN for _ , value in self.attributes.items()]))

    #Returns the median of the specified attribute (att).
    def median(self,att):
        name = str(att)
        if name in self.attributes.keys():
            if(isinstance(self.attributes[att],Numerical)):
                return self.attributes[name].median()
            else:
                raise NameError("Can't compute the median of a non-numerical attribute.")
        else:
            raise NameError("Attribute not found.")

    #Returns the medians of all the numerical attributes in the data set.
    def median_att(self):
        return dict(zip([key for key, _ in self.attributes.items()],[value.median() if isinstance(value,Numerical) else np.NaN for _ , value in self.attributes.items()]))

    #Returns the variance of the specified attribute (att).
    def variance(self,att):
        name = str(att)
        if name in self.attributes.keys():
            if(isinstance(self.attributes[att],Numerical)):
                return self.attributes[name].variance()
            else:
                raise NameError("Can't compute the variance of a non-numerical attribute.")
        else:
            raise NameError("Attribute not found.")

    #Returns the variances of all the numerical attributes in the data set.
    def variance_att(self):
        return dict(zip([key for key, _ in self.attributes.items()],[value.variance() if isinstance(value,Numerical) else np.NaN for _ , value in self.attributes.items()]))

    #Returns the mode of the specified attribute (att).
    def mode(self,att):
        name = str(att)
        if name in self.attributes.keys():
            if(not isinstance(self.attributes[att],Numerical)):
                return self.attributes[name].mode()
            else:
                raise NameError("Can't compute the mode of a numerical attribute.")
        else:
            raise NameError("Attribute not found.")

    #Return the modes of all the non-numerical attributes in the data set.
    def mode_att(self):
        return dict(zip([key for key, _ in self.attributes.items()],[value.mode() if isinstance(value,(Boolean,String,Categorical)) else np.NaN for _ , value in self.attributes.items()]))

    #Returns the entropy of the specified attribute (att).
    def entropy(self,att):
        name = str(att)
        if name in self.attributes.keys():
            if(not isinstance(self.attributes[att],Numerical)):
                return self.attributes[name].entropy()
            else:
                raise NameError("Can't compute the entropy of a numerical attribute.")
        else:
            raise NameError("Attribute not found.")
    
    #Return the entropies of all the non-numerical attributes in the data set.
    def entropy_att(self):
        return dict(zip([key for key, _ in self.attributes.items()],[value.entropy() if isinstance(value,(Boolean,String,Categorical)) else np.NaN for _ , value in self.attributes.items()]))
    
    #Returns the list of TPR and FPR values obtained when using the given numerical attribute (att) to predict the value of the boolean class variable.
    #This function is used to compute the ROC curve.
    def fpr_tpr(self,att):
        if self.att_class == None:
            raise NameError("Class attribute not specified.")
        elif len(set(self.attributes[self.att_class].data)) == 1:
            raise NameError("The ROC curve is not defined when the class attribute contains only one class.")
        if isinstance(self.attributes[self.att_class],Boolean):
            name = str(att)
            if name in self.attributes and isinstance(self.attributes[name],Numerical):
                df = pd.DataFrame.from_dict({name:self.attributes[name].data,self.att_class:self.attributes[self.att_class].data})
                df_sorted = df.sort_values(name,ascending=True).reset_index(drop=True)
                length = len(df_sorted)
                cut_points = list(df_sorted[name].searchsorted(df_sorted[name].unique()))+[length]
                TP = [len(df_sorted[(df_sorted.index >= i) & (df_sorted[self.att_class]==True)]) for i in cut_points]
                TN = [len(df_sorted[(df_sorted.index < i) & (df_sorted[self.att_class]==False)]) for i in cut_points]
                TPR = [TP[index]/(TP[index]+(i-TN[index])) for index,i in enumerate(cut_points)]
                FPR = [((length-i)-TP[index])/(((length-i)-TP[index])+TN[index]) for index,i in enumerate(cut_points)]
                return (FPR,TPR)
            else:
                raise NameError("Only existing numerical attributes can be predictor variables.")
        else:
            raise NameError("Class must be boolean.")
        
    #Returns the AUC score that is obtained when using the given numerical attribute (att) to predict the value of the boolean class variable.
    def roc_auc(self,att):
        try:
            results = self.fpr_tpr(att)
        except:
            raise NameError("Error when computing the ROC curve.")
        FPR = results[0]
        TPR = results[1]
        return sum([(FPR[i]-FPR[i+1])*((TPR[i]-TPR[i+1])/2+TPR[i+1]) for i in range(len(TPR)-1)])
    
    #Returns the AUC scores that are obtained when using each of the numerical attributes in the data set to predict the value of the boolean class variable.
    def roc_auc_att(self):
        if self.att_class == None:
            raise NameError("Class attribute not specified.")
        if not isinstance(self.attributes[self.att_class],Boolean):
            raise NameError("Class must be boolean.")
        return dict(zip([key for key, _ in self.attributes.items()],[self.roc_auc(key) if isinstance(value,Numerical) else np.NaN for key , value in self.attributes.items()]))
    
    #Returns the correlation between the given numerical attributes (att_A,att_B) using the specified method (method = pearson, spearman, kendall).
    def correlation(self,att_A,att_B,method="pearson"):
        if method!="pearson" and method!="spearman" and method!="kendall":
            raise NameError("Invalid correlation measure. Allowed measures are: pearson, spearman and kendall.")
        name_A = str(att_A)
        name_B = str(att_B)
        if name_A in self.attributes and name_B in self.attributes and isinstance(self.attributes[name_A],Numerical) and isinstance(self.attributes[name_B],Numerical):
            return self.attributes[name_A].data.corr(self.attributes[name_B].data,method=method)
        else:
            raise NameError("Correlation can only be computed between existing numerical attributes.")
    
    #Returns the correlations between all pairs of numerical attributes in the data set using the specified method (method = pearson, spearman, kendall).
    def correlation_att(self,method="pearson"):
        return dict(zip([key for key, _ in self.attributes.items()],[dict(zip([key2 for key2, _ in self.attributes.items()],[self.correlation(key1,key2,method=method) if isinstance(self.attributes[key1], Numerical) and isinstance(self.attributes[key2], Numerical) else np.NaN for key2, _ in self.attributes.items()])) for key1, _ in self.attributes.items()]))
    
    #Returns the normalized mutual information between the given non-numerical attributes (att_A,att_B).
    def norm_mutual_info(self,att_A,att_B):
        name_A = str(att_A)
        name_B = str(att_B)
        if name_A in self.attributes and name_B in self.attributes and isinstance(self.attributes[name_A],(Boolean,String,Categorical)) and isinstance(self.attributes[name_B],(Boolean,String,Categorical)):
            if isinstance(self.attributes[name_A],Categorical):
                vals = self.attributes[name_A].values
            else:
                vals = set(str(ind) for ind in set(self.attributes[name_A].data))
            H_x = self.attributes[name_A].entropy()
            H_y = self.attributes[name_B].entropy()
            H_xy = 0
            tot = len(self.attributes[name_B].data)
            col_vals = self.attributes[name_B].data.value_counts()
            auxiliar_cat = Categorical()
            for value in set(self.attributes[name_B].data):
                auxiliar_cat.set_data([str(ind) for ind in self.attributes[name_A].data[self.attributes[name_B].data == value]],vals)
                H_xy += (col_vals[value]/tot)*auxiliar_cat.entropy()
            return (2*(H_x - H_xy))/(H_x + H_y)
        else:
            raise NameError("Mutual information can only be computed between existing boolean, string or categorical attributes.")
    
    #Returns the normalized mutual informations between all pairs of non-numerical attributes in the data set.
    def norm_mutual_info_att(self):
        return dict(zip([key for key, _ in self.attributes.items()],[dict(zip([key2 for key2, _ in self.attributes.items()],[self.norm_mutual_info(key1,key2) if isinstance(self.attributes[key1], (Boolean,String,Categorical)) and isinstance(self.attributes[key2], (Boolean,String,Categorical)) else np.NaN for key2, _ in self.attributes.items()])) for key1, _ in self.attributes.items()]))

    #Discretizes the specified numerical attribute (att) using the specified method (method = frequency, width, custom) and number of intervals (num_bins) or cut points (cut_points).
    def discretize(self, att, method, num_bins=None, cut_points=None):
        name = str(att)
        if(name not in self.attributes.keys()):
            raise NameError("Attribute not found.")
        if(not isinstance(self.attributes[name],Numerical)):
            raise NameError("Can't discretize a non-numerical attribute.")
        if method == "frequency":
            self.attributes[name] = self.attributes[name].discretizeEF(num_bins)[0]
        elif method == "width":
            self.attributes[name] = self.attributes[name].discretizeEW(num_bins)[0]
        elif method == "custom":
            self.attributes[name] = self.attributes[name].discretize(cut_points)[0]
        else:
            raise NameError("Invalid discretization method. Accepted methods are: frequency, width, custom.")
    
    #Discretizes all the numerical attributes in the data set using the specified method (method = frequency, width) and number of intervals (num_bins).
    def discretize_att(self, num_bins, method):
        if(type(num_bins)==int):
            if method != "frequency" and method != "width":
                raise NameError("Invalid discretization method. Accepted methods are: frequency, width.")
            for key,value in self.attributes.items():
                if isinstance(value, Numerical):
                    if method == "frequency":
                        self.attributes[key] = self.attributes[key].discretizeEF(num_bins)[0]
                    else:
                        self.attributes[key] = self.attributes[key].discretizeEW(num_bins)[0]
        else:
            raise NameError("Number of intervals must be an integer.")

    #Standarizes the specified numerical attribute (att) so that it has mean = 0 and variance = 1.
    def standarize(self, att):
        name = str(att)
        if(name not in self.attributes.keys()):
            raise NameError("Attribute not found.")
        if(not isinstance(self.attributes[name],Numerical)):
            raise NameError("Can't standarize a non-numerical attribute.")
        self.attributes[name].standarize()
    
    #Standarizes all the numerical attributes in the data set so that they have mean = 0 and variance = 1.
    def standarize_att(self):
        for key,value in self.attributes.items():
            if isinstance(value, Numerical):
                self.attributes[key].standarize()

    #Normalizes the specified numerical attribute (att) between 0 and 1.
    def normalize(self, att):
        name = str(att)
        if(name not in self.attributes.keys()):
            raise NameError("Attribute not found.")
        if(not isinstance(self.attributes[name],Numerical)):
            raise NameError("Can't normalize a non-numerical attribute.")
        self.attributes[name].normalize()
         
    #Normalizes all the numerical attributes in the data set between 0 and 1.
    def normalize_att(self):
        for key,value in self.attributes.items():
            if isinstance(value, Numerical):
                self.attributes[key].normalize()

    #Filters the attributes in the data set according to a given metric (metric = variance, mean, median, entropy, auc).
    #The value of the metric for each attribute is compared to a given value (threshold) using the specified comparator (comparator = lt, gt, le, ge, eq, neq).
    #If the comparison returns False, the attribute is removed from the data set.
    def filter_by(self,metric,comparator,threshold):
        def _lt(a, b): return a < b
        def _gt(a, b): return a > b
        def _le(a, b): return a <= b
        def _ge(a, b): return a >= b
        def _eq(a, b): return a == b
        def _neq(a, b): return a != b
        if type(threshold) != int and type(threshold) != float:
            raise NameError("The threshold value must be integer or float.")
        if metric == "entropy":
            values = self.entropy_att()
        elif metric == "auc":
            values = self.roc_auc_att()
        elif metric == "variance":
            values = self.variance_att()
        elif metric == "mean":
            values = self.mean_att()
        elif metric == "median":
            values = self.median_att()
        else:
            raise NameError("Invalid metric. Accepted metrics are: entropy, auc, variance, mean, median.")
        if comparator == "lt":
            func = _lt
        elif comparator == "gt":
            func = _gt
        elif comparator == "le":
            func = _le
        elif comparator == "ge":
            func = _ge
        elif comparator == "eq":
            func = _eq
        elif comparator == "neq":
            func = _neq
        else:
            raise NameError("Invalid comparator. Accepted comparators: lt, gt, le, ge, eq, neq.")
        for key, value in values.items():
            if not np.isnan(value) and not func(value, threshold):
                if key != self.att_class:
                    del self.attributes[key]
