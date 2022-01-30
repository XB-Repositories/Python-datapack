import pandas as pd
import numpy as np
import math
import warnings

### GENERIC ATTRIBUTE CLASS
class Attribute():
    
    ###################CONSTRUCTOR FUNCTION####################
    def __init__(self,*args):
        self.data=None
        self.length=0
        if len(args) == 1:
            if isinstance(args[0],pd.Series):
                self.data = args[0]
            elif isinstance(args[0],(np.ndarray,list)):
                self.data = pd.Series(args[0])
            else:
                raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
            if self.data.dtype != int and self.data.dtype != float and self.data.dtype != bool and self.data[self.data.apply(type) != str].count() > 0:
                self.data=None
                raise NameError("Accepted data types are: numerical, bool, str. The attribute data can only contain one data type.")
            self.length=len(self.data)
        elif len(args)!=0:
            raise NameError("Too many parameters.")

    ####################SETTERS#######################
    
    #Initializes the data of the attribute according to the input array-like parameter (d).
    def set_data(self,d):
        self.data=None
        self.length=0
        if isinstance(d,pd.Series):
            self.data = d
        elif isinstance(d,(np.ndarray,list)):
            self.data = pd.Series(d)
        else:
            raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
        if self.data.dtype != int and self.data.dtype != float and self.data.dtype != bool and self.data[self.data.apply(type) != str].count() > 0:
            self.data=None
            raise NameError("Accepted data types are: numerical, bool, str. The attribute data can only contain one data type.")
        self.length=len(self.data)

    ###################GETTERS###################

    #Returns the data of the attribute.
    def get_data(self):
        return self.data

    #Returns the number of values in the attribute.
    def get_number_values(self):
        return self.length

    #Returns the value in the specified index (ind).
    def get_value(self,ind):
        if(type(ind)==int and ind>=0 and ind<self.length):
            return self.data[ind]
        else:
            raise NameError("Index must be an integer between 0 and the total number of values.")

    #############################################

    #Updates the value in the specified index (ind).
    def update_value(self,ind,val):
        if(type(ind)==int and ind>=0 and ind<self.length):
            self.data[ind]=val
        else:
            raise NameError("Index must be an integer between 0 and the total number of values.")
    
    #Prints the data of the attribute.
    def print_data(self):
        print("Attribute values:")
        print(self.data)

### NUMERICAL ATTRIBUTE CLASS (inherits ATTRIBUTE)
class Numerical(Attribute):
    
    ###################CONSTRUCTOR FUNCTION####################
    def __init__(self,*args):
        if len(args) == 1:
            if isinstance(args[0],pd.Series):
                data = args[0]
            elif isinstance(args[0],(np.ndarray,list)):
                data = pd.Series(args[0])
            else:
                raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
            if data.dtype != int and data.dtype != float:
                raise NameError("Numerical attribute must be numerical type.")
            Attribute.__init__(self,data)
        elif len(args) == 0:
            Attribute.__init__(self)
        else:
            raise NameError("Too many parameters.")

    ###################SETTERS####################
    
    #Initializes the data of the numerical attribute according to the input array-like parameter (d).
    def set_data(self,d):
        if isinstance(d,pd.Series):
            data = d
        elif isinstance(d,(np.ndarray,list)):
            data = pd.Series(d)
        else:
            raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
        if data.dtype != int and data.dtype != float:
                raise NameError("Numerical attribute must be numerical type.")
        Attribute.set_data(self,data)

    ##############################################

    #Updates the value in the specified index (ind).
    def update_value(self,ind,val):
        if(isinstance(val,int) or isinstance(val,float)):
            Attribute.update_value(self,ind,val)
        else:
            raise NameError("New value must be numerical type.")
    
    #Normalizes the data of the numerical attribute between 0 and 1.
    def normalize(self):
        self.data -= self.data.min()
        self.data /= self.data.max()
    
    #Standarizes the data of the numerical attribute so that it has mean = 0 and variance = 1.
    def standarize(self):
        self.data -= self.data.mean()
        self.data /= self.data.std(ddof=1)
    
    #Returns a new categorical attribute created from the discretization of the numerical attribute.
    #It uses the equal width discretization strategy with the given number of intervals (num_bins).
    def discretizeEW(self, num_bins):
        if type(num_bins) != int:
            raise NameError("Number of intervals must be an integer.")
        if num_bins < 2:
            raise NameError("Number of intervals must be equal to or higher than 2.")
        min_val = self.data.min()
        size_cut = (self.data.max()-min_val)/num_bins
        cut_points = [min_val+(size_cut*i) for i in range(1,num_bins)]
        cat_values = ["(" + str(cut_points[i-1]) + ", " + str(cut_points[i]) + "]" if i!=0 and i!=num_bins-1 else "(-infinity, " + str(cut_points[i]) + "]" if i==0 else "(" + str(cut_points[i-1]) + ", infinity)" for i in range(num_bins)]
        discretized = Categorical([cat_values[np.searchsorted(cut_points,self.data[i])] for i in range(len(self.data))],cat_values)
        return(discretized, cut_points)
    
    #Returns a new categorical attribute created from the discretization of the numerical attribute.
    #It uses the equal frequency discretization strategy with the given number of intervals (num_bins).
    def discretizeEF(self, num_bins):
        if type(num_bins) != int:
            raise NameError("Number of intervals must be an integer.")
        if num_bins < 2:
            raise NameError("Number of intervals must be equal to or higher than 2.")
        num_bins = min(num_bins,len(self.data))
        cut_size = int(len(self.data)/num_bins)
        cut_mod = len(self.data)%num_bins
        sorted_list = sorted(list(self.data))
        cut_points = [sorted_list[((cut_size+1)*(i))-1] if i<cut_mod else sorted_list[(cut_size*(i))+(cut_mod-1)] for i in range(1,num_bins)]
        cat_values = ["(" + str(cut_points[i-1]) + ", " + str(cut_points[i]) + "]" if i!=0 and i!=num_bins-1 else "(-infinity, " + str(cut_points[i]) + "]" if i==0 else "(" + str(cut_points[i-1]) + ", infinity)" for i in range(num_bins)]
        discretized = Categorical([cat_values[np.searchsorted(cut_points,self.data[i])] for i in range(len(self.data))],cat_values)
        return(discretized, cut_points)
    
    #Returns a new categorical attribute created from the discretization of the numerical attribute.
    #The discretization is performed according to the given cut points (cut_points).
    def discretize(self, cut_points):
        if not all([type(item)==int or type(item)==float for item in cut_points]):
            raise NameError("Cut points must be a numerical list.")
        if len(cut_points) < 1:
            raise NameError("Cut point list must contain at least one cut point.")
        cut_points = sorted(cut_points)
        cat_values = ["(" + str(cut_points[i-1]) + ", " + str(cut_points[i]) + "]" if i!=0 and i!=len(cut_points) else "(-infinity, " + str(cut_points[i]) + "]" if i==0 else "(" + str(cut_points[i-1]) + ", infinity)" for i in range(len(cut_points)+1)]
        discretized = Categorical([cat_values[np.searchsorted(cut_points,self.data[i])] for i in range(len(self.data))],cat_values)
        return(discretized, cut_points)

    #Returns the mean of the data of the numerical attribute.
    def mean(self):
        return np.mean(self.data)

    #Returns the median of the data of the numerical attribute.
    def median(self):
        return np.median(self.data)
    
    #Returns the variance of the data of the numerical attribute.
    def variance(self):
        return np.var(self.data,ddof=1)

    #Prints the data of the numerical attribute.
    def print_data(self):
        print("Attribute type: Numerical")
        Attribute.print_data(self)

### BOOLEAN ATTRIBUTE CLASS (inherits ATTRIBUTE)
class Boolean(Attribute):
    
    ###################CONSTRUCTOR FUNCTION####################
    def __init__(self,*args):
        if len(args) == 1:
            if isinstance(args[0],pd.Series):
                data = args[0]
            elif isinstance(args[0],(np.ndarray,list)):
                data = pd.Series(args[0])
            else:
                raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
            if data.dtype != bool:
                raise NameError("Boolean attribute must be boolean type.")
            Attribute.__init__(self,data)
        elif len(args) == 0:
            Attribute.__init__(self)
        else:
            raise NameError("Too many parameters.")

    ###################SETTERS####################
    
    #Initializes the data of the boolean attribute according to the input array-like parameter (d).
    def set_data(self,d):
        if isinstance(d,pd.Series):
            data = d
        elif isinstance(d,(np.ndarray,list)):
            data = pd.Series(d)
        else:
            raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
        if data.dtype != bool:
                raise NameError("Boolean attribute must be boolean type.")
        Attribute.set_data(self,data)

    #############################################

    #Updates the value in the specified index (ind).
    def update_value(self,ind,val):
        if(type(val)==bool):
            Attribute.update_value(self,ind,val)
        else:
            raise NameError("New value must be boolean type.")
    
    #Returns the categorical version of the boolean attribute.
    def to_categorical(self,values=None):
        warnings.warn("Boolean data converted to string when creating the categorical attribute.")
        data_str = self.data.astype(str)
        return Categorical(data_str,["True","False"])

    #Returns the mode of the data of the boolean attribute.
    def mode(self):
        return self.data.mode()[0]
    
    #Returns the entropy of the data of the boolean attribute.
    def entropy(self):
        total = len(self.data)
        p = np.array(self.data.value_counts())/total
        return sum([-i*math.log2(i) for i in p])

    #Prints the data of the boolean attribute.
    def print_data(self):
        print("Attribute type: Boolean")
        Attribute.print_data(self)

### STRING ATTRIBUTE CLASS (inherits ATTRIBUTE)
class String(Attribute):
    
    ###################CONSTRUCTOR FUNCTION####################
    def __init__(self,*args):
        if len(args) == 1:
            if isinstance(args[0],pd.Series):
                data = args[0]
            elif isinstance(args[0],(np.ndarray,list)):
                data = pd.Series(args[0])
            else:
                raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
            if data[data.apply(type) != str].count() > 0:
                raise NameError("String attribute must be string type.")
            Attribute.__init__(self,data)
        elif len(args) == 0:
            Attribute.__init__(self)
        else:
            raise NameError("Too many parameters.")

    ###################SETTERS####################
            
    #Initializes the data of the string attribute according to the input array-like parameter (d).
    def set_data(self,d):
        if isinstance(d,pd.Series):
            data = d
        elif isinstance(d,(np.ndarray,list)):
            data = pd.Series(d)
        else:
            raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
        if data[data.apply(type) != str].count() > 0:
                raise NameError("String attribute must be string type.")
        Attribute.set_data(self,data)

    ##############################################

    #Updates the value in the specified index (ind).
    def update_value(self,ind,val):
        if(isinstance(val,str)):
            Attribute.update_value(self,ind,val)
        else:
            raise NameError("New value must be string type.")
    
    #Returns the categorical version of the string attribute.
    #The possible values of the categorical attribute can be specified through a parameter (values).
    def to_categorical(self,values=None):
        if values == None:
            return Categorical(self.data)
        else:
            return Categorical(self.data,values)

    #Returns the mode of the data of the string attribute.
    def mode(self):
        return self.data.mode()[0]
    
    #Returns the entropy of the data of the string attribute.
    def entropy(self):
        total = len(self.data)
        p = np.array(self.data.value_counts())/total
        return sum([-i*math.log2(i) for i in p])

    #Prints the data of the string attribute.
    def print_data(self):
        print("Attribute type: String")
        Attribute.print_data(self)

### CATEGORICAL ATTRIBUTE CLASS (inherits STRING). This class contains a collection of possible values in addition to the attribute data.
class Categorical(String):
    
    ###################CONSTRUCTOR FUNCTION####################
    def __init__(self,*args):
        self.values = None
        if len(args) == 2 or len(args) == 1:
            if isinstance(args[0],pd.Series):
                data = args[0]
            elif isinstance(args[0],(np.ndarray,list)):
                data = pd.Series(args[0])
            else:
                raise NameError("The attribute data must be a pd.Series, a np.array or a list.")
            if len(args) == 2:
                if isinstance(args[1],set):
                    self.values = args[1]
                elif isinstance(args[1],(pd.Series,np.ndarray,list)):
                    self.values = set(args[1])
                else:
                    raise NameError("The categorical value collection must be a pd.Series, a np.ndarray, a list or a set.")
                if not all([i in self.values for i in data]):
                    self.values = None
                    raise NameError("Some of the values in the attribute data are not valid.")
            else:
                warnings.warn("The categorical value collection has not been specified, only the values that appear in the attribute data will be considered.")
                self.values = set(data)
            String.__init__(self,data)
        elif len(args) == 0:
            String.__init__(self)
        else:
            raise NameError("Incorrect number of parameters.")

    ###################SETTERS####################
    
    #Initializes the data of the categorical attribute according to the input array-like parameter (d).
    #It also initializes the collection of possible values of the categorical attribute according to the input parameter (v).
    def set_data(self,d,v=None):
        self.values = None
        if isinstance(d,pd.Series):
            data = d
        elif isinstance(d,(np.ndarray,list)):
            data = pd.Series(d)
        else:
            raise NameError("The attribute data must be a pd.Series, a np.ndarray or a list.")
        if v == None:
            warnings.warn("The categorical value collection has not been specified, only the values that appear in the attribute data will be considered.")
            self.values = set(data)
        elif isinstance(v,set):
            self.values = v
        elif isinstance(v,(pd.Series,np.ndarray,list)):
            self.values = set(v)
        else:
            raise NameError("The categorical value collection must be a pd.Series, a np.ndarray, a list or a set.")
        if not all([i in self.values for i in data]):
            self.values = None
            raise NameError("Some of the values in the attribute data are not valid.")
        String.set_data(self,data)

    ##############################################

    #Updates the value in the specified index (ind).
    def update_value(self,ind,val):
        if val not in self.values:
            raise NameError("Invalid new value. Allowed values are: "+str(self.values))
        String.update_value(self,ind,val)

    #Prints the data of the categorical attribute.
    def print_data(self):
        print("Attribute type: Categorical")
        print("Allowed values: ", self.values)
        Attribute.print_data(self)
