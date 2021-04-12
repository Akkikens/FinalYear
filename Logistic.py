import math
import collections
from sklearn.metrics import average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import auc
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import re
from publicsuffixlist import PublicSuffixList
df=pd.read_csv("C:\\input\\mixed_domain.csv")
list1=df['DGA_family'].unique().tolist()
list1
list2=df['Domain'].unique().tolist()
list2
list1=df['Type'].unique().tolist()
list1
df.tail()
import sys
topLevelDomain = []
with open('C:\\input\\tlds-alpha-by-domain.txt', 'r') as content:
    for line in content:
        topLevelDomain.append((line.strip('\n')))
        
psl = PublicSuffixList()
def domain_length(domain):
  # Generate Domain Name Length (DNL)
  return len(domain)

def subdomains_number(domain):
      # Generate Number of Subdomains (NoS)
      subdomain = ignoreVPS(domain)
      return (subdomain.count('.') + 1)
  
def subdomain_length_mean(domain):
  # enerate Subdomain Length Mean (SLM) 
  subdomain = ignoreVPS(domain)
  result = (len(subdomain) - subdomain.count('.')) / (subdomain.count('.') + 1)
  return result

def has_www_prefix(domain):
  # Generate Has www Prefix (HwP)
  if domain.split('.')[0] == 'www':
    return 1
  else:
    return 0

def underscore_ratio(domain):
  # Generate Underscore Ratio (UR) on dataset
  subString = ignoreVPS(domain)
  result = subString.count('_') / (len(subString) - subString.count('.'))
  return result

def ignoreVPS(domain):
    # Return the rest of domain after ignoring the Valid Public Suffixes:
    validPublicSuffix = '.' + psl.publicsuffix(domain)
    if len(validPublicSuffix) < len(domain):
         # If it has VPS
        subString = domain[0: domain.index(validPublicSuffix)]  
    elif len(validPublicSuffix) == len(domain):
        return 0
    else:
        # If not
        subString = domain
    
    return subString

def contains_digit(domain):
  """
   Contains Digits 
  """
  subdomain = ignoreVPS(domain)
  for item in subdomain:
    if item.isdigit():
      return 1
  return 0

def vowel_ratio(domain):
  """
  calculate Vowel Ratio 
  """
  VOWELS = set('aeiou')
  v_counter = 0
  a_counter = 0
  subdomain = ignoreVPS(domain)
  for item in subdomain:
    if item.isalpha():
      a_counter+=1
      if item in VOWELS:
        v_counter+=1
  if a_counter>1:
    ratio = v_counter/a_counter
    return ratio

def contains_IP_address(domain):
      # Generate Contains IP Address (CIPA) on datasetx
        splitSet = domain.split('.')
        for element in splitSet:
            if(re.match("\d+", element)) == None:
                return 0
        return 1 
    
def digit_ratio(domain):
  """
  calculate digit ratio
  """
  d_counter = 0
  counter = 0
  subdomain = ignoreVPS(domain)
  for item in subdomain:
    if item.isalpha() or item.isdigit():
      counter+=1
      if item.isdigit():
        d_counter+=1
  if counter>1:
    ratio = d_counter/counter
    return ratio

def typeTo_Binary(type):
  # Convert Type to Binary variable DGA = 1, Normal = 0
  if type == 'DGA':
    return 1
  else:
    return 0

def contains_single_character_subdomain(domain):
  # Generate Contains Single-Character Subdomain (CSCS) 
    domain = ignoreVPS(domain)
    str_split = domain.split('.')
    minLength = len(str_split[0])
    for i in range(0, len(str_split) - 1):
        minLength = len(str_split[i]) if len(str_split[i]) < minLength else minLength
    if minLength == 1:
        return 1
    else:
        return 0
    
def contains_TLD_subdomain(domain):
  # Generate Contains TLD as Subdomain (CTS)
    subdomain = ignoreVPS(domain)
    str_split = subdomain.split('.')
    for i in range(0, len(str_split) - 1):
        if str_split[i].upper() in topLevelDomain:
            return 1
    return 0

def prc_rrc(domain):
    """
    calculate the Ratio of Repeated Characters in a subdomain
    """
    subdomain = ignoreVPS(domain)
    subdomain = re.sub("[.]", "", subdomain)
    char_num=0
    repeated_char_num=0
    d = collections.defaultdict(int)
    for c in list(subdomain):
        d[c] += 1
    for item in d:
        char_num +=1
        if d[item]>1:
            repeated_char_num +=1
    ratio = repeated_char_num/char_num
    return ratio

def prc_rcc(domain):
    """
    calculate the Ratio of Consecutive Consonants
    """
    VOWELS = set('aeiou')
    counter = 0
    cons_counter=0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        i = 0
        if item.isalpha() and item not in VOWELS:
            counter+=1
        else:
            if counter>1:
                cons_counter+=counter
            counter=0
        i+=1
    if i==len(subdomain) and counter>1:
        cons_counter+=counter
    ratio = cons_counter/len(subdomain)
    return ratio

def prc_rcd(domain):
    """
    calculate the ratio of consecutive digits
    """
    counter = 0
    digit_counter=0
    subdomain = ignoreVPS(domain)
    for item in subdomain:
        i = 0
        if item.isdigit():
            counter+=1
        else:
            if counter>1:
                digit_counter+=counter
            counter=0
        i+=1
    if i==len(subdomain) and counter>1:
        digit_counter+=counter
    ratio = digit_counter/len(subdomain)
    return ratio

def has_hvltd(domain):
  # Generate Has a Valid Top Level Domain (HVTLD)
  if domain.split('.')[len(domain.split('.')) - 1].upper() in topLevelDomain:
    return 1
  else:
    return 0

def prc_entropy(domain):
    """
    calculate the entropy of subdomain
    :param domain_str: subdomain
    :return: the value of entropy
    """
    subdomain = ignoreVPS(domain)
    # get probability of chars in string
    prob = [float(subdomain.count(c)) / len(subdomain) for c in dict.fromkeys(list(subdomain))]

    # calculate the entropy
    entropy = - sum([p * math.log(p) / math.log(2.0) for p in prob])
    return entropy

def extract_features():
    df['DNL'] = df['Domain'].apply(lambda x: domain_length(x))
    df['NoS'] = df['Domain'].apply(lambda x: subdomains_number(x))
    df['SLM'] = df['Domain'].apply(lambda x: subdomain_length_mean(x))
    df['HwP'] = df['Domain'].apply(lambda x: has_www_prefix(x))
    df['HVTLD'] = df['Domain'].apply(lambda x: has_hvltd(x))
    df['CSCS'] = df['Domain'].apply(lambda x: contains_single_character_subdomain(x))
    df['CTS'] = df['Domain'].apply(lambda x: contains_TLD_subdomain(x))
    df['UR'] = df['Domain'].apply(lambda x: underscore_ratio(x))
    df['CIPA'] = df['Domain'].apply(lambda x: contains_IP_address(x))
    df['contains_digit']= df['Domain'].apply(lambda x:contains_digit(x))
    df['vowel_ratio']= df['Domain'].apply(lambda x:vowel_ratio(x))
    df['digit_ratio']= df['Domain'].apply(lambda x:digit_ratio(x))
    df['RRC']= df['Domain'].apply(lambda x:prc_rrc(x))
    df['RCC']= df['Domain'].apply(lambda x:prc_rcc(x))
    df['RCD']= df['Domain'].apply(lambda x:prc_rcd(x))
    df['Entropy']= df['Domain'].apply(lambda x:prc_entropy(x))
    
extract_features()

df

from sklearn import preprocessing 
  
# label_encoder object knows how to understand word labels. 
label_encoder = preprocessing.LabelEncoder() 
df['DGA_family']= label_encoder.fit_transform(df['DGA_family']) 
df['Domain']= label_encoder.fit_transform(df['Domain']) 
df['Type']= label_encoder.fit_transform(df['Type']) 

df.drop(columns=["Domain"],axis=1,inplace=True)
df.dropna()
df.fillna(df.mean(), inplace=True)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X = df.drop(columns='Type',axis=1) # here we have 2 variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y=df[["Type"]]
X = X.values.astype(np.float)
Y = Y.values.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
y_train1 = y_train.flatten()
classifier.fit(X_train, y_train1)
X_test
new_input = [[45.,17.,1.,8.,3.,67.,23.,67.,49.,17.,1.,8.,3.,67.,23.,67.,23.]]
y_pred=classifier.predict(X_test)
y_pred
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import precision_recall_fscore_support
precision_recall_fscore_support(y_test, y_pred, average='macro')
precision_recall_fscore_support(y_test, y_pred, average=None,labels=[0,1])
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# actual values
actual = y_test
# predicted values
predicted =y_pred

# confusion matrix
matrixresult = confusion_matrix(actual,predicted, labels=[1,0])
print('Confusion matrix : \n',matrixresult)

# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(actual,predicted,labels=[1,0])
print('Classification report : \n',matrix)
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

binary1 = np.array(matrixresult)

fig, ax = plot_confusion_matrix(conf_mat=binary1)
plt.show()
# import pickle
# domain=str(input("Enter the Domain:"))
# domain
# def domain_length(domain):
#   # Generate Domain Name Length (DNL)
#   return len(domain)