#!/usr/bin/env python

#  Author:  Ben Tannenwald
##  Date:    May 20, 2020
##  Purpose: Class to load data, create user-defined columns, save some desired data collections


#TO-DO
# Suggestions:
# -3) Are different adverse events reported in different countries?
#  X --> make plot for REACTION for different countries
#    --> age splits
#  X --> dot product over reaction-space to give single number estimation of "effect overlap"

# -2) What are the different adverse events associated with different disease areas? 
#   --> does disease area mean e.g. drugindication = RHEUMATOID ARTHRITIS
#  X --> make plot of REACTION for drugindiciation
#  X --> make plot of REACTION for most common drug(s) associated with particular indication
#   --> show gender splits?
#   --> show splits by age?

# -1) What drugs tend to be taken together? 
#  X --> make a correlation matrix of drugs

# Ben Ideas
# X --> Find most common drugs for given drugindication
#  --> For particular drug show N_adverse as a ftn of time/patient age
#  --> Show N_adverse as a function of dosage, patient age/sex
# X --> Learn what seriousness reporting means
# X --> find common drugs, show 
#  --> make graphs for different seriousness categories: most common generics, most common brands, most common side effects
#  --> turn into class. can use class to run over all files OR run over subset (e.g. one year) to make time-trends
#  --> make time-trend of adverse events for a particular drug. split by adverse categories
#  X --> make encoded vector of adverse effects. run k-means cluster to see if "areas" emerge- could classify drugs. DOESN'T WORK
#  --> drugWatcher : code that uses apply(lambda) where lambda calculate timeToDouble/timeToSomeThreshold 



import pandas as pd
from pandas.core.common import flatten
import numpy as np
import json
import zipfile
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


# In[2]:


filename2='/home/btannenw/Desktop/life/adverseDrugFDA/data/2018-Q4/drug-event-0001-of-0029.json.zip'
filename='/home/btannenw/Desktop/life/adverseDrugFDA/data/2011-Q3/drug-event-0001-of-0012.json.zip'
#jsonFile = json.loads(open(filename),)
#json.loads(data.decode("utf-8"))
#oneFile = pd.read_json('/home/btannenw/Desktop/life/adverseDrugFDA/data/2011-Q3/drug-event-0001-of-0012.json.zip', 
#                       encoding='utf-8', compression='zip')


# In[ ]:





# In[3]:


d = []
data = []

with zipfile.ZipFile(filename, 'r') as z:
    for filename in z.namelist():  
        print(filename)  
        with z.open(filename) as f:  
            data = f.read()  
            d = json.loads(data.decode("utf-8"))  


# In[4]:


print(d.keys())
#d['meta']
df = pd.DataFrame(d['results'])
df


# In[5]:


print(df.patient[60]['drug'][0].keys())
df.patient[60]['drug'][0]['openfda']
df.patient[60]['drug'][0]['openfda'].keys()
#df.patient[60]['drug'][0]['openfda']['substance_name']


# In[8]:


#df.patient[60]['drug'][1].keys()
#df.patient[3]['drug'][0]['openfda'].keys()
#df3.dropna(subset=['pharm_class_cs']).pharm_class_cs.value_counts()


# In[9]:


def returnThresholdPoint(counts, labels, threshold, verbose=False):
    """return point where threshold"""

    # *** 0. Find threshold break
    _total = sum(counts)
    _processed = 0
    
    # *** 1. Escape for when you just want the top X in a list regardless of percentage
    if threshold > 1:
        if verbose:
            for iKey in range(0, threshold):
                print( "{}: {:1.2f}%".format(labels[iKey], 100*counts[iKey]/_total))
            print( "{}: {:1.2f}%".format('Other', 100*sum(counts[iKey:])/_total))

        return threshold
    
    # *** 2. Actually calculate point where > X% of sample contained
    for iKey in range(0, len(labels)):
        if _processed/_total < threshold:
            _processed += counts[iKey]
            if verbose:
                print( "{}: {:1.2f}%".format(labels[iKey], 100*counts[iKey]/_total))
                #print( "{}: {} {}".format(labels[iKey], counts[iKey], _total))
        else:
            break
    
    if verbose:
        print( "{}: {:1.2f}%".format('Other', 100*sum(counts[iKey:])/_total))
    
    return iKey


def makePiePlot(counts, labels, threshold=0.9, legend=True, title='', verbose=False):
    """ make pie plot that lumps tail above some threshold into 'other' category """
    
    # *** 0. Find threshold break
    threshKey = returnThresholdPoint(counts, labels, threshold, verbose)
     
    # *** 1. Make counts/labels below threshold and store rest as 'Other'
    _pieCounts = counts.copy()[:threshKey]
    _pieCounts.append( sum(counts[threshKey:]) )
    _pieLabels = labels
    if type(_pieLabels)!=list:
        _pieLabels = _pieLabels.to_list()
    _pieLabels = _pieLabels[:threshKey]
    _pieLabels.append( 'Other')
    
    
    # *** 2. Remove 'Other' if top-X plot rather than with threshold
    if threshold > 1:
        _pieCounts = _pieCounts[:(threshKey-1)]
        _pieLabels = _pieLabels[:(threshKey-1)]

    # *** 3. Make pie plot
    #_colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    #_explode = (0.0, ) + (0.2,)*(iKey-1) #explode all slices but one
    _pieOpts = dict(autopct='%1.1f%%', 
                shadow=True, 
                startangle=90
                #explode= _explode, 
                #labels= _pieLabels, 
                #colors= _colors, 
    )
    if not legend:
        _pieOpts['labels'] = _pieLabels
    
    plt.pie(_pieCounts, 
        **_pieOpts
        )
    
    if legend:
        plt.legend( _pieLabels, loc="best")

    plt.axis('equal')
    plt.title( title )
    plt.show()

    
def returnCountsAndLabels(freqPair):
    """return easily digestible list of reactions and counts"""
    
    _reactions = [ pair[0] for pair in freqPair ]
    _counts    = [ int(pair[1]) for pair in freqPair ]
    
    return _reactions, _counts



def countUnique(reactionsList, threshold=0.90, verbose=False):
    
    # *** 0. Make pairs of uniques
    (unique, counts) = np.unique( reactionsList, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    #print(frequencies)
    freq2 = sorted( frequencies, key=lambda pair: int(pair[1]), reverse=True) 
    #print(freq2[:10])
    #print( len(frequencies) )
    
    # *** 1. Find threshold break
    _reactions, _counts = returnCountsAndLabels( freq2)
    _threshKey = returnThresholdPoint(_counts, _reactions, threshold, verbose)

    
    return _threshKey, freq2


# In[47]:


def imputePrimarySource(primarySourceInfo, expectedKey):
    """function to impute some dummy info for missing fields"""

    
    if expectedKey in primarySourceInfo.keys():
        dfValue = primarySourceInfo[ expectedKey ]
        
        if dfValue == None:
            return -1
        elif dfValue.isnumeric():
            return float(dfValue)
        else:
            return dfValue
    else:
        return -1
    

def imputePatientReaction(patientInfo, expectedKey):
    """function to impute some dummy info for missing fields"""

    if expectedKey in patientInfo.keys():
        #dfValue = primarySourceInfo[ expectedKey ]
        dfValue = [ reaction.get('reactionmeddrapt') for reaction in patientInfo[ expectedKey]]

        #if typedfValue == None:
        #    return -1
        #elif dfValue.isnumeric():
        #    return float(dfValue)
        #else:
        return dfValue
    else:
        return -1
    
def getPatientData(patientInfo, categoryKey, dataKey):
    """function to impute some dummy info for missing fields"""

    dfValue = []
        
    # return NaN if no data
    if len(patientInfo[ categoryKey]) == 0:
        return np.nan
    
    for iEntry in np.arange(0, len(patientInfo[ categoryKey])):
        if dataKey in patientInfo[ categoryKey ][iEntry]:
            dfValue.append( patientInfo[ categoryKey][iEntry][ dataKey ] )
        #if expectedKey == 'drugindication':
        #    return [dfValue]
        #return dfValue
    #else:
    #    return -1
    
    # return NaN if no results
    if len(dfValue)==0:       
        return np.nan
    
    return set(list(flatten(dfValue)))


def getDrugData(patientInfo, categoryKey, subCategoryKey, dataKey):
    """function to impute some dummy info for missing fields"""

    df.patient[60]['drug'][0]['openfda']['substance_name']
    
    
    dfValue = []
        
    # return NaN if no data
    if len(patientInfo[ categoryKey]) == 0:
        return np.nan
    
    for iEntry in np.arange(0, len(patientInfo[ categoryKey])):
        if subCategoryKey in patientInfo[ categoryKey ][iEntry] and dataKey in patientInfo[ categoryKey ][iEntry][ subCategoryKey]:
            dfValue.append( patientInfo[ categoryKey ][iEntry][ subCategoryKey][ dataKey ] )
        #if expectedKey == 'drugindication':
        #    return [dfValue]
        #return dfValue
    #else:
    #    return -1
    
    # return NaN if no results
    if len(dfValue)==0:
        if dataKey=='generic_name':
            return 'No Generic Listed'
        if dataKey=='brand_name':
            return 'No Brand Listed'
        
        return np.nan
    
    return set(list(flatten(dfValue)))


# In[48]:


#%%time
df2 = df.dropna(subset=['primarysource', 'patient'])
df2['reportercountry'] = df2.apply( lambda x: imputePrimarySource( x['primarysource'], 'reportercountry'), axis=1)
df2['reporterqualification'] = df2.apply( lambda x: imputePrimarySource( x['primarysource'], 'qualification'), axis=1)
df2['patientOnsetAge'] = df2.apply( lambda x: imputePrimarySource( x['patient'], 'patientonsetage'), axis=1)
df2['patientReactions'] = df2.apply( lambda x: imputePatientReaction( x['patient'], 'reaction'), axis=1)
df2['drugIndication'] = df2.apply( lambda x: getPatientData( x['patient'], 'drug', 'drugindication'), axis=1)
df2['drugSubstanceName'] = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'openfda', 'substance_name'), axis=1)
df2['drugGenericName'] = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'openfda', 'generic_name'), axis=1)
df2['drugBrandName'] = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'openfda', 'brand_name'), axis=1)
df2['pharm_class_moa'] = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_moa'), axis=1)
df2['pharm_class_pe']  = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_pe'), axis=1)
df2['pharm_class_cs']  = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_cs'), axis=1)
df2['pharm_class_epc'] = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_epc'), axis=1)
df2['drugActiveSubstance'] = df2.apply( lambda x: getDrugData( x['patient'], 'drug', 'activesubstance', 'activesubstancename'), axis=1)

df3= df2[ df2['reportercountry']!=-1 ]

df_USA = df2[ df2['reportercountry']=='UNITED STATES']

print( len(df), len(df2), len(df3), len(df_USA))


# In[18]:


a= df3.dropna(subset=['drugSubstanceName']).drugSubstanceName.value_counts()
a


# In[19]:


countryCounts = df3.reportercountry.value_counts()

counts = countryCounts.to_list()
countries = countryCounts.keys()

makePiePlot(counts, countries, threshold=0.925, title='% Response by Country', verbose=True)

print(counts)
print(countries)


# In[ ]:





# In[20]:


def makeEfficiencyCurve(uniqueCountsAndKeys, keyType, maxKey=-1):
    """function to make efficiency curve and find x of plateau"""
    _total = sum(np.array(uniqueCountsAndKeys)[:,1].astype(np.int))
    
    _maxKey = maxKey if maxKey != -1 else len(np.array(uniqueCountsAndKeys)[:,1])
    _topX  = np.arange(0, _maxKey, 1).astype(np.int)
    _percentCaptured = [ sum(np.array(uniqueCountsAndKeys)[:x,1].astype(np.int))/ _total for x in _topX]

    plt.plot(_topX, _percentCaptured)
    plt.xlabel('Number of {}'.format(keyType))
    plt.ylabel('% Included')
    plt.show()


# In[21]:


# assume ages >100 are something weird
plt.hist(df3.patientOnsetAge, bins=np.arange(0, 100, 10))
#print(len(ages), max(ages), min(ages))
# adverse affects peak around 50... is that weird?


# In[ ]:





# In[ ]:





# In[ ]:


df3.patient[0]['drug'][0]['drugindication']


# In[ ]:





# In[22]:


reactions  = list(flatten(df3.patientReactions.to_list()))
#reactions2 =[ reaction.get('reactionmeddrapt') for patient in df.patient for reaction in patient['reaction']]
#get list of all reactions
print( len(reactions), len(set(reactions)))
# shows that there's a lot of reactions (~35k) and a lot of unique reactions (~3k). but since the number of unique
# is so much smaller, that means there should be a lot of correlation --> matrix


# In[23]:


df3_under45 = df3[ (df3.patientOnsetAge<45) & (df3.patientOnsetAge>-1) ]
df3_under45_reactions = list(flatten(df3_under45['patientReactions']))

df3_50to60 = df3[ (df3.patientOnsetAge>=50) & (df3.patientOnsetAge<60) & (df3.patientOnsetAge>-1) ]
df3_50to60_reactions = list(flatten(df3_50to60['patientReactions']))

df3_over65 = df3[ (df3.patientOnsetAge>65) & (df3.patientOnsetAge>-1) ]
df3_over65_reactions = list(flatten(df3_over65['patientReactions']))

threshold=10
print(type(df3_under45_reactions), np.shape(df3_under45_reactions))
threshKey_under45, all_under45 = countUnique(df3_under45_reactions, threshold)
all_reactions_under45, all_counts_under45 = returnCountsAndLabels( all_under45)

threshKey_50to60, all_50to60 = countUnique(df3_50to60_reactions, threshold)
all_reactions_50to60, all_counts_50to60 = returnCountsAndLabels( all_50to60)

threshKey_over65, all_over65 = countUnique(df3_over65_reactions, threshold)
all_reactions_over65, all_counts_over65 = returnCountsAndLabels( all_over65)


# In[24]:


makePiePlot(all_counts_under45, all_reactions_under45, threshold=10, title='Age < 45', legend=False, verbose=True)
makePiePlot(all_counts_50to60, all_reactions_50to60, threshold=10, title='50 < Age < 60', legend=False, verbose=True)
makePiePlot(all_counts_over65, all_reactions_over65, threshold=10, title='Age > 65', legend=False, verbose=True)


# In[ ]:





# In[25]:


topReactions_allAges = list(set(top10_reactions_under45 + top10_reactions_50to60 + top10_reactions_over65))
topReactions_allAges


# In[ ]:





# In[26]:


plt.xticks(rotation=45)
plt.rcParams.update({'font.size': 10})
plt.bar( top10_reactions_under45, top10_counts_under45)


# In[ ]:


#maybe think of some way to store original tuple as dict and then plot n_reactions/n_population for three age groups
# then compare a few common reactions across ages


# In[28]:


bins=np.arange(0, 50, 1)
plt.hist( all_counts_under45, bins=bins)
plt.xlabel('Effect Incidence')
plt.ylabel('N_effects')
plt.yscale('log')
#turns out most reactions only have < 10 recorded instances. probably focus on things that are much more common


# In[ ]:





# In[ ]:





# In[29]:


def drawTopReactionsForLabel(_df, columnKey, keyValue, threshold=10, verbose=False):
    """ Are different reactions reported in different countries?"""
    
    # *** 1. Select by label make flat list of reactions
    if 'Name' in columnKey:
        _df_key = _df_key[ _df_key.drugGenericName != 'No Generic Given']
        _df_key = _df_key[ (_df_key.drugGenericName.str.join('-').str.find(keyValue) != -1)]
    else:
        _df_key = _df[ (_df[ columnKey ] == keyValue) ]
    _reactions = list(flatten(_df_key['patientReactions']))

    # *** 2. Get unique counts/labels
    _theshKey, _all = countUnique(_reactions, threshold)
    _all_reactions, _all_counts = returnCountsAndLabels( _all)

    makePiePlot(_all_counts, _all_reactions, threshold=threshold, title='Top Reactions: {}'.format(keyValue), 
                legend=False, verbose=verbose)
    

def drawTopDrugsForLabel(_df, columnKey, keyValue, drugName, threshold=10, verbose=False):
    """ Are different reactions reported in different countries?"""
    
    # *** 1. Select by country make flat list of reactions
    _df_key = _df[ (_df[ columnKey ] == keyValue) ]
    _drugs = list(flatten(_df_key[ 'drug'+drugName]))

    # *** 2. Get unique counts/labels
    _threshKey, _all = countUnique(_drugs, threshold)
    _all_drugs, _all_counts = returnCountsAndLabels( _all)
    print(len(_df_key), len(_drugs), len(_all))

    
    makePiePlot(_all_counts, _all_drugs, threshold=threshold, title='Top Drug {}: {}'.format(drugName, keyValue), 
                legend=False, verbose=verbose)


# In[30]:


# *** 0. Find threshold break
countryThreshold = 0.925
counts = df3.reportercountry.value_counts().to_list()
countries = df3.reportercountry.value_counts().keys()

threshKey = returnThresholdPoint(counts, countries, countryThreshold)
         
# *** 1. Make counts/labels below threshold and store rest as 'Other'
for iCountry in countries[:threshKey]:
    drawTopReactionsForLabel( df3, 'reportercountry', iCountry)
    


# In[ ]:





# In[ ]:





# In[31]:


indicationThreshold =14
df3_indication = df3[ df3.drugIndication!=-1]
counts = df3_indication.drugIndication.value_counts().to_list()
indications = df3_indication.drugIndication.value_counts().keys()
reactions = df3_indication.patientReactions.value_counts().keys()
print(len(indications))
print(len(reactions))
threshKey = returnThresholdPoint(counts, indications, indicationThreshold, verbose=True)
topIndications = indications[:indicationThreshold].to_list()


# In[32]:


plt.hist(counts, bins=np.arange(0,400,10))
plt.yscale('log')
50/sum(counts)*100 # threshold, pick causes ~above this percentage
plt.show()

reactions_temp = list(flatten(df3.patientReactions))
threshKey, uniqueTemp = countUnique( reactions_temp)
makeEfficiencyCurve(uniqueTemp, 'All Reactions')

reactions_temp = list(flatten(df3.reportercountry))
threshKey, uniqueTemp = countUnique( reactions_temp)
makeEfficiencyCurve(uniqueTemp, 'Countries')


# In[38]:


# *** 1. Make pie charts of most common REACTIONS from a given INDICATION
for iIndication in indications[:indicationThreshold]:
    drawTopReactionsForLabel( df3, 'drugIndication', iIndication, threshold=10)


# In[45]:


# *** 2. Make pie charts of most common GENERIC_NAME from a given INDICATION
for iIndication in indications[:indicationThreshold]:
    drawTopDrugsForLabel( df3, 'drugIndication', iIndication, 'GenericName', threshold=.925)


# In[49]:


# *** 2. Make pie charts of most common BRAND_NAME from a given INDICATION
for iIndication in indications[:indicationThreshold]:
    drawTopDrugsForLabel( df3, 'drugIndication', iIndication, 'BrandName', threshold=.925)


# In[40]:


df3.drugGenericName


# In[50]:


# *** 3. Make pie charts of most common REACTION from a given GENERIC NAME
drugThreshold =100
generics = list(flatten(df3[ df3.drugGenericName != 'No Name Given'].drugGenericName))

# ** A. Make pairs of uniques
(unique, counts) = np.unique( generics, return_counts=True)
frequencies = np.asarray((unique, counts)).T
freq2 = sorted( frequencies, key=lambda pair: int(pair[1]), reverse=True) 
print( len(freq2) )
    
# ** B. Find threshold break
drugs, counts = returnCountsAndLabels( freq2)
threshKey = returnThresholdPoint(counts, drugs, drugThreshold, verbose=True)
    
plt.hist(counts, bins=np.arange(0, 200, 2))
plt.yscale('log')
plt.show()
print(50/len(drugs)) # ignore drugs when percentage ~beneath this level

makeEfficiencyCurve(freq2, 'Generics')

# ** C. Look at top drugs
topGenerics = drugs[:drugThreshold]
print(topGenerics)

# ** D. Make some plots 
for iGeneric in drugs[:drugThreshold]:
    drawTopReactionsForLabel( df3, 'drugGenericName', iGeneric, threshold=drugThreshold)
    


# In[ ]:





# In[51]:


# *** 4. Look at drug correlation
def returnOneHotVector(eventKeys, allKeys):
    """function to create one-hot encoded vector of drug names for calculating correlation"""
    
    _oneHotVector=[]
    for iKey in allKeys:
        
        #print(eventKeys, type(eventKeys))
        if iKey in eventKeys: # found name... probably
            _oneHotVector.append(1)
        else:                   # did not find name
            _oneHotVector.append(0)
    
    #if sum(_oneHotVector)>1:
    #    print( allDrugNames, eventDrugs, _oneHotVector )
    
    return _oneHotVector


def findCorrelatedDrugs(df_matrix, lowCorrThreshold= 0.25):
    """function that returns drug names if 0.1 < correlation < 1.0"""
    
    for iGeneric in df_matrix.columns:
        correlatedDrugs = df_matrix.index[ (np.abs(df_matrix[iGeneric]) > lowCorrThreshold) & (df_matrix[iGeneric]!=1.0)].to_list()
        
        for d in correlatedDrugs:
            shortWord, longWord = (d, iGeneric) if len(d)<len(iGeneric) else (iGeneric, d)
            if longWord.find(shortWord)==-1: # protection for repeated word:
                print('{} correlates with {}, Pearson = {:1.2f}'.format(iGeneric, d, df_matrix.at[iGeneric, d]))

                
def returnEncodedVector(uniqueKeysAndCounts, allKeys):
    """function to create one-hot encoded vector of drug names for calculating correlation"""
    
    _encodedVector=[]
    _uniqueKeys = np.array(uniqueKeysAndCounts)[:,0]
    _uniqueCounts = np.array(uniqueKeysAndCounts)[:,1]
    # i. create vector encoded by all keys
    for iKey in allKeys:
        if iKey in _uniqueKeys: # found name... probably
            #_keyIndex = list(uniqueKeys)[:,0]).index( iKey )
            _keyIndex = np.where(_uniqueKeys == iKey)
            _keyFreq = _uniqueCounts[ _keyIndex ]
            _encodedVector.append( int(_keyFreq) )
        else:                   # did not find name
            _encodedVector.append(0)
    
    # * ii. Turn result into numpy array
    _encodedVector = np.array(_encodedVector)
    # * iii. Normalize vector
    _encodedVector = _encodedVector / np.sqrt(sum(_encodedVector**2))

    return _encodedVector


# In[52]:



topGenerics = drugs[:drugThreshold]
#topGenerics = drugs
df4 = df3[ df3.drugGenericName!='No Name Given']
genericsVectors = df4.apply( lambda x: returnOneHotVector( x.drugGenericName, topGenerics), axis=1).to_list()
df4_corr = pd.DataFrame( genericsVectors, columns = topGenerics)
df4_corrMatrix = df4_corr.corr(method ='pearson') 


# In[89]:


smallKey='generic_name'

for p in np.arange(0, 10):
    print(p, len(df4['patient'][p]['drug']))
    for d in df4['patient'][p]['drug']:
        #print(d['openfda'].keys())
        if smallKey in d['openfda'].keys():
            print(d['openfda'][smallKey])
    
df4.drugGenericName.value_counts()

uniqueMultiDrugs = set(list(flatten(df4[ df4.drugSubstanceName.str.len()>1].drugSubstanceName.value_counts().keys())))
print(len(uniqueMultiDrugs), uniqueMultiDrugs)


# In[53]:


findCorrelatedDrugs( df4_corrMatrix)


# In[ ]:





# In[55]:


def compareOverlap(_encodedVectors, _key1, _key2):
    """ compare two encoded vectors"""

    _key1Vector = _encodedVectors[ _key1 ]
    _key2Vector = _encodedVectors[ _key2 ]
    
    print( '{}-{} Dot Product: {:1.3f}'.format( _key1, _key2, _key1Vector.dot(_key2Vector)))
    
    
# *** 5. Dot-product for determining country similarity
# ** A. Get list of all reactions and all countries
allReactions = set(list(flatten(df3.patientReactions)))
countries = df3.reportercountry.value_counts().keys()

# ** B. make dict of encoded vectors
encodedReactionVectors = {}
for iCountry in countries:
    print('Process {}'.format(iCountry))
    reactions_temp = list(flatten(df3[ df3.reportercountry==iCountry].patientReactions))
    threshKey, uniqueTemp = countUnique( reactions_temp)
    encoded = returnEncodedVector( uniqueTemp, allReactions)
    encodedReactionVectors[ iCountry ] = encoded

<h1>Header 1,title<h1>

compareOverlap( encodedReactionVectors, 'UNITED STATES', 'UNITED KINGDOM')
compareOverlap( encodedReactionVectors, 'UNITED STATES', 'AUSTRALIA')
compareOverlap( encodedReactionVectors, 'UNITED STATES', 'ITALY')
compareOverlap( encodedReactionVectors, 'UNITED STATES', 'SWITZERLAND')

compareOverlap( encodedReactionVectors, 'AUSTRALIA', 'UNITED KINGDOM')
compareOverlap( encodedReactionVectors, 'JAPAN', 'AUSTRALIA')
compareOverlap( encodedReactionVectors, 'ITALY', 'SWITZERLAND')


# In[ ]:





# In[ ]:


def vectorEncoder(patientReactions, allReactions):
    """function to create encoded vector as new column"""

    if drugCategory in patientInfo['drug'][0]:
        if expectedKey in patientInfo['drug'][0][ drugCategory]:
            dfValue = patientInfo['drug'][0][ drugCategory][ expectedKey ]
            #return '-'.join(dfValue)
            return dfValue
        else:
            return 'No Name Given'
    else:
        return 'No Name Given'
    
df4 = df3[ df3.drugIndication != -1]
allIndications = set(list(flatten(df4.drugIndication)))
df4['encodedIndicationVector'] = df4.apply( lambda x: returnOneHotVector( x.drugIndication, allIndications), axis=1)
#returnEncodedOneHotVector(df3.patientReactions[0], allReactions)


# In[ ]:


# *** 6. Try to k-means cluster the reactions
# ** A. Get list of all reactions
Sum_of_squared_distances = []
encodedIndicationVectorList = df4.encodedIndicationVector.to_list()
K = range(1,30)
for k in K:
    print (k)
    km = KMeans(n_clusters=k)
    km = km.fit(encodedIndicationVectorList)
    Sum_of_squared_distances.append(km.inertia_)

    


# In[ ]:


plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# In[ ]:





# In[ ]:


kmeans = KMeans(n_clusters=10, random_state=0).fit(encodedIndicationVectorList)
kmeans.labels_

df4['indicationClusterLabel'] = df4.apply( lambda x: kmeans.predict( [x.encodedIndicationVector]), axis=1)

#kmeans.predict( df4.encodedIndicationVector[0])


# In[ ]:


set(list(flatten(df4[ df4.indicationClusterLabel==9].drugIndication)))


# In[ ]:





# In[ ]:


df3.seriousness


# In[ ]:





# In[ ]:


df3_sDeath.drugIndication.value_counts().values
plt.hist( df3_sDeath.drugIndication.value_counts().values, bins=np.arange(0,50,1), alpha=.5)
plt.hist( df3_sDeath.patientReactions.value_counts().values, bins=np.arange(0,50,1), alpha=.5)

plt.yscale('log')


# In[ ]:


key, uniqueDeath = countUnique( list(flatten(  df3.dropna(subset=['seriousnessdeath']).patientReactions)))
plt.hist(np.array(uniqueDeath)[:,1].astype(np.int), bins=np.arange(0,50,1))
plt.yscale('log')
plt.show()
makeEfficiencyCurve( uniqueDeath, 'Adverse Reactions (Death)')
print( np.array(uniqueDisabling)[:15,0])


# In[ ]:


key, uniqueDisabling = countUnique( list(flatten(  df3.dropna(subset=['seriousnessdisabling']).patientReactions)))
plt.hist(np.array(uniqueDisabling)[:,1].astype(np.int), bins=np.arange(0,50,1))
plt.yscale('log')
plt.show()
makeEfficiencyCurve( uniqueDisabling, 'Adverse Reactions (Disabling)')
print( np.array(uniqueDisabling)[:15,0])


# In[ ]:


key, uniqueHospital = countUnique( list(flatten(  df3.dropna(subset=['seriousnesshospitalization']).patientReactions)))
plt.hist(np.array(uniqueHospital)[:,1].astype(np.int), bins=np.arange(0,50,1))
plt.yscale('log')
plt.show()
makeEfficiencyCurve( uniqueHospital, 'Adverse Reactions (Hospitalization)')
print( np.array(uniqueHospital)[:15,0])


# In[ ]:


key, uniqueThreat = countUnique( list(flatten(  df3.dropna(subset=['seriousnesslifethreatening']).patientReactions)))
plt.hist(np.array(uniqueThreat)[:,1].astype(np.int), bins=np.arange(0,50,1))
plt.yscale('log')
plt.show()
makeEfficiencyCurve( uniqueThreat, 'Adverse Reactions (Life Threatening)')
print( np.array(uniqueThreat)[:15,0])


# In[ ]:


key, uniqueHospital = countUnique( list(flatten(  df3.dropna(subset=['seriousnesshospitalization']).patientReactions)))
plt.hist(np.array(uniqueHospital)[:,1].astype(np.int), bins=np.arange(0,50,1))
plt.yscale('log')
plt.show()
makeEfficiencyCurve( uniqueHospital, 'Adverse Reactions (Hospitalization)')
print( np.array(uniqueHospital)[:15,0])

