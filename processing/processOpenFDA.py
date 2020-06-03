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
#  X --> turn into class.
# --> use class to run over all files OR run over subset (e.g. one year) to make time-trends
#  --> make time-trend of adverse events for a particular drug. split by adverse categories
#  X --> make encoded vector of adverse effects. run k-means cluster to see if "areas" emerge- could classify drugs. DOESN'T WORK
#  --> drugWatcher : code that uses apply(lambda) where lambda calculate timeToDouble/timeToSomeThreshold 



import os
import pandas as pd
from pandas.core.common import flatten
import numpy as np
import json
import zipfile
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans


class processOpenFDA:

    def __init__(self, inputSource, verbose=False):
        """auto-intialization of processOpenFDA class"""

        self.inputSource = inputSource
        self.df_all = []
        self.verbose= verbose

    def loadTables(self):
        """ load either single file or all files in a given directory """

        #filename2='/home/btannenw/Desktop/life/adverseDrugFDA/data/2018-Q4/drug-event-0001-of-0029.json.zip' #DELETEME
        #filename='/home/btannenw/Desktop/life/adverseDrugFDA/data/2011-Q3/drug-event-0001-of-0012.json.zip' #DELETEME
        if os.path.isfile( self.inputSource ): # user passes single file
            self.loadSingleFile( self.inputSource)
        elif os.path.isdir( self.inputSource): # user passes a directory
            nfiles = 0
            for dirpath, dirnames, filenames in os.walk( self.inputSource ):
                for filename in [f for f in filenames if f.endswith(".json.zip")]:
                    #print (os.path.join(dirpath, filename))
                    self.loadSingleFile( os.path.join(dirpath, filename) )
                for filename in [f for f in filenames if f.endswith(".csv")]:
                    if 'All-other' in filename:
                        continue
                    if int(filename.split('-')[0])<2009:
                        continue
                    #if nfiles > 100:
                    if nfiles > 10:
                        continue

                    print ("nfiles: {}, {}".format(nfiles, os.path.join(dirpath, filename)))
                    
                    self.loadSingleFile( os.path.join(dirpath, filename) )
                    nfiles += 1
                    
                                            
        return

    def loadData(self):
        """load data of single file for example"""
        self.inputSource = '/home/btannenw/Desktop/life/adverseDrugFDA/data/2011-Q3/drug-event-0001-of-0012.json.zip'
        self.loadTables()
        self.addColumns()
        
        return
    
    def processSequentialFilesFromDirectory(self, outputDir):
        """ load either single file or all files in a given directory """

        if ( not os.path.exists(outputDir) ):
            print( "Specified output directory ({0}) DNE.\nCREATING NOW".format(outputDir))
            os.system("mkdir {0}".format(outputDir))
        
        if os.path.isdir( self.inputSource): # user passes a directory
            for dirpath, dirnames, filenames in os.walk( self.inputSource ):
                for filename in [f for f in filenames if f.endswith(".json.zip")]:
                    print (os.path.join(dirpath, filename))
                    self.df_all = []
                    self.loadSingleFile( os.path.join(dirpath, filename) )
                    print(filename, len(self.df_all))
                    self.addColumns()
                    self.saveColumnsToCSV(outputDir, os.path.join(dirpath, filename), ['drugGenericName', 'patientReactions'])
                        
        return


    def loadSingleFile(self, filename):
        """ function to load a single file"""
        d = []
        data = []

        # *** 1. Open single file from zipped json
        if 'json.zip' in filename:
            with zipfile.ZipFile(filename, 'r') as z:
                for filename in z.namelist():  
                    if self.verbose:
                        print(filename)  
                    with z.open(filename) as f:  
                        data = f.read()  
                        d = json.loads(data.decode("utf-8"))  
                        _df = pd.DataFrame(d['results'])
        elif '.csv' in filename:
            _df = pd.read_csv( filename )
                
        # *** 2. Append to global df_all
        if type(self.df_all)==list:
            self.df_all = _df
        else:
            self.df_all = self.df_all.append(_df, ignore_index=True, sort=False)

        return
    

    def returnThresholdPoint(self, counts, labels, threshold, verbose=False):
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


    def makePiePlot(self, counts, labels, threshold=0.9, legend=True, title='', verbose=False):
        """ make pie plot that lumps tail above some threshold into 'other' category """
    
        # *** 0. Find threshold break
        threshKey = self.returnThresholdPoint(counts, labels, threshold, verbose)
     
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

        return

    
    def returnCountsAndLabels(self, freqPair):
        """return easily digestible list of reactions and counts"""
    
        _reactions = [ pair[0] for pair in freqPair ]
        _counts    = [ int(pair[1]) for pair in freqPair ]
        
        return _reactions, _counts



    def countUnique(self, reactionsList, threshold=0.90, verbose=False):
    
        # *** 0. Make pairs of uniques
        (unique, counts) = np.unique( reactionsList, return_counts=True)
        frequencies = np.asarray((unique, counts)).T
        #print(frequencies)
        freq2 = sorted( frequencies, key=lambda pair: int(pair[1]), reverse=True) 
        #print(freq2[:10])
        #print( len(frequencies) )
        
        # *** 1. Find threshold break
        _reactions, _counts = self.returnCountsAndLabels( freq2)
        _threshKey = self.returnThresholdPoint(_counts, _reactions, threshold, verbose)
        
        
        return _threshKey, freq2



    def imputePrimarySource(self, primarySourceInfo, expectedKey):
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
            return np.nan
    

    def imputePatientReaction(self, patientInfo, expectedKey):
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
            return np.nan

        
    def getPatientData(self, patientInfo, categoryKey, dataKey):
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


    def getDrugData(self, patientInfo, categoryKey, subCategoryKey, dataKey):
        """function to impute some dummy info for missing fields"""

        #df.patient[60]['drug'][0]['openfda']['substance_name']  # DELETEME
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
                return 'No Name Listed'
            if dataKey=='brand_name':
                return 'No Name Listed'
        
            return np.nan
    
        return set(list(flatten(dfValue)))


    def addColumns(self):
        """ add some columns to dataframe via apply"""

        #print("Number entries (all): {}".format(len(self.df_all)))
        self.df_all = self.df_all.dropna(subset=['primarysource', 'patient'])
        #print("Number entries (w/ patient data): {}".format(len(self.df_all)))

        self.df_all['reportercountry'] = self.df_all.apply( lambda x: self.imputePrimarySource( x['primarysource'], 'reportercountry'), axis=1)
        self.df_all['reporterqualification'] = self.df_all.apply( lambda x: self.imputePrimarySource( x['primarysource'], 'qualification'), axis=1)
        self.df_all['patientOnsetAge'] = self.df_all.apply( lambda x: self.imputePrimarySource( x['patient'], 'patientonsetage'), axis=1)
        self.df_all['patientReactions'] = self.df_all.apply( lambda x: self.imputePatientReaction( x['patient'], 'reaction'), axis=1)
        self.df_all['drugIndication'] = self.df_all.apply( lambda x: self.getPatientData( x['patient'], 'drug', 'drugindication'), axis=1)
        self.df_all['drugSubstanceName'] = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'openfda', 'substance_name'), axis=1)
        self.df_all['drugGenericName'] = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'openfda', 'generic_name'), axis=1)
        self.df_all['drugBrandName'] = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'openfda', 'brand_name'), axis=1)
        self.df_all['pharm_class_moa'] = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_moa'), axis=1)
        self.df_all['pharm_class_pe']  = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_pe'), axis=1)
        self.df_all['pharm_class_cs']  = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_cs'), axis=1)
        self.df_all['pharm_class_epc'] = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'openfda', 'pharm_class_epc'), axis=1)
        self.df_all['drugActiveSubstance'] = self.df_all.apply( lambda x: self.getDrugData( x['patient'], 'drug', 'activesubstance', 'activesubstancename'), axis=1)

        self.df_all = self.df_all.dropna(subset=['drugSubstanceName'])
        #print("Number entries (w/ substance data): {}".format(len(self.df_all)))


    def makeCountryPlots(self, threshold=0.925, verbose=False):
        """ function for some simple country plots"""

        # *** 0. Drop NaN and get counts/countries
        #_df = self.df_all.dropna(subset=['reportercountry'])
        _df = self.df_all.dropna(subset=['reportercountry'])
        countryCounts = _df.reportercountry.value_counts()
        counts = countryCounts.to_list()
        countries = countryCounts.keys()
        
        self.makePiePlot(counts, countries, threshold=threshold, title='% Response by Country', verbose=verbose)

        if self.verbose:
            print(counts)
        print(countries)

        # *** 1. Find threshold break
        countryThreshold = 0.925        
        threshKey = self.returnThresholdPoint(counts, countries, countryThreshold)
        
        # *** 2. Make counts/labels below threshold and store rest as 'Other'
        for iCountry in countries[:threshKey]:
            self.drawTopReactionsForLabel( _df, 'reportercountry', iCountry)

        # *** 3. Threshold efficiency
        reactions_temp = list(flatten(_df.reportercountry))
        threshKey, uniqueTemp = self.countUnique( reactions_temp)
        self.makeEfficiencyCurve(uniqueTemp, 'Countries')

        # *** 4. Dot-product for determining country similarity
        # ** A. Get list of all reactions and all countries
        allReactions = set(list(flatten(_df.patientReactions)))

        # ** B. make dict of encoded vectors
        encodedReactionVectors = {}
        for iCountry in countries:
            if self.verbose:
                print('Process {}'.format(iCountry))
            reactions_temp = list(flatten(_df[ _df.reportercountry==iCountry].patientReactions))
            threshKey, uniqueTemp = self.countUnique( reactions_temp)
            encoded = self.returnEncodedVector( uniqueTemp, allReactions)
            encodedReactionVectors[ iCountry ] = encoded

        # ** C. make comparisons
        self.compareOverlap( encodedReactionVectors, 'UNITED STATES', 'UNITED KINGDOM')
        self.compareOverlap( encodedReactionVectors, 'UNITED STATES', 'AUSTRALIA')
        self.compareOverlap( encodedReactionVectors, 'UNITED STATES', 'ITALY')
        self.compareOverlap( encodedReactionVectors, 'UNITED STATES', 'SWITZERLAND')
        
        self.compareOverlap( encodedReactionVectors, 'AUSTRALIA', 'UNITED KINGDOM')
        self.compareOverlap( encodedReactionVectors, 'JAPAN', 'AUSTRALIA')
        self.compareOverlap( encodedReactionVectors, 'ITALY', 'SWITZERLAND')

        return

    
    def makeEfficiencyCurve(self, uniqueCountsAndKeys, keyType, maxKey=-1):
        """function to make efficiency curve and find x of plateau"""
        _total = sum(np.array(uniqueCountsAndKeys)[:,1].astype(np.int))
    
        _maxKey = maxKey if maxKey != -1 else len(np.array(uniqueCountsAndKeys)[:,1])
        _topX  = np.arange(0, _maxKey, 1).astype(np.int)
        _percentCaptured = [ sum(np.array(uniqueCountsAndKeys)[:x,1].astype(np.int))/ _total for x in _topX]
        
        plt.plot(_topX, _percentCaptured)
        plt.xlabel('Number of {}'.format(keyType))
        plt.ylabel('% Included')
        plt.show()

        return

    
    def makeAgePlot(self, _df):
        """ function for age plot"""
        # assume ages >100 are something weird
        plt.hist(_df.patientOnsetAge, bins=np.arange(0, 100, 10))
        #print(len(ages), max(ages), min(ages))
        # adverse affects peak around 50... is that weird?
        plt.show()

        return



    def reactionsByAge(self):
        """historical plots of reactions by age. inherited from exploration notebook"""
        #maybe think of some way to store original tuple as dict and then plot n_reactions/n_population for three age groups
        # then compare a few common reactions across ages

        
        _df = self.df_all.dropna(subset=['patientReactions'])
        reactions  = list(flatten( _df.patientReactions.to_list()))
        #reactions2 =[ reaction.get('reactionmeddrapt') for patient in df.patient for reaction in patient['reaction']]
        #get list of all reactions
        print( len(reactions), len(set(reactions)))
        # shows that there's a lot of reactions (~35k) and a lot of unique reactions (~3k). but since the number of unique
        # is so much smaller, that means there should be a lot of correlation --> matrix

        _df_under45_reactions = list(flatten(_df[ (_df.patientOnsetAge<45) & (_df.patientOnsetAge>-1) ]['patientReactions']))
        _df_50to60_reactions  = list(flatten(_df[ (_df.patientOnsetAge>50) & (_df.patientOnsetAge<60) & (_df.patientOnsetAge>-1) ]['patientReactions']))
        _df_over65_reactions  = list(flatten(_df[ (_df.patientOnsetAge>65) & (_df.patientOnsetAge>-1) ]['patientReactions']))
        
        threshold=10

        threshKey_under45, all_under45 = self.countUnique(_df_under45_reactions, threshold)
        all_reactions_under45, all_counts_under45 = self.returnCountsAndLabels( all_under45)

        threshKey_50to60, all_50to60 = self.countUnique(_df_50to60_reactions, threshold)
        all_reactions_50to60, all_counts_50to60 = self.returnCountsAndLabels( all_50to60)

        threshKey_over65, all_over65 = self.countUnique(_df_over65_reactions, threshold)
        all_reactions_over65, all_counts_over65 = self.returnCountsAndLabels( all_over65)

        self.makePiePlot(all_counts_under45, all_reactions_under45, threshold=10, title='Age < 45', legend=False, verbose=True)
        self.makePiePlot(all_counts_50to60, all_reactions_50to60, threshold=10, title='50 < Age < 60', legend=False, verbose=True)
        self.makePiePlot(all_counts_over65, all_reactions_over65, threshold=10, title='Age > 65', legend=False, verbose=True)

        # see set() of unqiue top reactions by age. probably unnecessary (DELTEME?)
        #topReactions_allAges = list(set(top10_reactions_under45 + top10_reactions_50to60 + top10_reactions_over65))
        #topReactions_allAges

        # make bar graph of top reactions. not very communicative (DELETEME?)
        #plt.xticks(rotation=45)
        #plt.rcParams.update({'font.size': 10})
        #plt.bar( top10_reactions_under45, top10_counts_under45)

        # DELETEME?
        bins=np.arange(0, 50, 1)
        plt.hist( all_counts_under45, bins=bins)
        plt.xlabel('Effect Incidence (Under 45)')
        plt.ylabel('N_effects')
        plt.yscale('log')
        plt.show()
        #turns out most reactions only have < 10 recorded instances. probably focus on things that are much more common

        self.makeEfficiencyCurve(all_under45, 'All Reactions (Under 45)')
                
        return


    def drawTopReactionsForLabel(self, _df, columnKey, keyValue, threshold=10, verbose=False):
        """ Are different reactions reported in different countries?"""

        # *** 1. Select by label make flat list of reactions
        if 'Name' in columnKey:
            _df = _df[ _df.drugGenericName != 'No Generic Given']
            _df = _df[ (_df.drugGenericName.str.join('-').str.find(keyValue) != -1)]
        else:
            _df = _df[ (_df[ columnKey ] == keyValue) ]
        _reactions = list(flatten(_df['patientReactions']))
        #if type(_df['patientReactions'][0])==str:
        #    _reactions = list(flatten(_df['patientReactions'].to_list))
        
        # *** 2. Get unique counts/labels
        _theshKey, _all = self.countUnique(_reactions, threshold)
        _all_reactions, _all_counts = self.returnCountsAndLabels( _all)

        self.makePiePlot(_all_counts, _all_reactions, threshold=threshold, title='Top Reactions: {}'.format(keyValue), 
                    legend=False, verbose=verbose)

        return
    

    def drawTopDrugsForLabel(self, _df, columnKey, keyValue, drugName, threshold=10, verbose=False):
        """ Are different reactions reported in different countries?"""
    
        # *** 1. Select by country make flat list of reactions
        _df_key = _df[ (_df[ columnKey ] == keyValue) ]
        _drugs = list(flatten(_df_key[ 'drug'+drugName]))

        # *** 2. Get unique counts/labels
        _threshKey, _all = self.countUnique(_drugs, threshold)
        _all_drugs, _all_counts = self.returnCountsAndLabels( _all)
        if self.verbose:
            print(len(_df_key), len(_drugs), len(_all))

        # *** 3. Make pie chart
        self.makePiePlot(_all_counts, _all_drugs, threshold=threshold, title='Top Drug {}: {}'.format(drugName, keyValue), 
                legend=False, verbose=verbose)

        return

    
    def drugIndicationStudies(self, depVariable):
        """ make a bunch of plots for slices by indication"""

        indicationThreshold =14
        _df = self.df_all.dropna(subset=['drugIndication'])
        counts = _df.drugIndication.value_counts().to_list()
        indications = _df.drugIndication.value_counts().keys()
        reactions = _df.patientReactions.value_counts().keys()
        if self.verbose:
            print(len(indications))
            print(len(reactions))
        threshKey = self.returnThresholdPoint(counts, indications, indicationThreshold, verbose=True)
        topIndications = indications[:indicationThreshold].to_list()


        plt.hist(counts, bins=np.arange(0,400,10))
        plt.yscale('log')
        50/sum(counts)*100 # threshold, pick causes ~above this percentage
        plt.xlabel('Drug Indications')
        plt.ylabel('Unique Counts')
        plt.show()

        reactions_temp = list(flatten(_df.patientReactions))
        threshKey, uniqueTemp = self.countUnique( reactions_temp)
        self.makeEfficiencyCurve(uniqueTemp, 'All Reactions')


        # *** 1. Make pie charts of most common REACTIONS from a given INDICATION
        if depVariable == 'reactions':
            for iIndication in indications[:indicationThreshold]:
                self.drawTopReactionsForLabel( _df, 'drugIndication', iIndication, threshold=10)
            
        # *** 2. Make pie charts of most common GENERIC_NAME from a given INDICATION
        if depVariable == 'genericNames':
            for iIndication in indications[:indicationThreshold]:
                self.drawTopDrugsForLabel( _df, 'drugIndication', iIndication, 'GenericName', threshold=.925)

        # *** 3. Make pie charts of most common BRAND_NAME from a given INDICATION
        if depVariable == 'brandNames':
            for iIndication in indications[:indicationThreshold]:
                self.drawTopDrugsForLabel( _df, 'drugIndication', iIndication, 'BrandName', threshold=.925)

        return

    
    def genericNameStudies(self):
        """ make a bunch of plots for slices by generic name"""          

        # *** 0. Make pie charts of most common REACTION from a given GENERIC NAME
        drugThreshold =100
        generics = list(flatten(self.df_all[ self.df_all.drugGenericName != 'No Name Listed'].drugGenericName))

        # ** A. Make pairs of uniques
        _threshKey, _all = self.countUnique(generics, drugThreshold)
            
        # ** B. Find threshold break
        drugs, counts = self.returnCountsAndLabels( _all)
        threshKey = self.returnThresholdPoint(counts, drugs, drugThreshold, verbose=True)
    
        plt.hist(counts, bins=np.arange(0, 200, 2))
        plt.yscale('log')
        plt.show()
        print(50/len(drugs)) # ignore drugs when percentage ~beneath this level

        self.makeEfficiencyCurve(_all, 'Generics')

        # ** C. Look at top drugs
        topGenerics = drugs[:drugThreshold]
        print(topGenerics)

        # ** D. Make some plots 
        for iGeneric in drugs[:drugThreshold]:
            self.drawTopReactionsForLabel( self.df_all, 'drugGenericName', iGeneric, threshold=drugThreshold)
    
        return

    

    def returnOneHotVector(self, eventKeys, allKeys):
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


    def findCorrelatedDrugs(self, df_matrix, lowCorrThreshold= 0.25):
        """function that returns drug names if 0.1 < correlation < 1.0"""
    
        for iGeneric in df_matrix.columns:
            correlatedDrugs = df_matrix.index[ (np.abs(df_matrix[iGeneric]) > lowCorrThreshold) & (df_matrix[iGeneric]!=1.0)].to_list()
        
            for d in correlatedDrugs:
                shortWord, longWord = (d, iGeneric) if len(d)<len(iGeneric) else (iGeneric, d)
                if longWord.find(shortWord)==-1: # protection for repeated word:
                    print('{} correlates with {}, Pearson = {:1.2f}'.format(iGeneric, d, df_matrix.at[iGeneric, d]))

        return

    
    def returnEncodedVector(self, uniqueKeysAndCounts, allKeys):
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


    def genericCorrelation(self):
        """ calculate some correlations between generics using inner product"""

        _df = self.df_all
        # *** 0. Make pie charts of most common REACTION from a given GENERIC NAME
        drugThreshold =100
        generics = list(flatten( _df[ _df.drugGenericName != 'No Name Listed'].drugGenericName))

        # ** A. Make pairs of uniques
        _threshKey, _all = self.countUnique(generics, drugThreshold)
            
        # ** B. Find threshold break
        drugs, counts = self.returnCountsAndLabels( _all)

        topGenerics = drugs[:drugThreshold]
        #topGenerics = drugs
        _df = _df[ _df.drugGenericName!='No Name Listed']
        genericsVectors = _df.apply( lambda x: self.returnOneHotVector( x.drugGenericName, topGenerics), axis=1).to_list()
        _df_corr = pd.DataFrame( genericsVectors, columns = topGenerics)
        _df_corrMatrix = _df_corr.corr(method ='pearson') 

        self.findCorrelatedDrugs( _df_corrMatrix)


    def findMultiDrugs(self):
        """ code stub for finding list of drug combinations from multi-drug reports"""
        #smallKey='generic_name'
        #
        #for p in np.arange(0, 10):
        #    print(p, len(df4['patient'][p]['drug']))
        #    for d in df4['patient'][p]['drug']:
        #        #print(d['openfda'].keys())
        #        if smallKey in d['openfda'].keys():
        #            print(d['openfda'][smallKey])
        #    
        #df4.drugGenericName.value_counts()
        #
        #uniqueMultiDrugs = set(list(flatten(df4[ df4.drugSubstanceName.str.len()>1].drugSubstanceName.value_counts().keys())))
        #print(len(uniqueMultiDrugs), uniqueMultiDrugs)
        return
    
        
    def compareOverlap(self, _encodedVectors, _key1, _key2):
        """ compare two encoded vectors"""

        _key1Vector = _encodedVectors[ _key1 ]
        _key2Vector = _encodedVectors[ _key2 ]
    
        print( '{}-{} Dot Product: {:1.3f}'.format( _key1, _key2, _key1Vector.dot(_key2Vector)))
    
        return
    
        
    def vectorEncoder(self, patientReactions, allReactions):
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

        return


    def devKMeansCluster(self):
        """ start working on k-means clustering using encoded vectors"""
        _df = self.df_all.dropna(subset=['drugIndication'])

        allIndications = set(list(flatten(_df.drugIndication)))
        _df['encodedIndicationVector'] = _df.apply( lambda x: self.returnOneHotVector( x.drugIndication, allIndications), axis=1)
        #returnEncodedOneHotVector(df3.patientReactions[0], allReactions)

        # *** 6. Try to k-means cluster the reactions
        # ** A. calculate fit for specificed n_clusters
        Sum_of_squared_distances = []
        encodedIndicationVectorList = _df.encodedIndicationVector.to_list()
        K = range(1,30)
        for k in K:
            print (k)
            km = KMeans(n_clusters=k)
            km = km.fit(encodedIndicationVectorList)
            Sum_of_squared_distances.append(km.inertia_)

        # ** B. Show variance by cluster
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

        # ** C. Make some k-means prediction
        kmeans = KMeans(n_clusters=10, random_state=0).fit(encodedIndicationVectorList)
        kmeans.labels_

        _df['indicationClusterLabel'] = _df.apply( lambda x: kmeans.predict( [x.encodedIndicationVector]), axis=1)
        #kmeans.predict( df4.encodedIndicationVector[0])
        set(list(flatten(_df[ _df.indicationClusterLabel==9].drugIndication)))

        return


    def devSeriousnessCategory(self, category):
        """dev function to start playing around with seriousness category"""

        _df = self.df_all.dropna(subset=['seriousness{}'.format(category)])
        key, unique = self.countUnique( list(flatten(  _df.patientReactions)))

        plt.hist(np.array(unique)[:,1].astype(np.int), bins=np.arange(0,50,1))
        plt.yscale('log')
        plt.show()
        
        self.makeEfficiencyCurve( unique, 'Adverse Reactions ({})'.format(category))
        print( np.array(unique)[:15,0])

        plt.hist( _df.drugIndication.value_counts().values, bins=np.arange(0,50,1), alpha=.5)
        plt.hist( _df.patientReactions.value_counts().values, bins=np.arange(0,50,1), alpha=.5)
        plt.yscale('log')
        plt.show()

        return


    def runOverSeriousness(self):
        """ top-level fucntion for running over different seriousness categories"""

        self.devSeriousnessCategory('death')
        self.devSeriousnessCategory('disabling')
        self.devSeriousnessCategory('hospitalization')
        self.devSeriousnessCategory('lifethreatening')



    def returnGenericEncoded(self, testGeneric):
    
        #make reaction fingerprint for drug (active substance)
        _df = self.df_all.dropna(subset=['drugGenericName'])
        print("Number generic names/combos: {}".format(len(_df.drugGenericName.value_counts().keys())))

        _df = _df[ (_df.drugGenericName.str.join('-').str.find(testGeneric) != -1)]
        _reactions = list(flatten(_df['patientReactions']))
        #if type(_df['patientReactions'][0])==str:
        #_reactions = list(flatten(_df['patientReactions'].to_list))

        self.drawTopReactionsForLabel( _df, 'drugGenericName', testGeneric, threshold=100)
        
        # *** 2. Get unique counts/labels              
        print("Number of unique reactions to {}: {}".format(testGeneric, len(_reactions)))
        _theshKey, _uniqueGeneric = self.countUnique(_reactions, 10)

        # *** 3. Make encdoed vector
        _all = set(list(flatten(self.df_all.patientReactions)))
        encoded = self.returnEncodedVector( _uniqueGeneric, _all)

        print( encoded.dot(encoded), sum(encoded))
        print(encoded)    
    
        return encoded


    def saveColumnsToCSV(self, dirName, fileName, columnsToSave):
        """ function to save some minimal amount of data in .csv"""

        print(fileName.split('/'))
        outName =  fileName.split('/')[7] + '_' + fileName.split('/')[8].split('drug-event-')[1].split('.')[0]
        
        _df = self.df_all[ columnsToSave ]
        _df = _df.dropna(subset=columnsToSave)
        _df.to_csv('{}/{}_{}.csv'.format(dirName, outName, '-'.join(columnsToSave)), index=False)

        return
                    
