import s_globalProps
from array import array
import math
import ROOTzoo
from sklearn.externals import joblib
from ROOT import TVector3, TLorentzVector, TFile, TObject
from sklearn.preprocessing import StandardScaler 
import ntpath, sys

treesDict=dict()
dirsDict=dict()
prefix = 'prob_'
years = ["2012", "2016"]

def toFourVector(px, py, pz, m):
    vec = TVector3(px, py, pz)
    return TLorentzVector(vec, math.sqrt(m*m + vec.Mag2()))

print 'Loading classifiers...'
classifiers = {}
for year in years:
    classifiers[year] = joblib.load( "classifier_%s.pkl" % year )

# Where are our files
pabloPath = '~/Documents/STBC-Drive/data/forPablo/'
basePath='~/Documents/STBC-Drive/data/sbenson/'
out_path = '~/Documents/STBC-Drive/data/sbenson/processed/'

# 2012 locations ########################
#run1streams = [ 'Data11_Up2_merged.root', 'Data11_Up1_merged.root', 'Data12_Dn1_merged_probNNpi0p2.root', 'Data12_Up1_merged_probNNpi0p2.root']
run1streams = [ 'Data11_Up1_merged.root', 'Data12_Dn1_merged_probNNpi0p2.root', 'Data12_Up1_merged_probNNpi0p2.root']
run1streams+= ['Data11_Dn2_merged.root', 'Data11_Up2_merged.root', 'Data12_Dn2_merged.root', 'Data12_Dn1_merged.root']
#
fileDict = {"2012" : [basePath+'largeProd_2012_bkg50.root']}
fileDict["2012"] += [pabloPath+n for n in run1streams]
# tree location in files 
trees = ['DecayTree'] + [ 'newtree' ] * 1 + [ 'DecayTree' ] * 2 + [ 'newtree' ] * 4
dirs = [ 'B2Xmumu' ] + [ None ] * 7 
treesDict["2012"] = trees
dirsDict['2012'] = dirs

# 2016 locations ########################
run2streams = [basePath+'BKGtuples/Bd2Xmumu_2016_BKG.root']
fileDict['2016'] = [basePath+'largeProd_2012_bkg50.root']
fileDict["2016"] += [n for n in run1streams]
# tree location in files 
trees = ['DecayTree']*2
dirs = [ 'B2Xmumu' ]*2 
treesDict["2016"] = trees
dirsDict['2016'] = dirs

# get our global properties
properties = s_globalProps.classifierProps()
print "Using properties:"
print properties
ranges = s_globalProps.propRanges()
print "Corresponding sanity ranges:"
print ranges

MJpsi = ROOTzoo.getMass( 'Jpsi' )
MPsi2S = ROOTzoo.getMass( 'psi2S' )

selection=s_globalProps.createSelection(properties,ranges)
print "Using sanity selection:"
print selection

for year in years:
    files = fileDict[year]
    trees = treesDict[year]
    dirs = dirsDict[year]

    # need scaling info
    dataSc = joblib.load( '%s_KstMuMu.p' % year )
    dataSc[ 1 ] = dataSc[ 1 ][ : len( dataSc[ 0 ] ) ]
    trainFraction = 0.75
    cutIndex = int( trainFraction * len( dataSc[ 1 ] ) )
    sigTrain = dataSc[ 0 ][ : cutIndex ] 
    # Create the scaler to preprocess the data
    scaler = StandardScaler(copy=True, with_mean=True, with_std=True).fit(sigTrain)
    
    for i, file, tree, dir in zip( xrange( len( files ) ), files, trees, dirs):
        print 'Opening file %s...' % ( file )
        f_data = TFile.Open( file )
        if not f_data:
            print "Unable to open file, exiting..."
            sys.exit()
            
        if dir is not None:
            temp = f_data
            f_data = f_data.Get( dir )
            
        t_data = f_data.Get( tree )
        
        selection+="&&(piplus_ProbNNpi>0.2)&&(piminus_ProbNNpi>0.2)"
        ntpath.basename("a/b/c")
        file = ntpath.basename(file)
        fileName = out_path + prefix + file
        cutFile = TFile( fileName, 'RECREATE' )
            
        cutTree = t_data.CopyTree( selection )
            
        entries = cutTree.GetEntries('')
            
        mCorrLeaf = array( 'd' )
        mCorrLeaf.append( 0 )
            
        probLeaf = array( 'd' )
        probLeaf.append( 0 )
        
        diMuonLeaf = array( 'd' )
        diMuonLeaf.append( 0 )
        
        p = ROOTzoo.fromTree( cutTree, 1.0 )
        B0 = p[ 'B0' ]
        pip = p[ 'piplus' ]
        pim = p[ 'piminus' ]
        mup = p[ 'muplus' ]
        mum = p[ 'muminus' ]
        
        mCorrBranch = cutTree.Branch( 'mCorr', mCorrLeaf, 'mCorr/D' )
        DiMuonBranch = cutTree.Branch( 'mumu_M', diMuonLeaf, 'mumu_M/D' )
        probBranch = cutTree.Branch( 'probB0', probLeaf, 'probB0/D' )
            
        for j in xrange(entries):
                    
            cutTree.GetEntry( j )
                    
            diMuonLeaf[0]=(mup.P4 + mum.P4).M()
            
            partial = mup.P4 + mum.P4 + pip.P4 + pim.P4
            partialFlight = TVector3(cutTree.B0_ENDVERTEX_X,cutTree.B0_ENDVERTEX_Y,cutTree.B0_ENDVERTEX_Z)-TVector3(cutTree.B0_OWNPV_X,cutTree.B0_OWNPV_Y,cutTree.B0_OWNPV_Z)
            #partialFlight = partialFlight.Unit()
            #
            dot = partial.Vect().Unit().Dot(partialFlight)
            #ptMiss = math.sqrt( 1 - dot**2 ) * partial.Pt()
            #print math.acos(dot)
            ptMiss = partial.Vect() - partialFlight * partial.Vect().Dot(partialFlight) * (1.0 / partialFlight.Mag2())
            ptMiss = ptMiss.Mag()
            #print ptMiss
            mCorrLeaf[ 0 ] = math.sqrt( max(partial.M(),0.0)**2 + ptMiss**2 ) + ptMiss
                    
            data = [ getattr( cutTree, k ) for k in properties ]
            # transform the sample
            data = scaler.transform([data])

            probLeaf[ 0 ] = classifiers[year].predict_proba( data )[ 0 ][ 0 ]
                    
            mCorrBranch.Fill()
            probBranch.Fill()
            DiMuonBranch.Fill()
            
        print "Saving file %s..." % fileName
        cutTree.Write( '', TObject.kOverwrite )
        cutFile.Close()
