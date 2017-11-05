# Which variables have we used to make our classifier
def classifierProps():
    properties = [ 'B0_ENDVERTEX_CHI2', 'B0_FDCHI2_OWNPV', 'B0_PT', 'KS0_PT', 'KS0_FDCHI2_ORIVX' ]
    muon_properties = [ 'ORIVX_CHI2', 'IPCHI2_OWNPV', 'P', 'PT', 'TRACK_CHI2NDOF' ]
    pion_properties = [ 'P', 'PT', 'TRACK_CHI2NDOF' ]
    #
    for s in [ 'muplus', 'muminus' ]:
        for k in muon_properties:
            properties.append( '_'.join( ( s, k ) ) )
    for s in [ 'piplus', 'piminus' ]:
        for k in pion_properties:
            properties.append( '_'.join( ( s, k ) ) )
    return properties

def propLatex():
    titles = { 'B0_ENDVERTEX_CHI2' : 'B_{d} v. #chi^{2}', 
              'B0_FDCHI2_OWNPV' : 'B_{d} FD #chi^{2}', 
              'B0_PT' : 'B_{d} p_{T} [MeV]', 
              'KS0_PT' : 'K_{S} p_{T} [MeV]', 
              'KS0_FDCHI2_ORIVX' : 'K_{S} FD #chi^{2}'}
    #
    muprops = ['ORIVX_CHI2', 'IPCHI2_OWNPV', 'P', 'PT', 'TRACK_CHI2NDOF']
    parts = [ 'piplus', 'piminus', 'muplus', 'muminus' ]
    parts_vals = ['#pi^{+} ', '#pi^{-} ', '#mu^{+} ', '#mu^{-} ']
    for p in muprops:
        for pa,pv in zip(parts,parts_vals):
            key = pa + '_' + p
            if p=='P':
                val=pv+' p [MeV]'
            if p=='PT':
                val=pv+' p_{T} [MeV]'
            if 'ORIV' in p:
                val=pv+' PV #chi^{2}'
            if 'IP' in p:
                val=pv+' IP #chi^{2}'
            if 'TRACK' in p:
                val=pv+' track #chi^{2}'
            titles[key]=val
    return titles

def propRanges():
    ranges = [ [ 0, 10 ], [ 0, 25e3 ], [0,20e3] , [ 0, 10e3 ], [ 0,25e3 ] ]
    muon_ranges = [ [ 0, 20 ], [ 0, 5e3 ], [ 0, 200000 ], [0,12e3], [ 0.35, 1.9 ] ]
    pion_ranges = [ [ 0, 100e3 ], [ 0, 6000 ], [0.35, 1.9 ] ]
    ranges+=muon_ranges
    ranges+=muon_ranges
    ranges+=pion_ranges
    ranges+=pion_ranges
    return ranges

def createSelection(props,ranges):
    selString=""

    assert len(props)==len(ranges)
    
    for n in range(len(props)):
        selString += "(%s > %s)&&(%s < %s)&&" % (props[n],ranges[n][0],props[n],ranges[n][1])
    selString=selString.rstrip("&&")
    return selString
