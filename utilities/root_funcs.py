import ROOT
from math import sqrt


def open_root(fname, treename):
    f = ROOT.TFile.Open(fname)
    if not f:
        print "open_root: Unable to open file, exiting..."
        exit()
    t = f.Get(treename)
    if not t:
        print "open_root: Tree not found, exiting..."
        exit()
    return f, t


def to_4vector(px, py, pz, m):
    vec = ROOT.TVector3(px, py, pz)
    return ROOT.TLorentzVector(vec, sqrt(m*m + vec.Mag2()))


def to_4vector_tree(tree, rootname, m):
    px = getattr(tree, rootname+'_PX')
    py = getattr(tree, rootname+'_PY')
    pz = getattr(tree, rootname+'_PZ')
    vec = ROOT.TVector3(px, py, pz)
    return ROOT.TLorentzVector(vec, sqrt(m*m + vec.Mag2()))
