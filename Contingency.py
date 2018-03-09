import sys
import math
import random
import array
import numpy as np
from scipy import stats
import Utils

# import rpy2.robjects as robjects
# r = robjects.r

from scipy.stats import norm


class Contingency:
    def __init__(self):
        self.n = 0
        self.counts = {} # (row, col) -> count
        self.marginalsRow = {} # val -> count
        self.marginalsCol = {} # val -> count

    def __add__(self, other):
        for tup, count in other.counts.items():
            self.counts[tup] = self.counts.get(tup, 0) + count
            
    def sample(self, num):
        contNew = Contingency()

        keys = self.counts.keys()
        keyCurrIdx = 0
        keyCurr = keys[0]
        keyCurrCutoff = self.counts[keyCurr]
        keyCurrCount = 0

        samp = random.sample(xrange(self.n), num)
        samp.sort()
        for x in samp:
            if (i < keyCurrCutoff):
                keyCurrCount += 1
            else:
                contNew.counts[keyCurr] = keyCurrCount
                keyCurrIdx += 1
                keyCurr = keys[keyCurrIdx]
                keyCurrCutoff += self.counts[keyCurr]
                keyCurrCount = 1
        contNew.counts[keyCurr] = keyCurrCount
        return contNew
        
                

    def add(self, valRow, valCol):
        self.n += 1
        valTup = (valRow, valCol)
        self.marginalsRow[valRow] = self.marginalsRow.get(valRow, 0) + 1
        self.marginalsCol[valCol] = self.marginalsCol.get(valCol, 0) + 1
        self.counts[valTup] = self.counts.get(valTup, 0) + 1
        
    def chisq(self):
        ret = 0.0
        for valRow, marginalRow in self.marginalsRow.items():        
            for valCol, marginalCol in self.marginalsCol.items():
                valTup = (valRow, valCol)
                observed = self.counts.get(valTup, 0)
                expected = marginalRow*marginalCol / float(self.n)
                ret += (observed - expected)**2 / expected
        return ret
        
    def gstat(self):
        ret = 0.0
        for valRow, marginalRow in self.marginalsRow.items():        
            for valCol, marginalCol in self.marginalsCol.items():
                valTup = (valRow, valCol)
                observed = self.counts.get(valTup, 0)
                if (observed > 0):
                    expected = marginalRow*marginalCol / float(self.n)
                    ret += 2 * observed * math.log(observed/expected)
        return ret
    
    def dof(self):
        return (len(self.marginalsRow) - 1)*(len(self.marginalsCol) - 1)
        
    def levels(self):
        return (len(self.marginalsRow), len(self.marginalsCol))
        
    def chisqPval(self):
        return chiSquareP(self.chisq(), self.dof())    
    def gstatPval(self):
        return chiSquareP(self.gstat(), self.dof())    
    def corrContCoeff(self): # from page 482 in Lothar Sachs
        chisq = self.chisq()
        r = min(len(self.marginalsRow), len(self.marginalsCol))
        ccMax = pow(float(r-1)/r, 0.5)
        cc = pow(chisq / (self.n + chisq), 0.5)
        return cc/ccMax

    def chisqValPval(self):
        stat = self.chisq()
        return ( stat, chiSquareP(stat, self.dof()) )    
    def gstatValPval(self):
        stat = self.gstat()
        return ( stat, chiSquareP(stat, self.dof()) )    


    def dump(self, norm=False):
        return self.dumpFromDicts(self.marginalsRow, self.marginalsCol, self.counts, norm)



    # this is broken out so subclasses with different data structures can use it
    def dumpFromDicts(self, marginalsRow, marginalsCol, counts, norm=False):
        ret = ""

        # column labels
        ret += " %10s " % ""
        for valCol in marginalsCol.keys():
            ret += "  %10s  " % str(valCol)
        ret += "\n"

        for valRow, marginalRow in marginalsRow.items():        
            ret += " %10s " % str(valRow)
            for valCol in marginalsCol.keys():
                count = counts.get((valRow, valCol), 0)
                if (norm):
                    ret += "[       %.2f ]" % (float(count)/self.n)
                else:
                    ret += "[ %10d ]" % count
            if (norm):
                ret += "        %.2f  " % (float(marginalRow)/self.n)    
            else:
                ret += "  %10d  " % marginalRow                        
            ret += "\n"        

        # column totals
        ret += "\n"
        ret += " %10s " % ""
        for valCol, marginalCol in marginalsCol.items():
            if (norm):
                ret += "        %.2f  " % (float(marginalCol)/self.n)
            else:
                ret += "  %10d  " % marginalCol
        if (norm):
            ret += "        1.00 "
        else:
            ret += "  %10d  " % self.n
        ret += "\n"
        ret += "chisq = %.2f, pval = %.8f\n" % (self.chisq(), self.chisqPval())
        ret += "    g = %.2f, pval = %.8f\n" % (self.gstat(), self.gstatPval())
        ret += "\n"    
        return ret

    def toTsv(self, outfileName, norm=False):
        marginalsRow = self.marginalsRow
        marginalsCol = self.marginalsCol
        counts = self.counts

        ret = ""

        # column labels
        ret += "\t"
        for valCol in marginalsCol.keys():
            ret += "\t" + str(valCol)
        ret += "\n"

        for valRow, marginalRow in marginalsRow.items():        
            ret += "\t" + str(valRow)
            for valCol in marginalsCol.keys():
                count = counts.get((valRow, valCol), 0)
                if (norm):
                    ret += "\t" + str(float(count)/self.n)
                else:
                    ret += "\t" + str(count)
            if (norm):
                ret += "\t" + str(float(marginalRow)/self.n)    
            else:
                ret += "\t" + str(marginalRow)                        
            ret += "\n"        

        # column totals
        ret += "\n"
        ret += "\t"
        for valCol, marginalCol in marginalsCol.items():
            if (norm):
                ret += "\t" + str(float(marginalCol)/self.n)
            else:
                ret += "\t" + str(marginalCol)
        if (norm):
            ret += "\t1.00"
        else:
            ret += "\t" + str(self.n)
        ret += "\n"
        ret += "chisq = %.2f, pval = %.8f\n" % (self.chisq(), self.chisqPval())
        ret += "    g = %.2f, pval = %.8f\n" % (self.gstat(), self.gstatPval())
        ret += "\n"    
        
        outfile = open(outfileName, 'w')
        outfile.write(ret)
        outfile.close()


    def shuffle(self):
        rowVals = []
        colVals = []
        for (rowVal, colVal), count in self.counts.items():
            rowVals.extend([rowVal]*count)
            colVals.extend([colVal]*count)
        random.shuffle(rowVals)
        self.counts = {} # (row, col) -> count
        self.marginalsRow = {} # val -> count
        self.marginalsCol = {} # val -> count
        for i in range(len(rowVals)):
            valRow = rowVals[i]
            valCol = colVals[i]
            valTup = (valRow, valCol)
            self.counts[valTup] = self.counts.get(valTup, 0) + 1
            self.n += 1
        for (valRow, valCol), count in self.counts.items():
            self.marginalsRow[valRow] = self.marginalsRow.get(valRow, 0) + count
            self.marginalsCol[valCol] = self.marginalsCol.get(valCol, 0) + count

    def getCounts(self):    
        return tuple([ c[1] for c in sorted(self.counts.items()) ])
        




class ContingencyOfInts(Contingency):
    def __init__(self, valsRow, valsCol, domainRow, domainCol):
        self.n = len(valsRow)
        # self.counts = [ [0] * domainCol ] * domainRow
        self.counts = [ [ 0 for r in range(domainRow) ] for c in range(domainCol) ]
        self.marginalsRow = [0] * domainRow
        self.marginalsCol = [0] * domainCol 
        # sys.stderr.write("\n%s\n" % str(self.counts))
        self.domainRow = domainRow
        self.domainCol = domainCol

        for valRow, valCol in zip(valsRow, valsCol):
            self.counts[valRow][valCol] += 1
            self.marginalsRow[valRow] += 1
            self.marginalsCol[valCol] += 1
            # sys.stderr.write("\n%s\n" % str(self.counts))

        
    def chisq(self):
        n = float(self.n)        
        ret = 0.0
        nonzeroRows = [ (z, self.marginalsRow[z]) for z in range(self.domainRow) if (self.marginalsRow[z] > 0) ]
        nonzeroCols = [ (z, self.marginalsCol[z]) for z in range(self.domainCol) if (self.marginalsCol[z] > 0) ]
        for valRow, marginalRow in nonzeroRows:
            for valCol, marginalCol in nonzeroCols:
                observed = self.counts[valRow][valCol]
                expected = marginalRow*marginalCol / n
                ret += (observed - expected)**2 / expected                            
        return ret

    def gstat(self):
        n = float(self.n)
        ret = 0.0
        for valRow, marginalRow in enumerate(self.marginalsRow):        
            for valCol, marginalCol in enumerate(self.marginalsCol):
                observed = self.counts[valRow][valCol]
                if (observed > 0):
                    expected = marginalRow*marginalCol / n
                    ret += 2 * observed * math.log(observed/expected)
        return ret
    
    def dump(self, norm=False):
        mRow = dict(enumerate(self.marginalsRow))
        mCol = dict(enumerate(self.marginalsCol))
        cts = {}
        for r in range(self.domainRow):
            for c in range(self.domainCol):
                tup = (r, c)
                cts[tup] = self.counts[r][c]
        return self.dumpFromDicts(mRow, mCol, cts, norm)

    def getCounts(self):
        return tuple(reduce(lambda x, y: x+y, self.counts))



class ContingencyOfIntsTups(ContingencyOfInts):
    def __init__(self, tups, domainRow, domainCol):
        self.n = len(tups)
        self.counts = [ [ 0 for r in range(domainRow) ] for c in range(domainCol) ]
        self.marginalsRow = [0] * domainRow
        self.marginalsCol = [0] * domainCol 
        self.domainRow = domainRow
        self.domainCol = domainCol

        for valRow, valCol in tups:
            self.counts[valRow][valCol] += 1
            self.marginalsRow[valRow] += 1
            self.marginalsCol[valCol] += 1
            # sys.stderr.write("\n%s\n" % str(self.counts))



class WeightedContingencyOfInts(ContingencyOfInts):
    def __init__(self, tupRowColWeight, domainRow, domainCol):
        self.n = 0.0
        self.counts = [ [ 0.0 for r in range(domainRow) ] for c in range(domainCol) ]
        self.marginalsRow = [0.0] * domainRow
        self.marginalsCol = [0.0] * domainCol 
        self.domainRow = domainRow
        self.domainCol = domainCol

        for valRow, valCol, valWeight in tupRowColWeight:
            self.counts[valRow][valCol] += valWeight
            self.marginalsRow[valRow] += valWeight
            self.marginalsCol[valCol] += valWeight
            self.n += valWeight
            # sys.stderr.write("\n%s\n" % str(self.counts))




    
# takes an NST with two columns (row, col) of values
class ContingencyFromNST(Contingency):
    def __init__(self, valsNST):
        Contingency.__init__(self)
        rs = valsNST.selectRows()
        while (rs.next()):
            valRow = rs.getString(1)
            valCol = rs.getString(2)
            self.add(valRow, valCol)
        
# takes two attribute names, one for row, other for column
class ContingencyFromAttrs(ContingencyFromNST):
    def __init__(self, attrName1, attrName2):        
        attrData = prox.objectAttrs.getAttrDataNST(attrName1).join(prox.objectAttrs.getAttrDataNST(attrName2), "id = id", "A.val,B.val")
        ContingencyFromNST.__init__(self, attrData)

# takes a list of tuples of the form (rowval, colval)
class ContingencyFromTups(Contingency):
    def __init__(self, tups):
        Contingency.__init__(self)
        for valTup in tups:
            valRow, valCol = valTup
            self.add(valRow, valCol)
        
        
# takes a list of tuples of the form (rowval, colval)
class WeightedContingencyFromTups(Contingency):
    def __init__(self, tups):
        Contingency.__init__(self)
        for valRow, valCol, weight in tups:
            self.n += weight
            valTup = (valRow, valCol)
            self.marginalsRow[valRow] = self.marginalsRow.get(valRow, 0) + weight
            self.marginalsCol[valCol] = self.marginalsCol.get(valCol, 0) + weight
            self.counts[valTup] = self.counts.get(valTup, 0) + weight
        
        
        
        
# takes two lists of same length: rowvals, colvals
class ContingencyFromLists(Contingency):
    def __init__(self, rowVals, colVals):
        Contingency.__init__(self)
        for i in range(max(len(rowVals), len(colVals))):
            valRow = rowVals[i]
            valCol = colVals[i]
            valTup = (valRow, valCol)
            self.counts[valTup] = self.counts.get(valTup, 0) + 1
            self.n += 1
        for (valRow, valCol), count in self.counts.items():
            self.marginalsRow[valRow] = self.marginalsRow.get(valRow, 0) + count
            self.marginalsCol[valCol] = self.marginalsCol.get(valCol, 0) + count
            
# not really necessary anymore since the regular function is overloaded
def chisqFromLists(valsRow, valsCol, pVal=False):
    return chisq(valsRow, valsCol, pVal)

def chisq(vals, valsY=None, pVal=False):
    if (valsY is None):
        cont = ContingencyFromTups(vals)
    else:
        cont = ContingencyFromLists(vals, valsY)
    if (pVal):
        return cont.chisqValPval()
    else:
        return cont.chisq()




def gstatFromLists(valsRow, valsCol):
    c = ContingencyFromLists(valsRow, valsCol)
    return c.gstat()

def chisqFromIntLists(valsRow, valsCol, domainRow, domainCol):
    if (len(valsRow) != len(valsCol)):
        raise Exception("lists of unequal length in chisqFromIntListsSci(): %d row vals, %d col vals" % (len(valsRow), len(valsCol)))
    c = ContingencyOfInts(valsRow, valsCol, domainRow, domainCol)
    # sys.stderr.write(c.dump(True))
    # sys.stderr.write("chisqFromIntLists: %.4f\n%s" % (c.chisq(), c.dump()))
    return c.chisq() 

def gstatFromIntLists(valsRow, valsCol, domainRow, domainCol):
    c = ContingencyOfInts(valsRow, valsCol, domainRow, domainCol)
    return c.gstat()

def chisqFromIntListsSci(valsRow, valsCol, domainRow=2, domainCol=2): # assumes binary!
    if (len(valsRow) != len(valsCol)):
        raise Exception("lists of unequal length in chisqFromIntListsSci(): %d row vals, %d col vals" % (len(valsRow), len(valsCol)))

    n = float(len(valsRow))
    c = ContingencyOfInts(valsRow, valsCol, domainRow, domainCol)
    observed = numpy.array(c.counts)
    marginalsRow = numpy.sum(observed, 1)
    marginalsCol = numpy.sum(observed, 0)
    marginalsRowMat = numpy.resize(marginalsRow, (domainCol, domainRow)).T # transposed!
    marginalsColMat = numpy.resize(marginalsCol, (domainRow, domainCol))
    expected = marginalsRowMat*marginalsColMat/n
    diffs = observed - expected
    squared = diffs * diffs
    norm = squared / expected
    chisq = numpy.sum(norm)
    return chisq
            
###########################################

def equalDoubles(value, otherValue):
    return float(abs(value - otherValue)) < 0.000001

def gammaLn(x):
    cof = [ 76.18009172947146, -86.50532032941677, 24.01409824083091, -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 ]
    xx = x
    y = x
    tmp = x + 5.5
    ser = 1.000000000190015
    tmp -= (xx + 0.5) * math.log(tmp)
    for j in range(6):
        y += 1
        ser += cof[j] / y
    return -tmp + math.log(2.5066282746310005 * ser / xx)

def gammaIncomplete(a, x):
    result = 0.0
    if (x < 0.0) or (a <= 0.0):
        sys.stderr.write("\nerror: invalid args for gammaIncomplete(%.4f, %.4f)\n" % (a, x))
        return 1/0 #zzz dirty
    
    if (equalDoubles(x, 0)):
        return 0.0
    
    gln = gammaLn(a)
    # gln = 1.0;
    itmax = 1000;
    eps = 0.0000003
    if (x < (a + 1.0)): # Use the series representation.
        ap = a
        summ = 1.0 / a
        dell = summ
        for n in range(1, itmax+1):
            ap += 1.0
            dell *= x / ap
            summ += dell
            if (abs(dell) < (eps*abs(summ))):
                result = summ * math.exp(-x + a * math.log(x) - gln)
                return result
        sys.stderr.write("\nerror: series didn't converge: either a is too large, or ITMAX is too small.\n")
        return 1/0 #zzz dirty
        
    else: # Use the continued fraction representation
        fpMin = 0.000000000000000000000000000001
        b = x + 1.0 - a
        c = 1.0 / fpMin
        d = 1.0 / b
        h = d
        an = None
        dell = None
        
        for i in range(itmax):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if (abs(d) < fpMin):
                d = fpMin
            c = b + an / c
            if (abs(c) < fpMin):
                c = fpMin
            d = 1.0 / d
            dell = d * c
            h *= dell
            if (abs(dell - 1.0) < eps):
                break
        if (i > itmax):
            sys.stderr.write("\nerror: continued Fraction didn't converge: Either a is too large, or ITMAX is too small.\n")
            return 1/0 #zzz dirty
        
        result = math.exp(-x + a * math.log(x) - gln) * h
        return 1.0 - result

# def chiSquareP(x, dof):
#     # result = 1.0
#     # if (dof == 0):
#     #     return result;
#     # if not (equalDoubles(x, 0)):
#     #     if (x < 0):
#     #         x *= -1.0
#     #     result = 1.0 - gammaIncomplete((dof * 0.5), (x * 0.5))
#     # return result
#     return 1.0 - r.pchisq(x, dof)[0]





# thefted some stuff from http://python.genedrift.org/
def choose(n, k):
    if (0 <= k <= n):
        ntok = 1
        ktok = 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

# Pr(K = k) = (n choose k) p^k (1-p)^(n-k)
def binomialPmf(k, n, p):
    return choose(n, k)*(p**k)*((1.0-p)**(n-k))

# Pr(X <= x)
def binomialCdf(x, n, p):
    tot = 0.0
    for i in range(x):
        tot += binomialPmf(i, n, p)
    return tot

# http://faculty.vassar.edu/lowry/binomialX.html
def binomialCdfApprox(k, n, p):
    M = n*p
    dev = math.pow(n*p*(1.0-p), 0.5)
    if (k > M):
        z = abs(((k - M) - 0.5)/dev)
    elif (k < M):
        z = abs(((k - M) + 0.5)/dev)
    else:
        z = 1.0

    p = (((((0.000005383*z+0.0000488906)*z+0.0000380036)*z+0.0032776263)*z+0.0211410061)*z+0.049867347)*z+1;
    probTwoTail = math.pow(p, -16)    
    probOneTail = probTwoTail/2

    # sys.stderr.write("n: %d, k: %d, p:, %.4f, mean: %.4f, dev: %.4f, z: %.4f, prob: %.4f\n" % (n, k, p, M, dev, z, probOneTail))    
    return probOneTail

def mode(things):
    tots = {}
    for t in things:
        tots[t] = tots.get(t, 0) + 1
    return max(tots.items(), key=lambda x: x[1])[0]

def avg(scores):
    sum_scores = 0.0
    for score in scores:
        sum_scores += score
    return 1.0*sum_scores / len(scores)

def avgTup(tups):
    size = len(tups[0])
    sums = [0.0]*size
    for tup in tups:
        for i in range(size):
            sums[i] += tup[i]
    return tuple([ x/len(tups) for x in sums ])

def median(lstguy):
    vals = sorted(lstguy)
    length = len(vals)
    if (length % 2 == 0):
        return (vals[length/2] + vals[length/2 - 1]) / 2.0
    else:
        return vals[length/2] # rounds down     

def var(scores):
    # mean = avg(scores)
    # s = 0.0
    # for x in scores:
    #     d = 1.0*(x - mean)
    #     s += d*d
    #     if len(scores) > 1:
    #         var = 1.0*s/(len(scores) - 1)
    #     else:
    #         var = 1.0*s/len(scores)
    # return var
    mean = sum(scores)/float(len(scores))
    s = sum([ (x - mean)**2 for x in scores ])
    if len(scores) > 1:
        return 1.0*s/(len(scores) - 1)
    else:
        return 1.0*s/len(scores)
    return var

def stdev(scores):
    return math.pow(var(scores), 0.5)

def regress(colX, colY):
    # sys.stderr.write("regress(): %s, %s\n" % (colX, colY))    
    # drop any pair that has a None in it
    x = []
    y = []
    skipped = 0
    for i in range(len(colX)):
        if (colX[i] != None) and (colY[i] != None):
            x.append(float(colX[i]))
            y.append(float(colY[i]))
        else:
            skipped += 1
    if (skipped > 0):
        sys.stderr.write("rsquare skipped %d pairs with None values\n" % skipped)
    ret = stats.linregress(x, y) # gradient, intercept, r_value, p_value, std_err 
    return ret
    
    
    
# from Sachs p.407
def pearson(vals, valsY=None):    
    if (valsY is None):
        tups = vals
        xVals = [ t[0] for t in tups ]
        yVals = [ t[1] for t in tups ]        
    else:        
        xVals = vals
        yVals = valsY
        tups = zip(xVals, yVals)
        
    xMean = avg(xVals)
    yMean = avg(yVals)
    xStdev = math.sqrt(var(xVals))
    yStdev = math.sqrt(var(yVals))

    
    # if (xStdev == 0.0) or (yStdev == 0.0):
    #      sys.stderr.write("got stdev=0 in pearson:\n\t%s\n\t%s\n" % (xVals, yVals))    
    #     return 0.0 #zzz
    # total = 0.0
    # for x, y in tups:
    #     xStd = (x - xMean)/xStdev
    #     yStd = (y - yMean)/yStdev
    #     total += xStd*yStd
    
    # sys.stderr.write("pearson xStdev = %f, yStdev = %f, %s\n" % (xStdev, yStdev, "***" if (xStdev and yStdev) else ""))
    
    xStdev_yStdev = xStdev*yStdev
    total = sum([ (t[0] - xMean)*(t[1] - yMean)/xStdev_yStdev for t in tups ])

    return total/(len(xVals) - 1)
# old implementation:
# def pearson(xVals, yVals):
#     total = 0.0
#     xMean = avg(xVals)
#     yMean = avg(yVals)
#     xStdev = math.pow(var(xVals), 0.5)
#     yStdev = math.pow(var(yVals), 0.5)
# 
#     # if (xStdev == 0.0) or (yStdev == 0.0):
#     #     return 0.0 #zzz
# 
#     for x, y in zip(xVals, yVals):
#         xStd = (x - xMean)/xStdev
#         yStd = (y - yMean)/yStdev
#         total += xStd*yStd
#     return total/(len(xVals) - 1)
        
def rsquare(vals, valsY=None):
    return pearson(vals, valsY)**2

# from Sachs p.456
def pearsonPartial(vals, yVals=None, condVals=None, pVal=False):
    if (yVals is None):
        xVals = [ v[0] for v in vals ]
        yVals = [ v[1] for v in vals ]
        condVals = [ v[2] for v in vals ]
    else:
        xVals = vals
        if isinstance(condVals[0], list) or isinstance(condVals[0], tuple):
            if len(condVals[0]) == 1:
                condVals = [ t[0] for t in condVals ]
            else:
                raise TypeError("partial correlations only works for single cond value")

    rxy = pearson(xVals, yVals)
    rxz = pearson(xVals, condVals)
    ryz = pearson(yVals, condVals)
    num = rxy - (rxz*ryz)
    den = math.pow((1.0-rxz**2)*(1.0-ryz**2), 0.5)
    stat = num/den

    # if (pVal):
    #     statP = zscoreP(abs(stat))
    #     return (stat, statP)
    # else:
    #     return stat

    zscore, pval = zscoreFromR(stat, len(vals))
    if (pVal):
        return (zscore, pval)
    else:
        return zscore


def partial_corr(xvals, yvals, condvals, pval=False):
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    condvals = np.array(condvals)

    num_conds = condvals.shape[1]
    if num_conds == 0:
        r, _ = stats.pearsonr(xvals, yvals)
    # elif num_conds == 1:
    #     rxy, _ = stats.pearsonr(xvals, yvals)
    #     rxz, _ = stats.pearsonr(xvals, condvals)
    #     rzy, _ = stats.pearsonr(condvals, yvals)
    #     num = rxy - (rxz*rzy)
    #     den = math.pow((1.0-rxz**2)*(1.0-rzy**2), 0.5)
    #     r = num/den
    else:
        zvals = condvals[:, 0]
        condvals_rest = condvals[:, 1:]

        rxy_rest = partial_corr(xvals, yvals, condvals_rest, pval=False)
        rxz_rest = partial_corr(xvals, zvals, condvals_rest, pval=False)
        rzy_rest = partial_corr(zvals, yvals, condvals_rest, pval=False)

        num = rxy_rest - (rxz_rest*rzy_rest)
        den = math.pow((1.0-rxz_rest**2)*(1.0-rzy_rest**2), 0.5)
        r = num / den

    # if we want the pval, we're going to return the zscore instead of r
    # see: https://tinyurl.com/ycdbexzj
    if pval:
        return zscoreFromR(r, len(xvals))
    else:
        return r


def isConstant(lstguy, fieldIndex=None):
    if (fieldIndex is None):
        for elt in lstguy[1:]:
            if (elt != lstguy[0]):
                return False
        return True
    else:
        first = lstguy[0][fieldIndex]
        # sys.stderr.write("first is: %s\n" % (first))        
        for tup in lstguy[1:]:
            if (tup[fieldIndex] != first):
                # sys.stderr.write("non-constant block: %s != %s\n" % (first, tup[fieldIndex]))
                return False
        #     else:
        #         sys.stderr.write("so far constant: %s == %s\n" % (first, tup[fieldIndex]))                
        # sys.stderr.write("constant block: %s\n" % (str([ x[fieldIndex] for x in lstguy ])))
        return True
        
    
def pearsonBlock(vals, yVals=None, blockVals=None, pVal=False, effect=False):
    if (yVals is None):
        tups = vals
        blockVals = [ t[2] for t in tups ]
    else: 
        tups = zip(vals, yVals, blockVals)

    blocks = {} # blockVal -> tups
    for tup in tups:
        blocks.setdefault(tup[2], []).append((tup[0], tup[1]))
    
    # for tup in tups[:20]:    
    #     sys.stderr.write("Guo got %s\n" % (str(tup)))
    # sys.stderr.write("Guo got %d blocks\n" % (len(blocks)))


    tupsCorrSizeVar = []
    smallBlocks = 0
    constBlocks = 0
    for blockVal, blockTups in blocks.items():
        # sys.stderr.write("\tblock %s:\n" % str(blockVal))
        # for blockTup in blockTups:
        #     sys.stderr.write("\t\t%s\n" % (str(blockTup)))
            
        blockSize = len(blockTups)
        if (blockSize > 2):#zzz    
            # sys.stderr.write("checking constant\n")
            # if (isConstant(blockTups, 0) or isConstant(blockTups, 1)):
            #     sys.stderr.write("constant block\n")
            try:
                tupsCorrSizeVar.append( (pearson(blockTups), blockSize, 1.0/blockSize) )
            except ZeroDivisionError:
                constBlocks += 1
                continue
            # else:
            #     sys.stderr.write("non-constant block\n")                
            #     tupsCorrSizeVar.append( (pearson(blockTups), blockSize, 1.0/blockSize) )
        else:
            #zzz is this correct?
            # tupsCorrSizeVar.append( (0.0, blockSize, 1.0/blockSize) )
            smallBlocks += 1
    # sys.stderr.write("got %d good blocks, %d constant, %d small\n" % (len(tupsCorrSizeVar), constBlocks, smallBlocks))
        
    sumSizes = float(sum([ t[1] for t in tupsCorrSizeVar ]))
    # sys.stderr.write("block correlations: %s\n" % str([ t[0] for t in tupsCorrSizeVar ]))
    # sys.stderr.write("block sizes: %s\n" % str([ t[1] for t in tupsCorrSizeVar ]))
    # sys.stderr.write("block sizes: %s\n" % str(histogram([ t[1] for t in tupsCorrSizeVar ])))

    # assert sumSizes > 0, "constant blocks in Guo"

    rho_hat = sum([ t[0]*t[1] for t in tupsCorrSizeVar ]) / sumSizes
    # sys.stderr.write("rho hat: %.8f\n" % rho_hat)
    stat = rho_hat*1/math.sqrt(sum([ t[2]*math.pow(t[1], 2) for t in tupsCorrSizeVar ]) / math.pow(sumSizes, 2))
    # sys.stderr.write("statistic: %.8f\n" % stat)
    statP = zscoreP(abs(stat))
    
    rets = [stat]
    if (pVal):
        rets.append(statP)
    if (effect):
        rets.append(rho_hat**2)
    return tuple(rets)


def histogram(lstguy):
    counts = {}
    for guy in lstguy:
        counts[guy] = counts.get(guy, 0) + 1 
    return counts




# from Sachs p.427
def zscore(xVals, yVals, condVals=None):
    if (condVals is None):
        r = pearson(xVals, yVals)
    else:
        r = pearsonPartial(xVals, yVals, condVals)
    fisher = 0.5*math.log((1.0+r)/(1.0-r))
    z = math.pow(len(xVals) - 3, 0.5) * fisher
    
    return z

def zscoreFromR(r, n):
    fisher = 0.5*math.log((1.0+r)/(1.0-r))
    z = math.pow(n - 3, 0.5) * fisher
    p = zscoreP(z)
    return z, p
    
# #zzz should this be abs(z)?!
def zscoreP(z):
    # return 2.0*(1.0 - r.pnorm(abs(z))[0])
    return 2.0*(1.0 - norm.cdf(abs(z)))

def stdProduct(xVals, yVals):
    if (len(xVals) < 2): #zzz is there a better way to handle this?
        return 0.0
    total = 0.0
    xMean = avg(xVals)
    yMean = avg(yVals)
    xStdev = math.pow(var(xVals), 0.5)
    yStdev = math.pow(var(yVals), 0.5)
    if (xStdev == 0.0) or (yStdev == 0.0):
        return 0.0 #zzz
    # sys.stderr.write("xVals: " + str(xVals) + "\n")
    # sys.stderr.write("yVals: " + str(yVals) + "\n")
    for x, y in zip(xVals, yVals):
        xStd = (x - xMean)/xStdev
        yStd = (y - yMean)/yStdev
        total += math.pow(xStd*yStd, 2)
    return total    

def pVal(statistic, distribution): 
    # pval = 1.0 - 1.0*len([val for val in distribution if statistic > val])/len(distribution)
    less = 0
    equal = 0
    # greater = 0
    for val in distribution:
        if (val < statistic):
            less += 1
        # elif (val > statistic):
        #     greater += 1
        elif (val == statistic): 
            equal += 1
    pV = 1.0 - ((less + 0.5*equal) / len(distribution))
    # distSorFor = [ "%.4f" % z for z in sorted(distribution) ]
    # sys.stderr.write("p-val of %.8f in [ %s ... %s ... %s ... %s ... %s ] = %.4f\n" % (statistic, distSorFor[0], distSorFor[int(len(distSorFor)*0.25)], distSorFor[int(len(distSorFor)*0.5)], distSorFor[int(len(distSorFor)*0.75)], distSorFor[-1], pV))
    return pV

def quantile(q, distribution):
    distrib = sorted(distribution)
    rank = int(q*len(distribution))
    return distrib[rank]

def pointBiserialFromLists(binVals, contVals):
    # sys.stderr.write("binVals: %s\ncontVals: %s\n\n" % (binVals[:20], contVals[:20]))
    posVals = []
    negVals = []
    for binVal, contVal in zip(binVals, contVals):
        if (str(binVal) == "1"):
            posVals.append(contVal)
        else:
            negVals.append(contVal)
    return pointBiserial(posVals, negVals)

def pointBiserial(posVals, negVals):
    allVals = posVals + negVals
    # sys.stderr.write("%d pos (avg %.4f), %d neg (avg %.4f), %d all\n" % (len(posVals), avg(posVals), len(negVals), avg(negVals), len(allVals)))
    t1 = (avg(posVals) - avg(negVals))/stdev(allVals)
    t2 = float(len(posVals))/len(allVals)
    t3 = float(len(negVals))/len(allVals)
    t4 = t2*t3
    t5 = math.pow(t4, 0.5)
    # sys.stderr.write("t1: %.8f, t2: %.8f, t3: %.8f, t4: %.8f, t5: %.8f\n" % (t1, t2, t3, t4, t5))
    return t1*t5
    

# Taken from http://udel.edu/~mcdonald/statcmh.html
# assumes that xVals and yVals are from {0, 1}, stratVals can be anything
def cmhFromLists(xVals, yVals, stratVals, pVal=False):
    contingFromStrat = {}
    for x, y, strat in zip(xVals, yVals, stratVals):
        if strat not in contingFromStrat:
            contingFromStrat[strat] = Contingency()
        contingFromStrat[strat].add(x, y)
    cmhNumSum = 0.0
    cmhDenSum = 0.0
    sys.stderr.write("conditioned %d tuples in to %d strats\n" % (len(stratVals), len(contingFromStrat)))
    for strat, conting in contingFromStrat.items():
        # sys.stderr.write("strat %s:\n%s\n" % (strat, conting.dump()))
        a = conting.counts.get((0, 0), 0) #  a  b
        b = conting.counts.get((0, 1), 0) #  c  d
        c = conting.counts.get((1, 0), 0)
        d = conting.counts.get((1, 1), 0)
        n = float(a + b + c + d) # ensure that we'll be floating throughout
    
        num = a - (a + b)*(a + c) / n
        den = (a + b)*(a + c)*(b + d)*(c + d) / (n**3 - n**2)
        cmhNumSum += num
        cmhDenSum += den
        # if (num == 0) or (den == 0) or True:
        #     sys.stderr.write("cmh error:\n\t%4d\t%4d\n\t%4d\t%4d\n" % (a, b, c, d))
                
    cmhNum = (abs(cmhNumSum) - 0.5)**2
    cmh = cmhNum / cmhDenSum

    if (pVal):
        return cmh, chiSquareP(cmh, 1)    
    else:
        return cmh

# def pearsonFromListsR(xVals, yVals, pVal=False):
#
#     # sys.stderr.write("cor gave " + str(r['cor'](robjects.FloatVector(xVals), robjects.FloatVector(yVals))) + "\n")
#
#     rets = r['cor.test'](robjects.FloatVector(xVals), robjects.FloatVector(yVals), "two.sided", "pearson")
#
#     # for i, thing in enumerate(rets):
#     #     sys.stderr.write("thing %d: %s\n" % (i, str(thing)))
#
#     t = rets[0][0]
#     dof = rets[1][0]
#     p = rets[2][0]
#     eff = rets[3][0]
#     rets = [eff]
#     if (pVal):
#         rets.append(p)
#     return tuple(rets)


# def ttestFromListsR(numVals, binVals, pVal=False):
#     binVal1, binVal2 = list(set(binVals))
#     numVals1 = []
#     numVals2 = []
#     for numVal, binVal in zip(numVals, binVals):
#         if (binVal == binVal1):
#             numVals1.append(numVal)
#         else:
#             numVals2.append(numVal)
#
#     rets = r['t.test'](robjects.FloatVector(numVals1), robjects.FloatVector(numVals2))
#     stat = rets[0][0]
#     dof = rets[1][0]
#     p = rets[2][0]
#     if (pVal):
#         return stat, p
#     else:
#         return stat
    
    
def cmhR(vals, yVals=None, stratVals=None, pVal=False, effect=False):
    if (yVals is None):
        xVals = [ t[0] for t in vals ]
        yVals = [ t[1] for t in vals ]
        stratVals = [ t[1] for t in vals ]
    else:
        xVals = vals
    return cmhFromListsR(xVals, yVals, stratVals, pVal=pVal, effect=effect)    

# def cmhFromListsR(xVals, yVals, stratVals, pVal=False, effect=False):
#     # mantelhaen.test(x, y = NULL, z = NULL,
#     #                 alternative = c("two.sided", "less", "greater"),
#     #                 correct = TRUE, exact = FALSE, conf.level = 0.95)
#     #
#     # A list with class "htest" containing the following components:
#     # statistic     Only present if no exact test is performed. In the classical case of a 2 by 2 by K table (i.e., of dichotomous underlying variables), the Mantel-Haenszel chi-squared statistic; otherwise, the generalized Cochran-Mantel-Haenszel statistic.
#     # parameter     the degrees of freedom of the approximate chi-squared distribution of the test statistic (1 in the classical case). Only present if no exact test is performed.
#     # p.value     the p-value of the test.
#     # conf.int     a confidence interval for the common odds ratio. Only present in the 2 by 2 by K case.
#     # estimate     an estimate of the common odds ratio. If an exact test is performed, the conditional Maximum Likelihood Estimate is given; otherwise, the Mantel-Haenszel estimate. Only present in the 2 by 2 by K case.
#     # null.value     the common odds ratio under the null of independence, 1. Only present in the 2 by 2 by K case.
#     # alternative     a character string describing the alternative hypothesis. Only present in the 2 by 2 by K case.
#     # method     a character string indicating the method employed, and whether or not continuity correction was used.
#     # data.name     a character string giving the names of the data.
#
#     # sample size in each stratum must be > 1
#     stratCounts = {}
#     for z in stratVals:
#         stratCounts[z] = stratCounts.get(z, 0) + 1
#     tups = [ (x, y, z) for x, y, z in zip(xVals, yVals, stratVals) if (stratCounts[z] > 1) ]
#     # for i, (x, y, z) in enumerate(tups):
#     #     sys.stderr.write("tup %d\tx: %s,\t y: %s,\t z: %s\n" % (i, x, y, z))
#
#     xVals = [ t[0] for t in tups ]
#     yVals = [ t[1] for t in tups ]
#     stratVals = [ t[2] for t in tups ]
#
#     rets = r['mantelhaen.test'](robjects.StrVector(xVals), robjects.StrVector(yVals), robjects.StrVector(stratVals))
#     # sys.stderr.write("rets: %s, %d\n" % (rets, len(rets)))
#     # sys.stderr.write("stat: %s\n" % (rets[0][0]))
#     # sys.stderr.write("dof: %s\n" % (rets[1][0]))
#     # sys.stderr.write("p-val: %s\n" % (rets[2][0]))
#     stat = rets[0][0]
#     dof = rets[1][0]
#     p = rets[2][0]
#
#     rets = [stat]
#     if (pVal):
#         rets.append(p)
#     if (effect):
#         c = (stat/(stat + len(xVals)))**0.5 # contingency coeff
#         k = min(len(set(xVals)), len(set(yVals))) # k is the min(num rows, num cols)
#         cmax = ((k - 1)/float(k))**0.5
#         acc = c / cmax
#         rets.append(acc)
#     return tuple(rets)

    
# def chisqFromListsR(xVals, yVals, pVal=False, effect=False):
#     # chisq.test(x, y = NULL, correct = TRUE,
#     #            p = rep(1/length(x), length(x)), rescale.p = FALSE,
#     #            simulate.p.value = FALSE, B = 2000)
#     #
#     # statistic     the value the chi-squared test statistic.
#     # parameter     the degrees of freedom of the approximate chi-squared distribution of the test statistic, NA if the p-value is computed by Monte Carlo simulation.
#     # p.value     the p-value for the test.
#     # method     a character string indicating the type of test performed, and whether Monte Carlo simulation or continuity correction was used.
#     # data.name     a character string giving the name(s) of the data.
#     # observed     the observed counts.
#     # expected     the expected counts under the null hypothesis.
#     # residuals     the Pearson residuals, (observed - expected) / sqrt(expected).
#     rets = r['chisq.test'](robjects.StrVector(xVals), robjects.StrVector(yVals))
#     stat = rets[0][0]
#     dof = rets[1][0]
#     p = rets[2][0]
#
#     rets = [stat]
#     if (pVal):
#         rets.append(p)
#     if (effect):
#         c = (stat/(stat + len(xVals)))**0.5 # contingency coeff
#         k = min(len(set(xVals)), len(set(yVals))) # k is the min(num rows, num cols)
#         cmax = ((k - 1)/float(k))**0.5
#         acc = c / cmax
#         rets.append(acc)
#     return tuple(rets)
    

def chisq3d(tups, pVal=False):
    xVals = [ t[0] for t in tups ]
    yVals = [ t[1] for t in tups ]
    stratVals = [ t[2] for t in tups ]
    return chisq3dFromLists(xVals, yVals, stratVals, pVal)
        
def chisq3dFromLists(xVals, yVals, stratVals, pVal=False):
    xValsByStrat = {}
    yValsByStrat = {}

    for x, y, strat in zip(xVals, yVals, stratVals):
        xValsByStrat.setdefault(strat, []).append(x)
        yValsByStrat.setdefault(strat, []).append(y)
    
    dofs = []
    chisqs = []

    levelsRow = []
    levelsCol = []
    
    sys.stderr.write("chisq3d partitioned %d tuples into %d strats\n" % (len(xVals), len(xValsByStrat)))
    for strat, xList in xValsByStrat.items():        
        yList = yValsByStrat[strat]

        cont = ContingencyFromLists(xList, yList)
        dof = cont.dof()
        dofs.append(cont.dof())
        chisqs.append(cont.chisq())
        
        levR, levC = cont.levels()
        levelsRow.append(levR)
        levelsCol.append(levC)
        sys.stderr.write("\t%s\n%s\n" % (str(strat), cont.dump()))
    
    sumDofs = sum(dofs)
    sumChisq = sum(chisqs)
    # assert sumDofs > 0, "constant strats in chisq 3d"

    # all strats had a 1x1 contingency table, conclude that items are conditionally independent
    sys.stderr.write("max levels row: %f, max levels col %f\n" % (max(levelsRow), max(levelsCol)))
    if (max(levelsRow) == 1) or (max(levelsCol) == 1): 
        pv = 1.0
        sys.stderr.write("levels %s, %s chisq = %.2f, pval = %.8f\n" % (levelsRow, levelsCol, sumChisq, pv))            
        return sumChisq, pv if pVal else sumChisq
    # 
    # if (sumDofs == 0):
    #     else:
    #         pv = 1.0
    # 
    else:
        if (pVal):
            pv = chiSquareP(sum(chisqs), sumDofs)
            sys.stderr.write("\tchisq = %s = %s, dof = %s = %s, pval = %.8f\n" % (" + ".join([ "%.2f" % c for c in chisqs]), sum(chisqs), " + ".join([ "%.2f" % c for c in dofs]), sum(dofs), pv)) 
            return sum(chisqs), pv
        else:
            return sum(chisqs)



def kendallTauFromTups(tups):
    return kendallTauFromLists([ t[0] for t in tups ], [ t[1] for t in tups ])

def kendallTauFromLists(ranks1, ranks2):
    con, dis = condiscordantPairs(ranks1, ranks2)
    tau = (con - dis) / (0.5*len(ranks1)*(len(ranks1) - 1))
    sys.stderr.write("concord %d, discord %d, tau %.4f\n" % (con, dis, tau))
    return tau

# returns a tuple (numConcordant, numdiscordant)
def condiscordantPairs(xVals, yVals):    
    con = 0
    dis = 0
    for i in range(len(xVals)):
        x1 = xVals[i]
        y1 = yVals[i]
        for j in range(i, len(xVals)):
            x2 = xVals[j]
            y2 = yVals[j]    
            if (cmp(x2 - x1, 0) == cmp(y2 - y1, 0)):
                con += 1
            else:
                dis += 1
    return (con, dis)


class TrialStats:
    def __init__(self, trials):
        self.trials = trials
        self.statVals = {} # name -> [ trial1, trial2, ... ]
    def addStat(self, trial, name, val):
        vals = self.statVals.setdefault(name, [None]*self.trials)
        vals[trial] = val
    def avg(self, name=None):
        if (name is not None):
            return sum(self.statVals[name]) / float(self.trials)
        else:
            rets = {}
            for name, vals in self.statVals.items():
                rets[name] = sum(vals) / float(self.trials)
            return rets
        
class TrialStatsIter:
    def __init__(self, num, reportInterval=None):
        self.num = num
        if (reportInterval is None):
            self.reportInterval = num
        elif (reportInterval < 1.0):
            self.reportInterval = reportInterval*num
        else:
            self.reportInterval = reportInterval
        self.trials = []
        self.prog = Utils.Progress(num)
    def __iter__(self):
        return self
    def next(self):
        if (len(self.trials) >= self.num):
            raise StopIteration
        else:
            trial = {}    
            self.trials.append(trial)
            if (self.reportInterval) and (len(self.trials) % self.reportInterval == 0):
                sys.stderr.write("\ttrial %d" % len(self.trials))
                sys.stderr.write("\t%s\n" % self.prog.report(len(self.trials)))
            return trial
            
    def statNames(self):
        names = set()
        for trial in self.trials:
            names.update(trial.keys())
        return names

    def avg(self, name=None):
        if (name is not None):
            return sum([ t[name] for t in self.trials ]) / float(len(self.trials))
        else:
            rets = {}            
            for stat in self.statNames():
                rets[stat] = self.avg(stat)
            return rets
    
    def propLessThan(self, name, crit):
        vals = [ t[name] for t in self.trials ]
        val1 = len(filter(lambda x: (x < crit), vals))/float(len(vals))
        
        # 
        # countLessThan = 0
        # for i, trial in enumerate(self.trials):
        #     if (trial[name] < crit):
        #         # sys.stderr.write("trial %d %s value %f < crit %f\n" % (i, name, trial[name], crit))
        #         countLessThan += 1
        # val2 = float(countLessThan)/len(self.trials) 
        # 
        # if (val1 != val2):
        #     sys.stderr.write("disagreement in propLessThan: %.4f vs %.4f\n" % (val1, val2))
        # 
        # return val2
        return val1

    def vals(self, name, castFunc=None):
        if (castFunc is None):
            return [ t[name] for t in self.trials ]
        else:
            return [ castFunc(t[name]) for t in self.trials ]
    
            
    def tsv(self, aggregFunc, cols, labels=True, constants=[]):
        if (labels):
            line = "\t".join([ c[0] for c in constants ] + cols) + "\n"
        else:
            line = ""        
        for const in constants:
            line += str(const[1]) + "\t"
        for col in cols:
            vals = [ t[col] for t in self.trials ]
            val = aggregFunc(vals)
            line += str(val) + "\t"
        return line[:-1] + "\n"

    def tsvAvg(self, cols, labels=True, constants=[]):
        return self.tsv(lambda x: sum(x)/float(len(x)), cols, labels, constants)

    def tsvPropLessThan(self, crit, cols, labels=True, constants=[]):
        return self.tsv(lambda lst: len(filter(lambda x: (x < crit), lst))/float(len(lst)), cols, labels, constants)

