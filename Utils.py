#!/usr/bin/python
import sys
import time
import os.path
import glob
import math



romans = ["", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix"]
for ten in ["x", "xx", "xxx", "xl", "l", "lx", "lxx", "lxxx", "xc", "c"]:
	romans.extend([ ten + s for s in romans[:10] ])

def romanize(intguy):
	def romanSegment(number, base):
		if (base == 1):
			numeralOne = "i"
			numeralFive = "v"
			numeralTen = "x"
		elif (base == 10):	
			numeralOne = "x"
			numeralFive = "l"
			numeralTen = "c"
		elif (base == 100):
			numeralOne = "c"
			numeralFive = "d"
			numeralTen = "m"					
		if (number < 4):
			return numeralOne*number
		elif (number == 4):
			return numeralOne + numeralFive
		elif (number == 9):
			return numeralOne + numeralTen
		else: # (number > 4) and (number < 9)
			return numeralFive + numeralOne*(number - 5)
	strguy = str(intguy)
	roman = ""
	if (intguy >= 1000):
		roman += "m"*int(strguy[:-3])
	for i in [3, 2, 1]:
		if (len(strguy) >= i):
			roman += romanSegment(int(strguy[-i]), 10**(i-1))
	return roman	
	




def timestamp(length=12, t=None, spaces=False):
	if not (t):
		t = time.localtime()
	if (length == 12):
		return time.strftime("%Y%m%d %H%M", t) if (spaces) else time.strftime("%Y%m%d%H%M", t)
	elif (length == 14):
		return time.strftime("%Y%m%d %H%M%S", t) if (spaces) else time.strftime("%Y%m%d%H%M%S", t)
	elif (length == 8):
		return time.strftime("%Y%m%d", t)

def formatSecs(secs):
	secs = int(secs)
	if (secs < 3600):
		return time.strftime("%M:%S", time.gmtime(secs))
	elif (secs < 86400):
		return time.strftime("%H:%M:%S", time.gmtime(secs))
	else:
		return str(secs/86400) + ":" + time.strftime("%H:%M:%S", time.gmtime(secs%86400))

class Timer:
	def __init__(self):
		self.start = time.time()
	def reset(self):
		self.start = time.time()
	def elapsed(self):
		return time.time() - self.start
	def elapsedPr(self):
		return formatSecs(time.time() - self.start)
	def stderr(self, txt=""):
		sys.stderr.write(self.elapsedPr() + " " + txt + "\n")


		
class Progress:
	def __init__(self, tot):
		self.total = tot
		self.timer = Timer()
	def perc(self, done):
		return "%.1f%%" % (100.0*done/self.total)
	def elapsed(self):
		return self.timer.elapsed()
	def togo(self, done):
		if (done):
			el = self.timer.elapsed()
			frac = float(done) / self.total
			whole = el / frac
			return whole - el
		else: 
			return -0
	def rate(self, done):
		return float(done) / self.elapsed()
	def report(self, done):
		spc = "   "
		ret = formatSecs(self.elapsed()) + spc
		ret += ("%d/%d" % (done, self.total)).rjust(1 + len(str(self.total))*2) + spc
		ret += "%6s" % (self.perc(done)) + spc 
		r = self.rate(done)
		if (r > 1.0):
			ret += "%3.0f/s" % (r) + spc
		elif (done > 0):
			ret += "%1.2fs   " % (float(self.elapsed())/done) + spc
		else:
			ret += "  n/a" + spc
		ret += "-" + formatSecs(self.togo(done)) + spc
		return ret

class ProgressIter:
	def __init__(self, stuff, reportPre="", reportFreq=0.2):
		if (type(stuff) == int):
			self.elts = range(stuff)
		else:
			self.elts = stuff[:]
		self.idx = 0
		self.prog = Progress(len(self.elts))
		self.reportPre = reportPre
		if (reportFreq == 0):
			self.reportInterval = len(self.elts) + 1 
		elif (reportFreq > 0.0) and (reportFreq < 1.0):
			self.reportInterval = max(int(reportFreq*len(self.elts)), 1)
		else: # freq > 1
			self.reportInterval = reportFreq
	def __iter__(self):
		return self
	def next(self):
		if (self.idx >= len(self.elts)):
			sys.stderr.write("%s %6s   %s  done.\n" % (self.reportPre, abbreviate(str(self.elts[-1]), 100), self.prog.report(self.idx)))
			raise StopIteration
		else:
			elt = self.elts[self.idx]
			if (self.idx % self.reportInterval == 0):
				sys.stderr.write("%s %6s   %s\n" % (self.reportPre, abbreviate(str(elt), 100), self.prog.report(self.idx)))
			self.idx += 1
			return elt
	def report(self):
		return 

def abbreviate(s, length, back=10):
	elipse = "..."
	if (len(s) > length):
		front = length - len(elipse) - back
		sAb = s[:front] + elipse + s[(-1*back):]
		# sys.stderr.write("sAb: %s\n" % sAb)
		return sAb
	else:
		return s
	
	

def histogram(listguy, bucketsize=0):
	if (bucketsize == 0):
		bucketsize = int((max(listguy) - min(listguy)) / 100) + 1
		totals = [0]*100
	else:
		# totals = [0]*((max(listguy) - min(listguy)) / bucketsize + 1)
		totals = [0]*(max(listguy) / bucketsize + 1)
		
	#sys.stderr.write("bucketsize: %d, num buckets: %d" % (bucketsize, len(totals)))	
	for val in listguy:
		bucket = int(val / bucketsize)
		#sys.stderr.write("val: %d, bucket: %d\n" % (val, bucket))
		totals[bucket] += 1
	return totals
	
def histogramDisc(listguy, normalize=False):
	counts = {}
	for val in listguy:
		counts[val] = counts.get(val, 0) + 1
	if (normalize):
		norm = float(len(listguy))
		for val in counts.keys():
			counts[val] /= norm			
	return counts
		
# def asciibar(listguy, maxbar=100.0, trim=True):
# 	#maxbar = 100.0 # longest bar we want to make
# 	maxval = max(listguy)			
# 	norm = maxbar / max(maxval, maxbar)
# 	ret = ""
# 	for i in range(len(listguy)):
# 		if (listguy[i] > 0) or (trim == False):			
# 			bar = "#" * int(norm*listguy[i] + 0.5)
# 			ret += " %4s|%s (%d)\n" % (i, bar, listguy[i])  
# 			trim = False
# 	return ret
def asciibar(val_count, scale=1.0, trim=True):
	maxval = max(val_count.values())			
	# if (length is None):
	# 	norm = maxVal
	ret = ""
	for key, count in sorted(val_count.items()):
		if (count > 0) or (trim == False):			
			bar = "#" * int(count*scale + 0.5)
			ret += " %4s|%s (%d)\n" % (key, bar, count)  
	return ret

def asciibarStack(listguy, length=None):
	hist = histogramDisc(listguy, normalize=False)
	if (length is not None):
		norm = float(length)/max(len(listguy), 1)
	else:
		norm = 1.0
	bar = ""
	for val, count in sorted(hist.items()):
		bar += str(val)*int(count*norm + 0.5)
	return bar
		

class LRU:
	def __init__(self, sz):
		self.size = sz
		self.counter = 0
		self.queue = [] # holds (access, key) tuples, sorted least recent to most recent
		self.key_value = {}
		self.key_access = {}
	def _update(self, key):
		access = self.key_access[key]
		idx = bisect.bisect_left(self.queue, (access, key))
		del self.queue[idx]		
		self.key_access[key] = self.counter
		self.queue.append((self.counter, key))
		self.counter += 1
	def __contains__(self, key):
		return key in self.key_value
	def __getitem__(self, key):
		self._update(key)
		return self.key_value[key]
	def __setitem__(self, key, value):
		if (key in self.key_value):
			self.key_value[key] = value
			self._update(key)
		else:
			self.key_value[key] = value
			self.key_access[key] = self.counter
			self.queue.append((self.counter, key))
			self.counter += 1
			if (len(self.key_value) > self.size):
				accessLeast, keyLeast = self.queue.pop(0)
				del self.key_value[keyLeast]
				del self.key_access[keyLeast]
	def __str__():
		ret = "[ "
		for access, key in self.queue:
			ret += str(access) + "[" + str(key) + "," + str(key_value[key]) + "]"
		ret += " ]"
		return ret
		
def factorialList(*paramLists):	
	# sys.stderr.write("paramLists: %s\n" % (str(paramLists)))		
	combos = [ [] ]
	paramListsRev = [x for x in paramLists ]
	paramListsRev.reverse()
	for paramList in paramListsRev:
		# sys.stderr.write("extending combos with %s\n" % (str(paramList)))
		combosNew = []
		for paramVal in paramList:
			for combo in combos:
				comboNew = [paramVal] + combo  
				combosNew.append(comboNew)
		combos = combosNew
	# sys.stderr.write("got %d combos:\n" % (len(combos)))
	# for combo in combos:
	# 	sys.stderr.write("%s\n" % (str(combo)))
	return [ tuple(x) for x in combos ]

# paramListDict: param -> [ setting1, setting2, ... ]
# returns list of dicts param -> setting
def factorialDict(paramListDict):	
	params = sorted(paramListDict.keys())
	paramLists = [ paramListDict[x] for x in params ]
	comboTups = factorialList(*paramLists)
	comboDicts = []
	for comboTup in comboTups:
		comboDicts.append(dict(zip(params, comboTup)))
	return comboDicts
	
def powerSet(lstguy):
	if (len(lstguy) == 0):
		return [[]]
	else:
		car = lstguy[0]
		cdr = lstguy[1:]
		# sys.stderr.write("car: %s, cdr: %s\n" % (car, cdr))
		smaller = powerSet(cdr)
		# sys.stderr.write("smaller: %s\n" % (smaller))	
		# rets = smaller[:]
		# for s in smaller:
		# 	rets.append([car] + s)
		# return rets
		return smaller + [ [car] + s for s in smaller ] 	
	
def qsub(pathToScript, argsScript=None, outDir=None, argsGrid=None):	
	cmd = "qsub -cwd -S /usr/bin/python "
	if (outDir):
		cmd += "-e %s " % outDir
		cmd += "-o %s " % outDir
	if (argsGrid):
		cmd += argsGrid + " "
	cmd += pathToScript + " "		
	if (argsScript):
		cmd += (" ".join([ str(arg) for arg in argsScript ])) + " "
	sys.stderr.write(cmd + "\n")
	os.system(cmd)

def qsubIfNotDone(pathToScript, combos, outputDir, outputGlob, argsGrid=None):
	done = getRunTotals(outputGlob, len(combos[0])) # (input0, input1, ..., input12) -> count

	jobs = 0
	for combo in combos:
		if not (combo in done):
			sys.stderr.write("submitting job for %s\n" % (str(combo)))
			qsub(pathToScript, combo, outputDir, argsGrid)	
			jobs += 1
		else:	
			sys.stderr.write("skipping run for %s\n" % (str(combo)))
	return jobs



# see what runs we've already done
def getRunTotals(fileGlob, numInputFields, delimit="\t", startIndexInputFields=0):
	outfiles = glob.glob(fileGlob)
	sys.stderr.write("found %d output files\n" % len(outfiles))

	done = {} # key -> count
	for outfile in outfiles:
		f = open(outfile, 'r')
		# labels = f.readline().strip().split(delimit)[startIndexInputFields:(startIndexInputFields+numInputFields)]
		# labelToIndex = dict(zip(labels, range(len(labels))))
		# labelsSorted = sorted(labels)
		# labelsSortedIndices = [ labelToIndex[z] for z in labelsSorted ]

		lines = f.readlines()
		sys.stderr.write("reading %s: %d lines\n" % (outfile, len(lines)))
		for line in lines:
			elts = line[:-1].split(delimit)
			eltsKey = tuple(elts[startIndexInputFields:(startIndexInputFields+numInputFields)])			
			# eltsKeySorted = tuple([ eltsKey[i] for i in labelsSortedIndices ])
			
			if (eltsKey in done):
				done[eltsKey] += 1
			else:
				done[eltsKey] = 1

	sys.stderr.write("done:\n")
	for combo, count in done.items():
		sys.stderr.write("%s\t%d\n" % (combo, count))
	sys.stderr.write("\n")
	return done


def serverEstimate(jobs, servers):
	ret = "%d jobs with %d servers:\n" % (jobs, servers)
	jobsPerServer = int(math.ceil(float(jobs)/servers))	
	for minsPerJob in [10, 20, 30, 40, 50, 60]:
		minsPerServer = minsPerJob*jobsPerServer
		hoursPerServer = minsPerServer/60.0
		ret += "\t%d minutes per job =\t%.1f hours\n" % (minsPerJob, hoursPerServer)
	return ret

def quot(stuff, format="%s"):
	return [ format % s for s in stuff ]
		
def tupsToLists(tups):
	arity = len(tups[0])
	lists = [ [ x[i] for x in tups ] for i in range(arity) ]
	return lists


def wrapToWidth(strguy, width, sep=None):
	sepNew = " " if (sep is None) else sep 
	elts = strguy.split(sep)
	pos = 0
	ret = ""
	for elt in elts:
		if ((pos + len(elt)) > width):
			ret += "\n" + elt + sepNew
			pos = len(elt)
		else:
			ret += elt + sepNew
			pos += len(elt)
	return ret

def clipText(strguy, width, length=None):
	return "\n".join([ line[:width] for line in strguy.split("\n")[:(sys.maxint if (length is None) else length)] ])

def fixedColumnWidth(elts, colwidth=20):
	return "".join([ str(e).rjust(colwidth) for e in elts ])	
	
def fixedColumnWidthTsv(tabsepguy, colwidth=20):
	rets = []
	for line in tabsepguy.split("\n"):
		rets.append(fixedColumnWidth(line.split("\t"), colwidth=colwidth))
	return "\n".join(rets)

def castNumberFromStr(sOrig):
	if (sOrig[0] == '-'):
		s = sOrig[1:]
	else:
		s = sOrig

	if (s.isdigit()):
		return int(sOrig)
	elif (s.replace(".", "", 1).isdigit()):
		return float(sOrig)
	else:
		return sOrig

def flipDict(dictguy, isBiject=True):
	flipped = {}
	if (isBiject):
		for k, v in dictguy.items():
			flipped[v] = k
	else:
		for k, v in dictguy.items():
			flipped.setdefault(v, []).append(k)
	return flipped



