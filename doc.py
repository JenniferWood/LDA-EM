#-*-encoding:utf-8-*-
import os,re

class Documents:
	def __init__(self):
		self.docs = []
		self.wd = []
		self.stopwordlist = []
		for line in open('stopwords.txt'):
			self.stopwordlist.append(line)

	def readDocs(self,path='data/'):
		dirlist = os.listdir(path)
		
		j = 1
		for dire in dirlist:
			if dire == '.DS_Store':continue
			doclist = os.listdir("%s%s" % (path,dire))
			for doc in doclist:
				if doc == '.DS_Store':continue
				print j,doc
				j+=1
				dw = []
				for line in open(path+dire+'/'+doc):

					sw = []
					line = line.split()
					for word in line:
						raw = word[0:word.index('/')]
						pos = word[word.index('/')+1:]
						if raw in self.stopwordlist: continue
						if 'x' not in pos and 'n' not in pos and 'a' not in pos: continue
						if raw not in self.wd:
							self.wd.append(raw)
						i = self.wd.index(raw)
						sw.append(i)
					if len(sw)>0:
						dw.append(sw)
				self.docs.append(dw)
		
	def separatewords(self,line):
		#wl = re.compile('\\W+').split(line)
		wl = line.split()
		return wl