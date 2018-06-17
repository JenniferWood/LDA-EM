import doc,math,copy

from scipy.special import gamma as gafunc
from scipy.special import digamma,polygamma,gammaln
import random,heapq
from gensim import corpora,models
import numpy as np

d = doc.Documents()
d.readDocs()

class LDA:

	def __init__(self,k):
		self.K = k
		self.rho = 0.001
		self.rhoe = 0.003
		self.rhoem = 0.001

		self.docs = d.docs
		self.wd = d.wd

		self.alpha = [50.0 / self.K] * self.K
		self.Beta = [[1.0/len(self.wd)] * len(self.wd)] * self.K

		self.maxL = float('-inf')

	def gensimtest(self):
		dic = corpora.Dictionary(self.docs)
		corpus = [dic.doc2bow(text) for text in self.docs]
		tfidf = models.TfidfModel(corpus)
		corpus_tfidf = tfidf[corpus]
		print corpus

	def samplefrom(self,p):
		i=1
		while i<len(p):
			p[i] += p[i-1]
			i+=1

		#print p
		sp = random.uniform(0,1)
		#print sp

		x = 0
		for pi in p:
			if pi > sp: return x
			x += 1

		if x==len(p): x-=1
		return x

	def EM(self):
		iteration = 0

		
		for k in range(self.K):
			#self.alpha[k] = random.uniform(0,50.0/self.K)
			for i in range(len(self.Beta[k])):
				self.Beta[k][i] = random.uniform(0,1)
			self.Beta[k] = self.normalize(self.Beta[k])
		
		#self.alpha = self.normalize(self.alpha)

		print self.alpha
		print self.Beta[0][0:10]

		while  True:
			print "iteration",iteration

			self.Phi = [[[float(1)/self.K] * self.K]*len(doc) for doc in self.docs]
			self.Gamma = [[ak+float(len(doc))/self.K for ak in self.alpha] for doc in self.docs]
			
			#E-STEP

			print "############# E-STEP ################"
			et = 0
			while  True:
				print "trying the %d times" % et,
				phi = copy.deepcopy(self.Phi)
				gamma = copy.deepcopy(self.Gamma)

				for d in range(len(self.docs)):
					lbias = [0 for a in self.alpha]
					for n in range(len(self.docs[d])):
						for k in range(self.K):
							i = self.docs[d][n]
							self.Phi[d][n][k] = self.Beta[k][i]*math.exp(digamma(self.Gamma[d][k]) - digamma(sum(self.Gamma[d])))
						self.Phi[d][n] = self.normalize(self.Phi[d][n]) # Update Phi
						lbias = map(lambda (a,b):a+b, zip(lbias,self.Phi[d][n]))

					for k in range(self.K):
						self.Gamma[d][k] = self.alpha[k] + lbias[k]

				edis = self.distance([phi,gamma],[self.Phi,self.Gamma])
				print edis
				if  edis < self.rhoe:
					break

				et+=1
					
			

			print "new Phi:",self.Phi[0][0]
			print "new Gamma:",self.Gamma[0]
			print ""

			print "############# M-STEP ################"
			
			self.alpha = [50.0 / self.K] * self.K
			
			#M-STEP
			altmp = copy.deepcopy(self.alpha)
			betmp = copy.deepcopy(self.Beta)

			#update alpha
			self.Newton(self.alpha)

			#Update Beta
			print("2, updating beta------------------")
			for k in range(self.K):
				for i in range(len(self.wd)):
					self.Beta[k][i] = 0
					for d in range(len(self.docs)):
						for n in range(len(self.docs[d])):
							if self.docs[d][n] == i:
								self.Beta[k][i] += self.Phi[d][n][k]
				self.Beta[k] = self.normalize(self.Beta[k])

			print self.Beta[0][0:10]

			#edis = self.distance([phi,gamma,altmp,betmp],[self.Phi,self.Gamma,self.alpha,self.Beta])
			#print "dis: ",edis
			#if  edis < self.rhoe:
			#	break
			#et +=1
			
			if self.converged() is True:
				print "model converged"
				break
			
			self.savemodel()

			iteration += 1
			print ""

	def savemodel(self):
		out = file('model/lda_iv.alpha','w')
		for alphai in self.alpha:
			out.write('%f\t' % alphai)
		out.write("\n")


		out = file('model/lda_iv.beta','w')
		for ltopic in self.Beta:
			for lw in ltopic:
				out.write('%f\t' % lw)
			out.write("\n")

		out = file('model/lda_iv.gamma','w')
		for gdoc in self.Gamma:
			gdoc = self.normalize(gdoc)
			for gw in gdoc:
				out.write('%f\t' % gw)
			out.write("\n")
			
		out = file('model/lda_iv.phi','w')
		for pdoc in self.Phi:
			for pn in pdoc:
				for pw in pn:
					out.write('%f\t' % pw)
				out.write("\n")

		out = file('model/lda_iv.words','w')
		for ptopic in self.Beta:
			s = self.findminm(ptopic,20)
			for (pki,i) in s:
				out.write("%s %f\t" % (self.wd[i], pki))
			out.write("\n\n")

		out = file('model/lda_iv.phi','w')
		dnk = {}
		for i in range(len(self.Phi)):
			for j in range(len(self.Phi[i])):
				k = pdoc.index(max(pdoc[j]))
				dnk.setdefault(k,[])
				if self.docs[i][j] not in dnk[k]:
					dnk[k].append(self.docs[i][j])
		for k in dnk:
			s = self.findminm(self.Beta[k],20,dnk[k])
			for (pki,i) in s:
				out.write("%s %f\t" % (self.wd[i], pki))
			out.write("\n\n")


	def findminm(self,l,num,atten=[]):
		tmp = l[0:num]
		heap = []

		i=0
		while i<num:
			if len(atten) >0 and i not in atten:continue
			heapq.heappush(heap,(l[i],i))
			i+=1
		#print heap

		while i<len(l):
			if len(atten) >0 and i not in atten:continue
			if l[i] > heap[0][0]:
				t = heapq.heapreplace(heap,(l[i],i))
				#print heap
			i+=1
		return heap


	def Newton(self,vec):
		if len(vec) == self.K:
			print "1, updating alpha------------------"
			ratio = len(self.docs)

		else:
			print "2, updating beta-------------------"
			ratio = self.K

		
		veck = copy.deepcopy(vec)
		
		t=0
		while True:
			print "updating the %d times" % t
			#print "x%d"%t,veck[0:10]

			gk = self.grad(veck)
			#print "gk%d"%t,gk[0:10]
			if self.normof(gk) < self.rho:
				print "after udating:",veck[0:10]
				print ""

				if len(vec) == self.K:
					self.alpha = veck
				else:
					self.beta = veck
				return

			Hk = [[ratio*polygamma(1,sum(veck))]*len(veck)]*len(veck)
			duijiao = [ratio*polygamma(1,vecki) for vecki in veck]
			Hk = np.mat(Hk) - np.mat(np.diag(duijiao))
			#print "Hk%d"%t,Hk[0]

			pk = (-1*(Hk.I)*(np.mat(gk).T)).T.tolist()[0]
			#print "pk%d"%t,pk[0:10]

			for i in range(len(veck)):
				veck[i] += pk[i]
			t += 1
			
    # It's actually math.pow(realnorm,2)
	def normof(self,vec):
		norm2 = sum([v*v for v in vec])
		#print "norm2",norm2
		return norm2

	def grad(self,vec):
		if len(vec) == self.K:
			ratio = len(self.docs)
			bias = self.Gamma

		else:
			ratio = self.K
			bias = self.Lambda

		ds = digamma(sum(vec))
		grads = [ratio*ds-ratio*digamma(veci) for veci in vec]
		for i in range(len(vec)):
			for j in range(ratio):
				grads[i] += digamma(bias[j][i])-digamma(sum(bias[j]))
		
		#print "grads",grads[0:10]
		return grads
		

	def normalize(self,p):
		#print "before normalize",p
		s = sum(p)
		p = [pi/s for pi in p]
		#print "after normalize",p
		return p

	def Dig(self,x):
		if x == 0: return 0
		if x<0:
			x = -1*x
			return self.Dig(x)+math.pi*math.cos(x*math.pi)+1/x
		if x == 1: return -0.57721566490153286
		
		z = int(x)
		d = x-z
		if d == 0: 
			value = -0.57721566490153286
			d = 1
			z -= 1
		else:
			d2 = d*d
			d3 = d2*d
			value = -0.515095835950807+1.389927456533864*d-1.008336779674558/d+0.000557958765350/d2-0.586786525683560*d2-0.000014382050162/d3+0.142984009331572*d3

		i=0
		while i<z:
			value += float(1)/d
			d+=1
			i+=1 

		return value

	def Ddig(self,x):
		if x==0: return 0
		return float(1)/x + float(1)/(2*x*x)

	def distance(self,listA,listB):
		result = 0.0
		for i in range(len(listA)):
			if isinstance(listA[i],list):
				result += self.distance(listA[i],listB[i])
			else:
				result += math.pow(listB[i]-listA[i],2)
		return result

	def converged(self):
		E = gammaln(sum(self.alpha))-sum([gammaln(alphai) for alphai in self.alpha])
		for d in range(len(self.docs)):
			for k in range(self.K):
				E += sum([gammaln(g) for g in self.Gamma[d]]) - gammaln(sum(self.Gamma[d]))
				E += (self.alpha[k] - self.Gamma[d][k])*(digamma(self.Gamma[d][k]) - digamma(sum(self.Gamma[d])))
			for n in range(len(self.docs[d])):
				for k in range(self.K):
					E += self.Phi[d][n][k]*(digamma(self.Gamma[d][k]) - digamma(sum(self.Gamma[d])))
					
					i = self.docs[d][n]
					E += self.Phi[d][n][k]*(digamma(self.Beta[k][i]) - digamma(sum(self.Beta[k])))
					E -= self.Phi[d][n][k]*math.log(self.Phi[d][n][k])

		#E = math.fabs(E)
		print "L NOW IS",E
		if E <= self.maxL:
			return True

		else:
			self.maxL = E
			return False