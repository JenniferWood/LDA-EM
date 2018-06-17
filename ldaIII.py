import doc,math,copy
import numpy as np
from scipy.special import gamma as gafunc
from scipy.special import digamma,polygamma,gammaln
import random

d = doc.Documents()
d.readDocs()

class LDA:

	def __init__(self,k):
		self.K = k
		self.rho = 0.000001
		self.rhoe = 0.000002
		self.rhoem = 0.0001

		self.docs = d.docs
		self.wd = d.wd
		self.wn = d.wn

		self.alpha = [50.0 / self.K] * self.K
		self.beta = [[0.01] * len(self.wd)] * self.K

		self.maxL = float('-inf')

	def EM(self):
		iteration = 0

		while  True:
			print "iteration",iteration
			self.Phi = [[[float(1)/self.K] * self.K]*len(doc) for doc in self.docs]
			self.Gamma = [[ak+float(len(doc))/self.K for ak in self.alpha] for doc in self.docs]
			#self.Lambda = [[bi+float(self.K)/len(self.beta) for bi in self.beta]]*self.K

			#E-STEP

			print "############# E-STEP ################"
			et = 0
			while  True:
				print "trying the %d times" % et,
				phi = copy.deepcopy(self.Phi)
				gamma = copy.deepcopy(self.Gamma)
				beta = copy.deepcopy(self.beta)


				for d in range(len(self.docs)):
					lbias = [0 for a in self.alpha]
					for n in range(len(self.docs[d])):
						for k in range(self.K):
							i = self.docs[d][n]
							self.Phi[d][n][k] = math.exp(digamma(self.Gamma[d][k]) - digamma(sum(self.Gamma[d])) \
								+ digamma(self.beta[k][i]) - digamma(sum(self.beta[k])))
						self.Phi[d][n] = self.normalize(self.Phi[d][n]) # Update Phi
						lbias = map(lambda (a,b):a+b, zip(lbias,self.Phi[d][n]))

					for k in range(self.K):
						self.Gamma[d][k] = self.alpha[k] + lbias[k]
					

				#Update beta
				for k in range(self.K):
					for j in range(len(self.wd)):
						self.beta[k][j] = 0
						for d in range(len(self.docs)):
							for n in range(len(self.docs[d])):
								if self.docs[d][n] == j:
									self.beta[k][j] += self.Phi[d][n][k]

				edis = self.distance([phi,gamma,beta],[self.Phi,self.Gamma,self.beta])
				print ":",edis
				if  edis < self.rhoe:
					break

				et +=1

			print "new Phi:",self.Phi[0][0]
			print "new Gamma:",self.Gamma[0]
			print "new beta:",self.beta[0][0:10]
			print ""
			#break
			print "############# M-STEP ################"
			#M-STEP
			altmp = copy.deepcopy(self.alpha)
			
			#update alpha
			self.Newton()
			break
			#update beta
			#self.Newton(self.beta)

			if self.converged() is True:
				print "Model Converged"
				self.savemodel()
				break
			
			iteration += 1

		

	def Newton(self):
		print "1, updating alpha------------------"
		ratio = len(self.docs)
		
		veck = copy.deepcopy(self.alpha)
		
		t=0
		while True:
			print "updating the %d times" % t
			print "x%d"%t,veck[0:10]

			gk = self.grad()
			print "gk%d"%t,gk[0:10]
			if self.normof(gk) < self.rho:
				print "after udating:",veck[0:10]
				print ""

				self.alpha = veck
				return

			Hk = [[ratio*polygamma(1,sum(veck))]*len(veck)]*len(veck)
			duijiao = [ratio*polygamma(1,vecki) for vecki in veck]
			Hk = np.mat(Hk) - np.mat(np.diag(duijiao))
			#print "Hk%d"%t,Hk[0]

			pk = (-1*(Hk.I)*(np.mat(gk).T)).T.tolist()[0]
			print "pk%d"%t,pk[0:10]
			break
			for i in range(len(veck)):
				veck[i] += pk[i]
			t += 1
			

	def L(self,vec,type):
		result = 0
		if type == 0:
			for d in range(len(self.docs)):
				result += math.log(gafunc(sum(vec)) - sum([math.log(gafunc(a)) for a in vec]))
				for k in range(self.K):
					result += (vec[k]-1)*(digamma(self.Gamma[d][k] - digamma(sum(self.Gamma[d]))))
		else:
			result = self.K * math.log(gafunc(sum(vec))) - self.K * sum([math.log(gafunc(a)) for a in vec])
			for k in range(self.K):
				for i in range(len(self.wd)):
					result+= (vec[i]-1)*(digamma(self.Lambda[k][i] - digamma(sum(self.Lambda[k]))))
		print "L",result
		return result

    # It's actually math.pow(realnorm,2)
	def normof(self,vec):
		norm2 = sum([v*v for v in vec])
		#print "norm2",norm2
		return norm2

	def grad(self):
		ratio = len(self.docs)
		
		ds = digamma(sum(self.alpha))
		grads = [ratio*ds-ratio*digamma(veci) for veci in self.alpha]
		for i in range(self.K):
			for d in range(ratio):
				grads[i] += digamma(self.Gamma[d][i])-digamma(sum(self.Gamma[d]))
		
		#print "grads",grads[0:10]
		return grads
		

	# Save The Parameters
	def savemodel(self):
		print self.Phi[0]
		print self.Lambda

		'''
		z = {}
		for d in xrange(len(self.docs)):
			for n in xrange(len(self.docs[d])):
				v = self.docs[d][n]
				w = self.wd[v]
		'''	

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
		E = self.K*gammaln(sum(self.beta))-self.K*sum([gammaln(betai) for betai in self.beta])
		E += gammaln(sum(self.alpha))-sum([gammaln(alphak) for alphak in self.alpha])
		for k in range(self.K):
			E -= gammaln(sum(self.Lambda[k]))
			E += sum([gammaln(self.Lambda[k][i])+(self.beta[i]-self.Lambda[k][i])*(digamma(self.Lambda[k][i]) - digamma(sum(self.Lambda[k]))) for i in range(len(self.wd))])
		for d in range(len(self.docs)):
			E -= gammaln(sum(self.Gamma[d]))
			E += sum([gammaln(self.Gamma[d][k])+(self.alpha[k]-self.Gamma[d][k])*(digamma(self.Gamma[d][k])-digamma(sum(self.Gamma[d]))) for k in range(self.K)])
			for n in range(len(self.docs[d])):
				for k in range(self.K):
					E += self.Phi[d][n][k]*(digamma(self.Gamma[d][k])-digamma(sum(self.Gamma[d]))-math.log(self.Phi[d][n][k]))
					i = self.docs[d][n]
					E += self.Phi[d][n][k]*(digamma(self.Lambda[k][i])-digamma(sum(self.Lambda[k])))

		print "L NOW IS",E
		if E <= self.maxL:
			return True

		else:
			self.maxL = E
			return False