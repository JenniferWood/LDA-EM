import doc,math,copy
import numpy as np
from scipy.special import gamma as gafunc

d = doc.Documents()
d.readDocs()

class LDA:

	def __init__(self):
		self.K = 5
		self.rho = 0.000001
		self.rhoem = 0.00001

		self.docs = d.docs
		self.wd = d.wd

		self.alpha = [float(50)/self.K] * self.K
		self.beta = [0.01] * len(self.wd)
		self.Phi = [[[float(1)/self.K] * self.K]*len(doc) for doc in self.docs]
		self.Gamma = [[ak+float(len(doc))/self.K for ak in self.alpha] for doc in self.docs]
		self.Lambda = [[1.0]*len(self.wd)]*self.K

	def EM(self):
		iteration = 0
		while  True:
			print "iteration",iteration
			#E-STEP
			phi = copy.deepcopy(self.Phi)
			gamma = copy.deepcopy(self.Gamma)
			lamda = copy.deepcopy(self.Lambda)

			for d in range(len(self.docs)):
				Sgamma = sum(self.Gamma[d])
				Slambda = {}
				for n in range(len(self.docs[d])):
					for k in range(self.K):
						i = self.docs[d][n]
						if k not in Slambda:
							Slambda[k] = sum(self.Lambda[k])
						self.Phi[d][n][k] = self.Dig(self.Lambda[k][i])-self.Dig(Slambda[k])\
										+ self.Dig(self.Gamma[d][k])-self.Dig(Sgamma)
					self.Phi[d][n] = self.normalize(self.Phi[d][n]) # Update Phi
				for k in range(self.K):
					self.Gamma[d][k] = self.alpha[k] + sum([self.Phi[d][nn][k] for nn in range(len(self.docs[d]))]) # Update Gamma

			#Update Lambda
			for k in range(self.K):
				for i in range(len(self.wd)):
					self.Lambda[k][i] = self.beta[i] + sum([sum([self.Phi[d][n][k] for n in range(len(self.docs[d])) if self.docs[d][n]==i]) for d in range(len(self.docs))])

			#M-STEP
			altmp = copy.deepcopy(self.alpha)
			betmp = copy.deepcopy(self.beta)
			self.alpha = self.M(0)
			self.beta = self.M(1)

			if self.converged(phi,gamma,lamda,altmp,betmp) is True:
				print "Model Converged"
				self.savemodel()
				break
			iteration += 1

	def M(self,parm):
		if parm == 0:
			dim = len(self.alpha)
			vec = self.alpha

		else:
			dim = len(self.beta)
			vec = self.beta

		print "-------------",parm,"-----------------"

		Dk = np.mat(np.eye(dim,dim)) # Initial I
		veck = copy.deepcopy(vec)
		
		while True:
			gk = self.grad(veck,parm)
			if self.normof(gk) < self.rho:
				return veck

			pk = (-1*Dk*(np.mat(gk).T)).T.tolist()[0]

			print "Dk",Dk[0:10]
			print "gk",gk[0:10]
			print "pk",pk[0:10]

			miu = 0.1
			s = []

			minL = self.L(veck,parm)
			while miu<=1:
				veck1 = [vec[i]+miu*pk[i] for i in range(len(vec))]
				Ltmp = self.L(veck1,parm)
				if self.L(veck1,parm) > minL: 
					break
				minL = Ltmp
				miu += 0.1

			veck = [vec[i]+miu*pk[i] for i in range(len(vec))]
			s = [miu*pki for pki in pk]
			
			gk1 = self.grad(veck,parm)
			if self.normof(gk1) < self.rho:
				return veck

			y = [gk1[i]-gk[i] for i in range(len(pk))]
			s = np.mat(s)
			y = np.mat(y)

			tmp = y.T * s

			left = np.mat(np.eye(dim,dim)) - (s*(y.T))/tmp[0,0]
			Dk = left*Dk*(left.T) + (s*(s.T))/tmp[0,0]

	def L(self,vec,type):
		result = 0
		if type == 0:
			for d in range(len(self.docs)):
				result += math.log(gafunc(sum(vec)) - sum([math.log(gafunc(a)) for a in vec]))
				for k in range(self.K):
					result += (vec[k]-1)*(self.Dig(self.Gamma[d][k] - self.Dig(sum(self.Gamma[d]))))
		else:
			result = self.K * math.log(gafunc(sum(vec))) - self.K * sum([math.log(gafunc(a)) for a in vec])
			for k in range(self.K):
				for i in range(len(self.wd)):
					result+= (vec[i]-1)*(self.Dig(self.Lambda[k][i] - self.Dig(sum(self.Lambda[k]))))
		print "L",result
		return result

    # It's actually math.pow(realnorm,2)
	def normof(self,vec):
		norm2 = sum([v*v for v in vec])
		print "norm2",norm2
		return norm2

	def grad(self,vec,type):
		ds = self.Dig(sum(vec))
		if type == 0:
			M = len(self.docs)
			grads = [M*(ds-self.Dig(vec[k]))+sum([(self.Ddig(self.Gamma[d][k])-self.Ddig(sum(self.Gamma[d]))) for d in range(M)]) for k in range(self.K)]
		else:
			V = len(self.wd)
			grads = [self.K*(ds-self.Dig(vec[i]))+sum([(self.Ddig(self.Lambda[k][i])-self.Ddig(sum(self.Lambda[k]))) for k in range(self.K)]) for i in range(V)]
		
		print "grads",grads[0:10]
		return grads
		

	# Save The Parameters
	def savemodel(self):
		print self.alpha
		print self.beta[0:10]

	def normalize(self,p):
		s = sum(p)
		p = [pi/s for pi in p]
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

	def converged(self,phi,gamma,lamda,altmp,betmp):
		dalpha = sum([math.pow(altmp[i]-self.alpha[i],2) for i in range(self.K)])
		dbeta = sum([math.pow(betmp[i]-self.beta[i],2) for i in range(len(betmp))])
		dphi = 0
		dgamma = 0
		for d in range(len(self.docs)):
			for n in range(len(self.docs[d])):
				for k in range(self.K):
					dphi += math.pow(phi[d][n][k]-self.Phi[d][n][k],2)
			for k in range(self.K):
				dgamma += math.pow(gamma[d][k]-self.Gamma[d][k],2)
		dlambda = 0
		for k in range(self.K):
			for v in range(len(self.wd)):
				dlambda += math.pow(lamda[k][v]-self.Lambda[k][v],2)
		maxdiff = max([dalpha,dbeta,dphi,dgamma,dlambda])
		print "max-diff",maxdiff
		if maxdiff < self.rhoem:
			return True
		return False