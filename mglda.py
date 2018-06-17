#-*-encoding:utf-8-*-
import math,random,heapq
import doc
d = doc.Documents()
d.readDocs(path='/Users/apple/Documents/GITHUB/基于评论的推荐/Crawlers/jieba分词/')

class LDA:

	def __init__(self,gk,lk):
		self.GK, self.LK= gk,lk

		self.docs = d.docs
		self.wd = d.wd

		self.alphaG = 50.0 / self.GK
		self.alphaL = 50.0 / self.LK
		self.alphaM = [0,0]
		self.beta = 0.01

		self.maxL = float('-inf')

	def GibbsSamppling(self):
		#initial
		self.r = []
		self.z = []
		
		self.Nglzw = [[0] * len(self.wd)] * self.GK
		self.Nglz = [0] * self.GK
		self.Nloczw = [[0] * len(self.wd)] * self.LK
		self.Nlocz = [0] * self.LK

		self.Ndsr = []
		self.Nds = []
		
		self.Ndglz = [[0] * self.GK] * len(self.docs)
		self.Ndgl = [0] * len(self.docs)
		self.Ndslocz = []
		self.Ndsloc = []

		d = 0
		for doc in self.docs:
			#print "doc",d
			self.Ndsr.append([[0]*2]*len(doc))
			self.Nds.append([0]*len(doc))
			self.Ndslocz.append([[0]*self.LK]*len(doc))
			self.Ndsloc.append([0] * len(doc))

			rd = []
			zd = []
			s = 0
			for sent in doc:
				rs = []
				zs = []

				w = 0
				for sword in sent:
					v = self.docs[d][s][w]

					ther = random.randint(0,1)
					rs.append(ther)
					if ther == 0:
						thez = random.randint(0,self.GK-1)
						self.Nglzw[thez][v] += 1
						self.Nglz[thez] += 1
						self.Ndsr[d][s][0] += 1
						self.Nds[d][s] += 1
						self.Ndglz[d][thez] += 1
						self.Ndgl[d] += 1
						self.alphaM[0] += 1
					else:
						thez = random.randint(0,self.LK-1)
						self.Nloczw[thez][v] += 1
						self.Nlocz[thez] += 1
						self.Ndsr[d][s][1] += 1
						self.Nds[d][s] += 1
						self.Ndslocz[d][s][thez] += 1
						self.Ndsloc[d][s] += 1
						self.alphaM[1] += 1

					zs.append(thez)
					w+=1
				s+=1
				zd.append(zs)
				rd.append(rs)
			d+=1
			self.r.append(rd)
			self.z.append(zd)

		#iterations
		it = 0
		while it<1000:
			print "iterations",it
			if it % 100 == 0 or it>=700 and (it-700)%20 == 0:
				print "saving model at iteration",it
				self.savemodel(it)

			for d in range(len(self.docs)):
				for s in range(len(self.docs[d])):
					for w in range(len(self.docs[d][s])):
						oldr = self.r[d][s][w]
						oldz = self.z[d][s][w]
						v = self.docs[d][s][w]
						if oldr == 0:
							self.Nglzw[oldz][v] -= 1
							self.Nglz[oldz] -= 1
							self.Ndsr[d][s][0] -= 1
							self.Nds[d][s] -= 1
							self.Ndglz[d][oldz] -= 1
							self.Ndgl[d] -= 1
							self.alphaM[0] -= 1
						else:
							self.Nloczw[oldz][v] -= 1
							self.Nlocz[oldz] -= 1
							self.Ndsr[d][s][1] -= 1
							self.Nds[d][s] -= 1
							self.Ndslocz[d][s][oldz] -= 1
							self.Ndsloc[d][s] -= 1
							self.alphaM[1] -= 1

						p = []

						for k in range(self.GK):
							pk = (self.Nglzw[k][v]+self.beta)/(self.Nglz[k]+len(self.wd)*self.beta)\
								*(self.Ndglz[d][k]+self.alphaG)/(self.Ndgl[d]+self.GK*self.alphaG)*(self.Ndsr[d][s][0]+self.alphaM[0])
							p.append(pk)
						for k in range(self.LK):
							pk = (self.Nloczw[k][v]+self.beta)/(self.Nlocz[k]+len(self.wd)*self.beta)\
								*(self.Ndslocz[d][s][k]+self.alphaL)/(self.Ndsloc[d][s]+self.LK*self.alphaL)*(self.Ndsr[d][s][1]+self.alphaM[1])
							p.append(pk)
						
						p = self.normalize(p)

						newk = self.samplefrom(p)
						if newk < self.GK:
							self.r[d][s][w] = 0
							newz = newk
							self.Nglzw[newz][v] += 1
							self.Nglz[newz] += 1
							self.Ndsr[d][s][0] += 1
							self.Nds[d][s] += 1
							self.Ndglz[d][newz] += 1
							self.Ndgl[d] += 1
							self.alphaM[0] += 1
						else:
							self.r[d][s][w] = 1
							newz = newk - self.GK
							self.Nloczw[newz][v] += 1
							self.Nlocz[newz] += 1
							self.Ndsr[d][s][1] += 1
							self.Nds[d][s] += 1
							self.Ndslocz[d][s][newz] += 1
							self.Ndsloc[d][s] += 1
							self.alphaM[1] += 1
						self.z[d][s][w] = newz
			it+=1

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

	def normalize(self,p):
		#print "before normalize",p
		s = sum(p)
		p = [float(pi)/s for pi in p]
		#print "after normalize",p
		return p

	def savemodel(self,n):
		gphi = [[0.0]*len(self.wd)]*self.GK
		out = file('model/lda%d.gphi'%n,'w')
		for k in range(self.GK):
			for v in range(len(self.wd)):
				gphi[k][v] = (self.Nglzw[k][v]+self.beta)/(self.Nglz[k]+len(self.wd)*self.beta)
				out.write("%f\t" % gphi[k][v])
			out.write("\n")

		lphi = [[0.0]*len(self.wd)]*self.LK
		out = file('model/lda%d.lphi'%n,'w')
		for k in range(self.LK):
			for v in range(len(self.wd)):
				lphi[k][v] = (self.Nloczw[k][v]+self.beta)/(self.Nlocz[k]+len(self.wd)*self.beta)
				out.write("%f\t" % lphi[k][v])
			out.write("\n")

		out = file('model/lda%d.gtheta'%n,'w')
		for d in range(len(self.docs)):
			for k in range(self.GK):
				out.write("%f\t" % ((self.Ndglz[d][k]+self.alphaG)/(self.Ndgl[d]+self.GK*self.alphaG)))
			out.write("\n")

		out = file('model/lda%d.ltheta'%n,'w')
		for d in range(len(self.docs)):
			for s in range(len(self.docs[d])):
				for k in range(self.LK):
					out.write("%f\t" % ((self.Ndslocz[d][s][k]+self.alphaL)/(self.Ndsloc[d][s]+self.LK*self.alphaL)))
			out.write("\n")

		out = file('model/lda%d.gwords'%n,'w')
		for ptopic in gphi:
			s = self.findminm(ptopic,20)
			for (pki,i) in s:
				out.write("%s %f\t" % (self.wd[i], pki))
			out.write("\n\n")

		out = file('model/lda%d.lwords'%n,'w')
		for ptopic in lphi:
			s = self.findminm(ptopic,20)
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

	def converged(self):
		return False