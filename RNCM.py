# Imports
import numpy as np

#Code for SVNS nodes for RNCMs
class SVNS():
    def __init__(self,P,N,I):
        self.P=P
        self.N=N
        self.I=I
    def __str__(self):
        return "<{},{},{}>".format(self.P,self.N,self.I)
    def __add__(self,other):
        return SVNS(self.P+other.P,self.N+other.N,self.I+other.I)
    def __mul__(self,other):
        return SVNS(self.P*other.PP+self.N*other.NP,self.N*other.NN+self.P*other.PN,self.I*other.I)
    def __rmul__(self,other):
        return SVNS(self.P*other.PP+self.N*other.NP,self.N*other.NN+self.P*other.PN,self.I*other.I)
    def __sub__(self,other):
        return SVNS(self.P-other.P,self.N-other.N,self.I-other.I)
    def __repr__(self):
        return "<{:.2f},{:.2f},{:.2f}>".format(self.P,self.N,self.I)
    def sigmoid(self,lmb):
        return SVNS(1/(1+np.exp(-lmb * self.P)),1/(1+np.exp(-lmb * self.N)),1/(1+np.exp(-lmb * self.I)))
    def __eq__(self,other):
        return(self.P==other.P and self.N==other.N and self.I==other.I)
    def norm(self):
        if(max(self.P,self.N,self.I)<0.5):
            return SVNS(0,0,0)
        if(self.P==0 and self.N==0 and self.I==0):
            return(SVNS(0,0,0))
        if(self.P>=self.N and self.P>=self.I):
            return(SVNS(1,0,0))
        if(self.N>=self.I):
            return(SVNS(0,1,0))
        else:
            return(SVNS(0,0,1))
    def isNull(self):
        return(self.P==0 and self.N==0 and self.I==0)
    def conv_int(self):
        return(SVNS_lite(int(self.P),int(self.N),int(self.I)))
    

#Code for RDNS edges for RNCMs
class edge():
    def __init__(self,PP,PN,NP,NN,I):
        self.PP=PP
        self.PN=PN
        self.NP=NP
        self.NN=NN
        self.I=I
    def __str__(self):
        return "<{},{},{},{},{}>".format(self.PP,self.PN,self.NP,self.NN,self.I)
    def __repr__(self):
        return "<{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}>".format(self.PP,self.PN,self.NP,self.NN,self.I)
    def __mul__(self,other):
        return SVNS(other.P*self.PP+other.N*self.NP,other.N*self.NN+other.P*self.PN,self.I*other.I)
    

#read line from weigths.txt

def read_data(file_path):
    with open(file_path) as f:
        lines = [line.rstrip() for line in f]
    #remove whitespace characters like `\n` at the end of each line
    matr=[]
    for line in lines:
        ele=line.split(" ")
        l1=[]
        for e in ele:
            e=e.strip()
            e=e[1:-1]
            P,N,I=e.split(",")
            l1.append(SVNS(float(P),float(N),float(I)))
        matr.append(l1)
    matr=np.array(matr)
    return matr


def read_edges(file_path):
    with open(file_path) as f:
        lines = [line.rstrip() for line in f]
    #remove whitespace characters like `\n` at the end of each line
    matr=[]
    for line in lines:
        ele=line.split(" ")
        l1=[]
        for e in ele:
            e=e.strip()
            e=e[1:-1]
            PP,PN,NP,NN,I=e.split(",")
            l1.append(edge(float(PP),float(PN),float(NP),float(NN),float(I)))
        matr.append(l1)
    matr=np.array(matr)
    return matr

if __name__ == "__main__":
    weights=read_edges("weights.txt")
    initial_state=read_data("initial_state.txt")
    state=np.copy(initial_state)
    state=state[0]
    lst=[]
    lst.append(state)
    f = open("output.txt", "w")
    while(True):
            f.write(f"Iteration:{len(lst)}\n")
            flag=False
            state=np.matmul(state,weights)
            f.write(np.array_str(state,max_line_width=np.inf)+'\n')
            for x in range(len(state)):
                state[x]=state[x].norm()
                for k,x in enumerate(initial_state[0]):
                    if not x.isNull():
                        state[k]=initial_state[0][k]
            f.write(np.array_str(state,max_line_width=np.inf)+'\n')
            for k in lst:
                if np.array_equal(k,state):
                    flag=True
                    f.write("Match state found\n")
                    f.write(np.array_str(state,max_line_width=np.inf)+'\n')
                    break
            if flag:
                if(np.array_equal(lst[-1],state)):
                    pass
                    f.write("Fixed Point Reached\n")
                else:
                    pass
                    f.write("Limit Cycle Achieved\n")
                break
            lst.append(state)
    f.close()