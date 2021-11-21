import gudhi
import numpy as np
import random
from  gtda.homology import VietorisRipsPersistence
def Association(simplex:np.ndarray,Label:np.ndarray):
    unique=np.unique(Label)
    if len(simplex)==1:
        ass=np.zeros(len(unique))
        if simplex[0]<len(Label):
            for i in range(len(unique)):
                if unique[i]==Label[simplex[0]]:
                    ass[i]=1
        return ass
    else:
        return np.sum([Association([v],Label) for v in simplex],axis=0)

def Link(point:np.ndarray,K:gudhi.SimplexTree):
    St=K.get_star(point)
    Lk=[]
    for simplex in St:
        if len(np.setdiff1d(simplex[0],point))!=0:
            Lk.append((np.setdiff1d(simplex[0],point),K.filtration(np.setdiff1d(simplex[0],point))))
    return Lk

def Extension(point:np.ndarray,K:gudhi.SimplexTree,Labels:np.ndarray):
    Lk=Link(point,K)
    return np.sum([Association(p[0],Labels) if p[1]==0 else Association(p[0],Labels)/p[1] for p in Lk],axis=0)

def Labeling(point:np.ndarray,K:gudhi.SimplexTree,Labels):
    A=Extension(point,K,Labels)
    a=np.max(A)
    index=(A==a).nonzero()[0]
    return np.unique(Labels)[index[random.randint(0,len(index)-1)]]

def GetPersistenceIntervalSet2(diagrams:np.ndarray,q:int):
    D=[]
    i=q
    while ((len(D)==0)or(len(D)==1 and D[0][0]==0 and D[0][1]==0)) and 1<= i:
        D=diagrams[0][(diagrams[0][:,2]==i).nonzero()[0]][:,0:2]
        i=i-1
    return D

def Predict3(Xtrain:np.ndarray,Xtest:np.ndarray, ytrain:np.ndarray,q:int,mode:str='R'):
    VR= VietorisRipsPersistence(homology_dimensions=list(range(q)),collapse_edges=True,n_jobs=4)
    data=np.append(Xtrain,Xtest,axis=0)
    diagrams = VR.fit_transform([data])
    intervals=GetPersistenceIntervalSet2(diagrams,q)
    if mode=='R':
        d=intervals[random.randint(0,len(intervals)-1)]
    elif mode == 'M':
        life=intervals[:,1]-intervals[:,0]
        d=intervals[(life==np.max(life)).nonzero()[0]][0]
    elif mode == 'A':
        life=intervals[:,1]-intervals[:,0]
        life=life-np.mean(life)
        d=intervals[(life==np.min(life)).nonzero()[0]][0]
    else:
        raise ValueError("Mode must be 'R,'M'or'A'")
    Subcomplex=gudhi.RipsComplex(points = np.append(Xtrain,Xtest,axis=0),max_edge_length=d[0]).create_simplex_tree(max_dimension=1) 
    n=Subcomplex.num_simplices()   
    Subcomplex.collapse_edges(np.ceil(q/3))
    while(n!=Subcomplex.num_simplices()):
        n=Subcomplex.num_simplices()   
        Subcomplex.collapse_edges(np.ceil(q/3))
    Subcomplex.expansion(q)
    return [Labeling([i+len(Xtrain)],Subcomplex,ytrain) for i in range(len(Xtest))]