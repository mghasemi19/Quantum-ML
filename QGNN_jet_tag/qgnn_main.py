"""Paper-inspired boosted-Z versus gluon jet QGNN.

Differentiable PyTorch statevector is used for joint training. Qiskit builds an
independent, equivalent circuit for numerical validation. This is NOT the
paper authors' code or an exact reproduction of unpublished settings.
"""
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

N = 10

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

def preprocess(jets, energy_col=3, eta_col=6, phi_col=7):
    """Input [samples, particles>=11, 16]; output [samples,10,160].

    Feature column mapping MUST be checked against the dataset documentation.
    Ten nearest neighbors exclude the central particle itself. Padded particles
    are NOT supported: remove them with a dataset-specific validity mask first.
    """
    a = np.asarray(jets, dtype=np.float32)
    if a.ndim != 3 or a.shape[1] < 11 or a.shape[2] != 16:
        raise ValueError('Expected [jets, >=11 particles, 16 features]')
    if not np.isfinite(a).all(): raise ValueError('NaN/inf in jets')
    result = np.empty((len(a), N, 160), np.float32)
    for b, jet in enumerate(a):
        lead = np.argsort(-jet[:, energy_col], kind='stable')[:N]
        eta = jet[:, eta_col]; phi = jet[:, phi_col]
        for j, i in enumerate(lead):
            de = eta - eta[i]; dp = (phi - phi[i] + np.pi) % (2*np.pi) - np.pi
            d2 = de*de + dp*dp; d2[i] = np.inf
            neighbors = np.argsort(d2, kind='stable')[:N]
            result[b,j] = np.abs(jet[i] - jet[neighbors]).reshape(-1)
    return result

class TrainOnlyMinMax:
    def fit(self,x):
        self.low = x.min(axis=(0,1), keepdims=True)
        self.high = x.max(axis=(0,1), keepdims=True)
        return self
    def transform(self,x):
        return np.clip((x-self.low)/np.maximum(self.high-self.low,1e-8),0,1).astype('float32')

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(160,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,4))
    def forward(self,x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,64),nn.ReLU(),nn.Linear(64,128),nn.ReLU(),nn.Linear(128,160))
    def forward(self,x): return self.net(x)

class NoiseGenerator(nn.Module):
    """Conditional latent prior: concatenates random noise with binary label."""
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(65,128),nn.ReLU(),nn.Linear(128,40))
    def forward(self,labels):
        z=torch.rand(len(labels),64,device=labels.device)
        return self.net(torch.cat((z,labels.float().reshape(-1,1)),dim=-1)).reshape(-1,10,4)

def sinkhorn_divergence(x,y,epsilon=0.2,n_iter=15):
    """Batch-level DEBIASED entropic OT (uniform weights), squared L2 cost.
    Small batches and epsilon produce a biased/noisy estimate; tune for science.
    """
    def ot(a,b):
        c=torch.cdist(a,b,p=2).square()
        log_k=-c/epsilon
        u=torch.zeros(len(a),device=a.device,dtype=a.dtype)
        v=torch.zeros(len(b),device=a.device,dtype=a.dtype)
        la=-np.log(len(a)); lb=-np.log(len(b))
        for _ in range(n_iter):
            u=la-torch.logsumexp(log_k+v[None,:],dim=1)
            v=lb-torch.logsumexp(log_k+u[:,None],dim=0)
        plan=torch.exp(log_k+u[:,None]+v[None,:])
        return (plan*c).sum()
    a=x.reshape(len(x),-1); b=y.reshape(len(y),-1)
    return ot(a,b)-0.5*ot(a,a)-0.5*ot(b,b)

def edges_from_latent(z, k=3, eps=1e-3):
    """Undirected union of directed kNN edges; differentiable inverse distance.

    Neighbor selection is discrete (no gradient through changing topology),
    but weights DO propagate gradients through selected latent distances.
    """
    with torch.no_grad():
        discrete=torch.cdist(z.detach(),z.detach())
        discrete.fill_diagonal_(float('inf'))
        neighbors=torch.topk(discrete,k,largest=False).indices
        pairs=sorted({tuple(sorted((i,int(j)))) for i in range(N) for j in neighbors[i]})
    distances=torch.cdist(z,z)
    return [(i,j,1./(distances[i,j]+eps)) for i,j in pairs]

I2=torch.eye(2,dtype=torch.complex64)
X=torch.tensor([[0,1],[1,0]],dtype=torch.complex64)
Y=torch.tensor([[0,-1j],[1j,0]],dtype=torch.complex64)
Z=torch.diag(torch.tensor([1.,-1.])).to(torch.complex64)

def rot(axis,angle):
    axis={'x':X,'y':Y,'z':Z}[axis].to(angle.device)
    return torch.cos(angle/2)*I2.to(angle.device)-1j*torch.sin(angle/2)*axis

def apply_one(state,gate,q):
    v=state.reshape([2]*N).movedim(q,-1)
    return (v@gate.T).movedim(-1,q).reshape(-1)

def apply_two(state,gate,q1,q2):
    v=state.reshape([2]*N).movedim((q1,q2),(-2,-1))
    other=[j for j in range(N) if j not in (q1,q2)]
    v=(v.reshape(-1,4)@gate.T).reshape([2]*N)
    return v.movedim((-2,-1),(q1,q2)).reshape(-1)

def controlled_z(angle):
    """CRZ: diag(1,1,exp(-it/2),exp(it/2)) in control-target order."""
    t=angle; ones=torch.ones((),device=t.device,dtype=torch.complex64)
    return torch.diag(torch.stack((ones,ones,torch.exp(-.5j*t),torch.exp(.5j*t))))

def interaction_x(angle):
    """exp(-it X_i X_j), exactly implemented by RXX(2*t)."""
    eye=torch.eye(4,device=angle.device,dtype=torch.complex64)
    xx=torch.kron(X.to(angle.device),X.to(angle.device))
    return torch.cos(angle)*eye-1j*torch.sin(angle)*xx

class JetQGNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Four data reuploads; one RZ/RY and one chain of CRZ per round.
        self.gamma=nn.Parameter(.1*torch.randn(4,N))
        self.beta=nn.Parameter(.1*torch.randn(4,N))
        self.chain=nn.Parameter(.1*torch.randn(4,N-1))
        self.time=nn.Parameter(torch.tensor(.05))
        self.kappa=nn.Parameter(.1*torch.randn(N))
    def forward_one(self,z):
        edges=edges_from_latent(z)
        state=torch.zeros(2**N,dtype=torch.complex64,device=z.device)
        state[0]=1
        angles=2*torch.pi*torch.sigmoid(z)
        for k in range(4):
            for q in range(N):
                state=apply_one(state,rot('x',angles[q,k]),q)
            for q in range(N):
                state=apply_one(state,rot('y',self.beta[k,q]),q)
                state=apply_one(state,rot('z',self.gamma[k,q]),q)
            for q in range(N-1):
                state=apply_two(state,controlled_z(self.chain[k,q]),q,q+1)
        # First-order Trotter product for H=sum_{undirected edges} w_ij X_i X_j.
        for i,j,w in edges:
            state=apply_two(state,interaction_x(self.time*w),i,j)
        for q in range(N): state=apply_one(state,rot('y',self.kappa[q]),q)
        probs=state.abs().square().reshape([2]*N)
        # average <Z_i>; signal is negative (score = -mean Z)
        return torch.stack([probs.sum(dim=tuple(j for j in range(N) if j!=q))[0]-probs.sum(dim=tuple(j for j in range(N) if j!=q))[1] for q in range(N)]).mean().real
    def forward(self,z): return torch.stack([self.forward_one(zi) for zi in z])

class Hybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder(); self.decoder=Decoder(); self.generator=NoiseGenerator(); self.quantum=JetQGNN()
    def scores(self,x): return -self.quantum(self.encoder(x))

def load_data(path,seed=42,n_samples=8000):
    """NPZ keys: jets [J,P,16], labels [J] (Z=1, gluon=0);
    alternatively x [J,10,160] already preprocessed + labels.
    """
    with np.load(path) as f:
        y=np.asarray(f['labels'],dtype=np.int64)
        if 'x' in f: x=np.asarray(f['x'],dtype=np.float32)
        else: x=preprocess(f['jets'])
    if x.shape!=(len(y),10,160) or not set(np.unique(y)).issubset({0,1}):
        raise ValueError('Expected x [J,10,160], binary labels 0/1')
    if n_samples < 8 or n_samples % 4 or len(y)<n_samples: raise ValueError('samples must be divisible by 4, >=8 and <= dataset size')
    if any(np.sum(y==c)<n_samples//2 for c in (0,1)): raise ValueError('Each class needs at least samples/2 events')
    indices=np.arange(len(y)); rng=np.random.default_rng(seed)
    chosen=np.concatenate([rng.choice(indices[y==c], n_samples//2,replace=False) for c in [0,1]])
    rng.shuffle(chosen)
    x=x[chosen];y=y[chosen]
    # Paper: 4000 train, 2000 validation, 2000 test for 8000 jets.
    train,rest=train_test_split(np.arange(len(y)),train_size=.5,stratify=y,random_state=seed)
    val,test=train_test_split(rest,test_size=.5,stratify=y[rest],random_state=seed)
    scaler=TrainOnlyMinMax().fit(x[train]);x=scaler.transform(x)
    return (torch.tensor(x[train]),torch.tensor(y[train])),(torch.tensor(x[val]),torch.tensor(y[val])),(torch.tensor(x[test]),torch.tensor(y[test])),scaler

def batches(x,y,batch_size,shuffle=True):
    order=torch.randperm(len(y)) if shuffle else torch.arange(len(y))
    for ids in order.split(batch_size): yield x[ids],y[ids]

def pretrain(model,train,epochs=75,lr=1e-3,batch_size=32,with_sinkhorn=True):
    opt=torch.optim.Adam(list(model.encoder.parameters())+list(model.decoder.parameters())+list(model.generator.parameters()),lr=lr)
    for epoch in range(epochs):
        losses=[]
        for x,y in batches(*train,batch_size):
            z=model.encoder(x); reconstruction=F.mse_loss(model.decoder(z),x)
            reg=.001*sinkhorn_divergence(z,model.generator(y)) if with_sinkhorn else z.new_zeros(())
            loss=reconstruction+reg;opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        print(f'pretrain {epoch+1}/{epochs}: {np.mean(losses):.5f}')

def train_quantum(model,train,epochs=5,lr=1e-3,batch_size=4,joint=False,with_sinkhorn=False):
    for p in model.encoder.parameters():p.requires_grad_(joint)
    params=list(model.quantum.parameters())
    if joint:params+=list(model.encoder.parameters())
    if joint and with_sinkhorn:params+=list(model.decoder.parameters())+list(model.generator.parameters())
    opt=torch.optim.Adam(params,lr=lr)
    for epoch in range(epochs):
        losses=[]
        for x,y in batches(*train,batch_size):
            z=model.encoder(x); score=-model.quantum(z)
            # score in [-1,1] -> P(Z) in [0,1]. BCE directly on score as logit
            loss=F.binary_cross_entropy_with_logits(score,y.float())
            if joint and with_sinkhorn:
                loss=loss+F.mse_loss(model.decoder(z),x)+.001*sinkhorn_divergence(z,model.generator(y))
            opt.zero_grad();loss.backward();opt.step();losses.append(loss.item())
        print(f'{"joint" if joint else "quantum"} epoch {epoch+1}/{epochs}: {np.mean(losses):.5f}')

def evaluate(model,split):
    x,y=split; model.eval()
    with torch.no_grad():
        scores=np.array([model.scores(xi[None]).item() for xi in x])
    truth=y.numpy();auc=float(roc_auc_score(truth,scores))
    fpr,tpr,thresholds=roc_curve(truth,scores)
    rejection={}
    for eff in (.25,.6,.8):
        # Interpolate FPR at target TPR, 1/FPR is background rejection.
        fp=float(np.interp(eff,tpr,fpr));rejection[str(eff)]=None if fp==0 else 1/fp
    return {'auc':auc,'background_rejection':rejection,'n':len(y)},(fpr,tpr)

def qiskit_circuit(model,z):
    """Reference circuit; compare to Torch statevector using Qiskit Statevector."""
    from qiskit import QuantumCircuit
    qc=QuantumCircuit(N)
    with torch.no_grad():
        z=z.detach();angles=2*np.pi*torch.sigmoid(z).numpy()
        for k in range(4):
            for q in range(N):qc.rx(float(angles[q,k]),q)
            for q in range(N):
                qc.ry(float(model.quantum.beta[k,q]),q)
                qc.rz(float(model.quantum.gamma[k,q]),q)
            for q in range(N-1):qc.crz(float(model.quantum.chain[k,q]),q,q+1)
        for i,j,w in edges_from_latent(z):qc.rxx(2*float(model.quantum.time)*float(w),i,j)
        for q in range(N):qc.ry(float(model.quantum.kappa[q]),q)
    return qc

def validate_qiskit(model,z,atol=1e-4):
    from qiskit.quantum_info import Statevector, SparsePauliOp
    qc=qiskit_circuit(model,z)
    sv=Statevector.from_instruction(qc)
    ev=np.mean([sv.expectation_value(SparsePauliOp.from_list([('I'*(N-1-q)+'Z'+'I'*q,1.)])).real for q in range(N)])
    with torch.no_grad(): torch_ev=model.quantum.forward_one(z).item()
    if not np.isclose(ev,torch_ev,atol=atol):raise AssertionError(f'Qiskit={ev}, torch={torch_ev}')
    return ev,torch_ev

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data',required=True,help='NPZ, keys (jets or x), labels')
    ap.add_argument('--output',default='results')
    ap.add_argument('--samples',type=int,default=8000)
    ap.add_argument('--pretrain-epochs',type=int,default=75)
    ap.add_argument('--quantum-epochs',type=int,default=5)
    ap.add_argument('--joint-epochs',type=int,default=5)
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--sinkhorn',action='store_true')
    args=ap.parse_args();seed_all(args.seed)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    train,val,test,scaler=load_data(args.data,args.seed,args.samples)
    model=Hybrid();pretrain(model,train,args.pretrain_epochs,batch_size=max(4,args.batch_size),with_sinkhorn=args.sinkhorn)
    train_quantum(model,train,args.quantum_epochs,batch_size=args.batch_size)
    val_separate,_=evaluate(model,val)
    torch.save(model.state_dict(),output/'separate.pt')
    train_quantum(model,train,args.joint_epochs,batch_size=args.batch_size,joint=True,with_sinkhorn=args.sinkhorn)
    val_joint,_=evaluate(model,val)
    # Test ONCE, after model selection. No test-set tuning.
    test_metrics,(fpr,tpr)=evaluate(model,test)
    np.savez(output/'scaler.npz',low=scaler.low,high=scaler.high)
    torch.save(model.state_dict(),output/'joint.pt')
    np.savez(output/'test_roc.npz',fpr=fpr,tpr=tpr)
    metrics={'separate_validation':val_separate,'joint_validation':val_joint,'joint_test':test_metrics,'seed':args.seed,'sinkhorn':args.sinkhorn,'note':'single seed, paper-inspired implementation, not five-fold reproduction'}
    (output/'metrics.json').write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
