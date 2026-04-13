"""
Nystrom GP Experiments -- uses src/ library classes throughout.

Three experiments:
  1. Approximation Quality vs M (landmark count)
  2. Scalability Benchmark
  3. Hierarchical Multi-Subject Spatial Smoothing

All use the src/ library (NystromGP, FullGP, HierarchicalNystromGP).
"""

import os, sys, time, json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.kernels import rbf_kernel
from src.full_gp import FullGP
from src.nystrom_gp import NystromGP
from src.landmarks import select_landmarks
from src.hierarchical_gp import HierarchicalNystromGP

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':11,
    'axes.titlesize':12,'figure.dpi':300,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.spines.top':False,'axes.spines.right':False})
C = {'full':'#2c3e50','nys':'#2ecc71','random':'#e74c3c','kmeans':'#3498db','grid':'#e67e22','ref':'#95a5a6'}


def experiment1(fig_dir):
    print("="*60+"\n  Exp 1: Approximation Quality vs M\n"+"="*60)
    rng = np.random.default_rng(42)
    gs = 48; N = gs**2
    coords = np.array([(i/gs,j/gs) for i in range(gs) for j in range(gs)])
    ls,var,noise = 0.15,1.0,0.1
    K = rbf_kernel(coords,coords,ls,var)
    f = rng.multivariate_normal(np.zeros(N), K+1e-6*np.eye(N))
    y = f + rng.normal(0,np.sqrt(noise),N)
    idx = rng.permutation(N); nt = int(0.8*N)
    Xtr,ytr = coords[idx[:nt]], y[idx[:nt]]
    Xte,fte = coords[idx[nt:]], f[idx[nt:]]
    print(f"  Grid {gs}x{gs}={N}, Train {nt}, Test {len(fte)}")

    t0=time.time()
    full=FullGP(ls,var,noise); full.fit(Xtr,ytr)
    mu_f,var_f = full.predict(Xte)
    tf=time.time()-t0
    rmse_f = np.sqrt(np.mean((mu_f-fte)**2))
    cov_f = np.mean((fte>=mu_f-1.96*np.sqrt(var_f))&(fte<=mu_f+1.96*np.sqrt(var_f)))
    print(f"  Full GP: RMSE={rmse_f:.4f}, Cov={cov_f:.3f}, Time={tf:.2f}s")

    Ms = [25,50,100,200,400]; methods = ['random','kmeans','grid']
    R = {}
    for m in methods:
        R[m] = {k:[] for k in ['M','rmse','coverage','time','frobenius']}
        for M in Ms:
            if M>nt: continue
            t0=time.time()
            nys=NystromGP(M=M,landmark_method=m,lengthscale=ls,variance=var,noise_var=noise)
            nys.fit(Xtr,ytr); mu,v=nys.predict(Xte)
            tn=time.time()-t0
            rmse=np.sqrt(np.mean((mu-fte)**2))
            cov=np.mean((fte>=mu-1.96*np.sqrt(v))&(fte<=mu+1.96*np.sqrt(v)))
            sub=rng.choice(nt,min(500,nt),replace=False)
            Ks=rbf_kernel(Xtr[sub],Xtr[sub],ls,var)
            Z=nys.get_landmarks(); Cs=rbf_kernel(Xtr[sub],Z,ls,var)
            Ws=rbf_kernel(Z,Z,ls,var)+1e-6*np.eye(len(Z))
            Kn=Cs@np.linalg.solve(Ws,Cs.T)
            frob=np.linalg.norm(Ks-Kn,'fro')/np.linalg.norm(Ks,'fro')
            for k,val in zip(['M','rmse','coverage','time','frobenius'],[M,rmse,cov,tn,frob]):
                R[m][k].append(val)
            print(f"    {m:>8} M={M:4d}: RMSE={rmse:.4f} Cov={cov:.3f} Frob={frob:.4f} T={tn:.2f}s")

    fig,ax=plt.subplots(2,2,figsize=(11,9))
    for m,col in [('random',C['random']),('kmeans',C['kmeans']),('grid',C['grid'])]:
        r=R[m]
        ax[0,0].plot(r['M'],r['rmse'],'o-',color=col,lw=2,ms=6,label=m.capitalize())
        ax[0,1].plot(r['M'],r['frobenius'],'o-',color=col,lw=2,ms=6,label=m.capitalize())
        ax[1,0].plot(r['M'],r['coverage'],'o-',color=col,lw=2,ms=6,label=m.capitalize())
    ax[0,0].axhline(rmse_f,color=C['full'],ls='--',lw=1.5,label='Full GP')
    ax[0,0].set_xlabel('M');ax[0,0].set_ylabel('RMSE');ax[0,0].set_title('(a) RMSE');ax[0,0].legend(fontsize=8)
    ax[0,1].set_xlabel('M');ax[0,1].set_ylabel('Frobenius error');ax[0,1].set_title('(b) Kernel Error');ax[0,1].set_yscale('log');ax[0,1].legend(fontsize=8)
    ax[1,0].axhline(cov_f,color=C['full'],ls='--',lw=1.5);ax[1,0].axhline(0.95,color=C['ref'],ls=':')
    ax[1,0].set_xlabel('M');ax[1,0].set_ylabel('Coverage');ax[1,0].set_title('(c) Calibration');ax[1,0].legend(fontsize=8)
    nys100=NystromGP(M=100,lengthscale=ls,variance=var,noise_var=noise);nys100.fit(Xtr,ytr)
    mfg=full.predict(coords,return_var=False);mng=nys100.predict(coords,return_var=False)
    diff=np.abs(mfg-mng).reshape(gs,gs)
    im=ax[1,1].imshow(diff,cmap='hot',origin='lower');plt.colorbar(im,ax=ax[1,1])
    ax[1,1].set_title('(d) |Full-Nystrom| M=100')
    fig.tight_layout();fig.savefig(os.path.join(fig_dir,'nys_fig1_approx_quality.png'));plt.close()

    fig,ax=plt.subplots(1,4,figsize=(16,3.5))
    for a,d,t in [(ax[0],f.reshape(gs,gs),'True'),(ax[1],y.reshape(gs,gs),'Noisy'),
                   (ax[2],mfg.reshape(gs,gs),'Full GP'),(ax[3],mng.reshape(gs,gs),'Nystrom M=100')]:
        im=a.imshow(d,cmap='viridis',origin='lower');a.set_title(t,fontsize=10);plt.colorbar(im,ax=a,fraction=0.046)
    fig.tight_layout();fig.savefig(os.path.join(fig_dir,'nys_fig2_spatial_maps.png'));plt.close()
    print("  Saved figures")
    return {'full':{'rmse':float(rmse_f),'cov':float(cov_f)},'nystrom':R}


def experiment2(fig_dir):
    print("\n"+"="*60+"\n  Exp 2: Scalability\n"+"="*60)
    rng=np.random.default_rng(2024)
    gss=[16,32,48,64,80,100]; M0=100; ls,var,noise=0.15,1.0,0.1
    ft,nt,Ns=[],[],[]
    for gs in gss:
        N=gs*gs;Ns.append(N)
        coords=np.array([(i/gs,j/gs) for i in range(gs) for j in range(gs)])
        y=rng.normal(0,1,N)
        if N<=5000:
            t0=time.time();gp=FullGP(ls,var,noise);gp.fit(coords,y);_=gp.predict(coords[:10]);ft.append(time.time()-t0)
        else:
            ft.append(ft[-1]*(N/Ns[-2])**3)
        t0=time.time();nys=NystromGP(M=min(M0,N-1),lengthscale=ls,variance=var,noise_var=noise)
        nys.fit(coords,y);_=nys.predict(coords[:10]);nt.append(time.time()-t0)
        tag="actual" if N<=5000 else "extrap"
        print(f"  N={N:6d}: Full={ft[-1]:8.3f}s ({tag}), Nys={nt[-1]:.3f}s, {ft[-1]/max(nt[-1],0.001):.0f}x")
    fig,ax=plt.subplots(figsize=(7,5))
    ax.plot(Ns,ft,'o-',color=C['full'],lw=2,ms=7,label='Full GP')
    ax.plot(Ns,nt,'s-',color=C['nys'],lw=2,ms=7,label=f'Nystrom M={M0}')
    ax.set_xlabel('N');ax.set_ylabel('Time (s)');ax.set_yscale('log');ax.set_xscale('log')
    ax.legend();ax.grid(True,alpha=0.3);fig.tight_layout()
    fig.savefig(os.path.join(fig_dir,'nys_fig3_scaling.png'));plt.close()
    print("  Saved figure")
    return {'Ns':Ns,'full':ft,'nys':nt}


def experiment3(fig_dir):
    print("\n"+"="*60+"\n  Exp 3: Hierarchical Multi-Subject\n"+"="*60)
    rng=np.random.default_rng(42);gs=32;N=gs**2;J=10
    coords=np.array([(i/gs,j/gs) for i in range(gs) for j in range(gs)])
    ls,var,noise=0.15,1.0,0.3
    K=rbf_kernel(coords,coords,ls,var)+1e-6*np.eye(N)
    fpop=rng.multivariate_normal(np.zeros(N),K)
    offsets=rng.normal(0,0.5,J)
    nobs=np.array([20,30,40,50,80,120,200,400,600,800])
    subjects=[]
    for j in range(J):
        fj=fpop+offsets[j];oi=rng.choice(N,nobs[j],replace=False)
        yj=fj[oi]+rng.normal(0,np.sqrt(noise),nobs[j])
        subjects.append({'f_true':fj,'obs_idx':oi,'y':yj,'n':nobs[j]})
    print(f"  {J} subjects, obs: {list(nobs)}")

    naive=[np.sqrt(np.mean((np.full(N,np.mean(s['y']))-s['f_true'])**2)) for s in subjects]
    ir,ic=[],[]
    for j in range(J):
        nys=NystromGP(M=80,lengthscale=ls,variance=var,noise_var=noise)
        nys.fit(coords[subjects[j]['obs_idx']],subjects[j]['y'])
        mu,v=nys.predict(coords)
        ir.append(np.sqrt(np.mean((mu-subjects[j]['f_true'])**2)))
        ic.append(np.mean((subjects[j]['f_true']>=mu-1.96*np.sqrt(v))&(subjects[j]['f_true']<=mu+1.96*np.sqrt(v))))

    hier=HierarchicalNystromGP(M=80,lengthscale=ls,variance=var,noise_var=noise)
    hr=hier.fit_predict(coords,subjects)
    hrmse=[np.sqrt(np.mean((hr[j]['mu_hier']-subjects[j]['f_true'])**2)) for j in range(J)]
    hcov=[np.mean((subjects[j]['f_true']>=hr[j]['mu_hier']-1.96*np.sqrt(hr[j]['var_hier']))&
                   (subjects[j]['f_true']<=hr[j]['mu_hier']+1.96*np.sqrt(hr[j]['var_hier']))) for j in range(J)]

    print(f"\n  {'S':>2} {'n':>5} {'Naive':>7} {'Indep':>7} {'Hier':>7} {'Imp':>6} {'CovI':>5} {'CovH':>5}")
    for j in range(J):
        imp=(ir[j]-hrmse[j])/ir[j]*100
        print(f"  {j:2d} {nobs[j]:5d} {naive[j]:7.4f} {ir[j]:7.4f} {hrmse[j]:7.4f} {imp:5.1f}% {ic[j]:.3f} {hcov[j]:.3f}")
    ni=sum(1 for j in range(J) if hrmse[j]<ir[j])
    mi=np.mean([(ir[j]-hrmse[j])/ir[j]*100 for j in range(J)])
    print(f"\n  {ni}/{J} improved, mean: {mi:.1f}%")

    fig,ax=plt.subplots(2,2,figsize=(11,9));x=np.arange(J);w=0.25
    ax[0,0].bar(x-w,naive,w,color=C['ref'],alpha=0.7,label='Naive')
    ax[0,0].bar(x,ir,w,color=C['random'],alpha=0.8,label='Independent')
    ax[0,0].bar(x+w,hrmse,w,color=C['nys'],alpha=0.8,label='Hierarchical')
    ax[0,0].set_xticks(x);ax[0,0].set_xticklabels([f'n={n}' for n in nobs],fontsize=7,rotation=45)
    ax[0,0].set_ylabel('RMSE');ax[0,0].set_title('(a) RMSE');ax[0,0].legend(fontsize=7)
    ax[0,1].scatter(nobs,ir,s=70,color=C['random'],label='Indep',zorder=3)
    ax[0,1].scatter(nobs,hrmse,s=70,color=C['nys'],marker='s',label='Hier',zorder=3)
    for j in range(J):
        cl=C['nys'] if hrmse[j]<ir[j] else C['random']
        ax[0,1].annotate('',xy=(nobs[j],hrmse[j]),xytext=(nobs[j],ir[j]),arrowprops=dict(arrowstyle='->',color=cl,lw=1.2,alpha=0.6))
    ax[0,1].set_xlabel('n');ax[0,1].set_ylabel('RMSE');ax[0,1].set_title('(b) Shrinkage');ax[0,1].legend(fontsize=8);ax[0,1].set_xscale('log')
    ax[1,0].scatter(nobs,ic,s=70,color=C['random'],label='Indep')
    ax[1,0].scatter(nobs,hcov,s=70,color=C['nys'],marker='s',label='Hier')
    ax[1,0].axhline(0.95,color=C['ref'],ls='--');ax[1,0].set_xlabel('n');ax[1,0].set_ylabel('Coverage')
    ax[1,0].set_title('(c) Calibration');ax[1,0].legend(fontsize=8);ax[1,0].set_xscale('log')
    j0=0;nys0=NystromGP(M=80,lengthscale=ls,variance=var,noise_var=noise)
    nys0.fit(coords[subjects[j0]['obs_idx']],subjects[j0]['y']);mu0,_=nys0.predict(coords)
    ei=np.abs(mu0-subjects[j0]['f_true']).reshape(gs,gs)
    eh=np.abs(hr[j0]['mu_hier']-subjects[j0]['f_true']).reshape(gs,gs)
    cb=np.hstack([ei,np.full((gs,2),np.nan),eh])
    im=ax[1,1].imshow(cb,cmap='hot',origin='lower',vmin=0,vmax=max(ei.max(),eh.max()))
    ax[1,1].axvline(gs+0.5,color='white',lw=2);ax[1,1].set_title(f'(d) Error n={nobs[j0]}')
    plt.colorbar(im,ax=ax[1,1],fraction=0.046)
    fig.tight_layout();fig.savefig(os.path.join(fig_dir,'nys_fig4_hierarchical.png'));plt.close()
    print("  Saved figures")
    return {'n_improved':ni,'mean_imp':mi}


def main():
    fd=os.path.join(os.path.dirname(__file__),'..','figures');os.makedirs(fd,exist_ok=True)
    print("\n"+chr(9608)*60+"\n  NYSTROM GP EXPERIMENTS (3 experiments)\n"+chr(9608)*60)
    t=time.time();r={}
    r['exp1']=experiment1(fd);r['exp2']=experiment2(fd);r['exp3']=experiment3(fd)
    with open(os.path.join(fd,'nystrom_results.json'),'w') as f: json.dump(r,f,indent=2,default=str)
    print(f"\n  Total: {time.time()-t:.0f}s\n"+chr(9608)*60)

if __name__=='__main__': main()
