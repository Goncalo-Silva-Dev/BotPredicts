import json, numpy as np, collections
G=json.load(open('games.json'))
NS=92  # 0..44 H1 min 1..45, 45 H1 stoppage minute(s), 46..90 H2 min 46..90, 91 H2 stoppage
def L1_of(g):
    s=[e[3] for e in g['ev'] if e[1]==1 and e[3]>0]; return max(s+[2])
def slot_seq(g):
    seq=[(1,m,0,m-1) for m in range(1,46)]+[(1,45,99,45)]
    seq+=[(2,m,0,m) for m in range(46,91)]+[(2,90,99,91)]
    return seq
def key(e): return (e[1], e[2], 99 if e[3]>0 else 0)
def clipd(d): return max(-2,min(2,d))
rows=[]
for gi,g in enumerate(G):
    gh=ga=rh=ra=0
    evs=sorted(g['ev'],key=key); k=0
    for (h,m,s,idx) in slot_seq(g):
        yh=ya=0; nrh=nra=0
        while k<len(evs) and key(evs[k])<=(h,m,s):
            e=evs[k]; k+=1
            if e[0]=='g':
                if e[4]=='h': yh+=1
                else: ya+=1
            else:
                if e[4]=='h': nrh+=1
                else: nra+=1
        rows.append((gi,idx,0,g['lam'][0],clipd(gh-ga),max(-1,min(1,ra-rh)),yh))
        rows.append((gi,idx,1,g['lam'][1],clipd(ga-gh),max(-1,min(1,rh-ra)),ya))
        gh+=yh; ga+=ya; rh+=nrh; ra+=nra
R=np.array(rows,dtype=float)
gid=R[:,0].astype(int); idx=R[:,1].astype(int); side=R[:,2].astype(int); lam=R[:,3]; d=R[:,4].astype(int)+2; rd=R[:,5].astype(int)+1; y=R[:,6]
test=np.array([G[i]['date']>='2025-07-01' for i in range(len(G))])
tr=~test[gid]
def fit(mask, use_state=True, use_red=True, side_specific=False):
    rho=np.full(NS,1/96.); gm=np.ones((2,5)); rm=np.ones(3)
    for it in range(30):
        base=lam*rho[idx]; ex=base*gm[side if side_specific else 0*side,d]*rm[rd]
        # rho
        num=np.bincount(idx[mask],y[mask],NS); den=np.bincount(idx[mask],(ex/rho[idx])[mask],NS); rho=num/den
        # smooth regular minutes (moving average 5) separately per half
        for a,b in ((0,45),(46,91)):
            r=rho[a:b].copy(); sm=np.convolve(np.pad(r,2,mode='edge'),np.ones(5)/5,'valid'); rho[a:b]=sm
        ex=lam*rho[idx]*gm[side if side_specific else 0*side,d]*rm[rd]
        if use_state:
            for sd in ((0,1) if side_specific else (0,)):
                mm=mask&((side==sd) if side_specific else True)
                num=np.bincount(d[mm],y[mm],5); den=np.bincount(d[mm],(ex/gm[sd if side_specific else 0,d])[mm],5)
                gm[sd if side_specific else 0]=num/den
                if not side_specific: gm[1]=gm[0]
            ex=lam*rho[idx]*gm[side if side_specific else 0*side,d]*rm[rd]
        if use_red:
            num=np.bincount(rd[mask],y[mask],3); den=np.bincount(rd[mask],(ex/rm[rd])[mask],3); rm=num/den; rm=rm/rm[1]
    return rho,gm,rm
if __name__=="__main__":
    rho,gm,rm=fit(tr)
    print("rho sum per game (should ~1 incl. expected stoppage):", rho[:45].sum(), rho[46:91].sum(), "stoppage rates", rho[45], rho[91])
    print("state mult (d=-2..2):",np.round(gm[0],3)); print("red mult (down,even,up):",np.round(rm,3))
    rho2,gm2,rm2=fit(tr,side_specific=True); print("side specific home",np.round(gm2[0],3),"away",np.round(gm2[1],3))
    # stability: estimate on test too
    r3,g3,m3=fit(test[gid]); print("test-set estimates",np.round(g3[0],3),np.round(m3,3))
    json.dump({"rho":rho.tolist(),"gm":gm[0].tolist(),"rm":rm.tolist()},open("params.json","w"))
    # stoppage length distributions (train)
    L1=collections.Counter(L1_of(g) for g in G if g['date']<'2025-07-01'); L2=collections.Counter(g['L2'] for g in G if g['date']<'2025-07-01')
    print(sorted(L1.items())); print(sorted(L2.items()))
