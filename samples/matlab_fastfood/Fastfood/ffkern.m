function Kappro = ffkern( U,V,para,sgm )
phi1 = FastfoodForKernel(U', para, sgm, false);
phi2 = FastfoodForKernel(V', para, sgm, false);
Kappro = phi1'*phi2;