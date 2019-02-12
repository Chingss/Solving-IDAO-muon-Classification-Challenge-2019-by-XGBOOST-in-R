# Solving-IDAO-muon-Classification-Challenge-2019-by-XGBOOST-in-R
Objective: The task is to build a classiﬁer that would distinguish muons from non-muons in the LHCb detector.

Input format
Dataset fields description:
label, integer in {0,1} — you need to predict it. 0 is background (pions and protons), 1 is signal (muons).
particle_type, integer in {0,1,2} — type of the particle. 0 — pion, 1 — muon, 2 — proton. Available only for the training dataset.
weight, float — example weight, used in both training and evaluation. Product of sWeight and kinWeight.
sWeight, float — a component of the example weight that accounts for uncertainty in labeling.
kinWeight, float > 0 — a component of the example weight that equalizes kinematic observables between signal and background.
id, integer — this is an identiﬁcation field of the example.
Lextra_{X,Y}[N], float — coordinates of the track extrapolation intersection with N-th station.
Mextra_{X,Y,Z}2[N], float — multiple scattering uncertainty for squared {X, Y, Z} coordinate of the track extrapolation.
MatchedHit_{X,Y,Z}[N], float — coordinates of the hit in the N-th station that a physics-based tracking algorithm associated with the track. See the poster about the algorithm.
MatchedHit_TYPE[N], categorical in {1, 2} — whether the matched hit is crossed. 1 means uncrossed, 2 means crossed. See pages 6-8 here.
MatchedHit_T[N], integer in [1, 20] — timing of the matched hit.
MatchedHit_D{X,Y,Z}[N], float > 0 — uncertainty of the matched hit coordinates.
MatchedHit_DT[N], integer — delta time for the matched hit in the N-th station.
FOI_hits_N, integer — number of hits inside a physics-deﬁned cone around the track (aka Field Of Interest, FOI).
FOI_hits_{,D}{X,Y,Z,T}, array of float of size FOI_hits_N — same as MatchedHit{,D}{X,Y,Z,T}, per hit.
FOI_hits_S, integer in {0, 1, 2, 3} — stations of the FOI hits.
ncl[N], integer — number of clusters in the N-th station (high-level variable).
avg_cs[N], float ≥ 1 — average cluster size in the N-th station (high-level variable).
ndof, integer in {4, 6, 8} — number of degrees of freedom used in 
χ
2
 computation, a function of momentum.
NShared, integer ≥ 0 — number of closest hits shared with the neighbouring tracks. See pages 10-11 here.
P, float ≥ 3000 — momentum modulo, MeV/c.
PT, float ≥ 800 — component of the momentum transverse (i.e. perpendicular) to the beam line, MeV/c.
-9999 is used for missing values.

The data can be found in the shared Yandex.Disk folder.
And The Link For the Same Is :  https://yadi.sk/d/pdwdp4Lt5X4DMQ
