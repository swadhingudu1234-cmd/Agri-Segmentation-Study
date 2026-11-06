"""Generate the synthetic 350,000-row Agri Segmentation Study dataset.
Run: pip install numpy pandas && python generate_dataset.py
"""
try:
    import numpy as np  # pyright: ignore[reportMissingImports]
    import pandas as pd  # pyright: ignore[reportMissingImports, reportMissingModuleSource]
except ImportError as exc:
    raise SystemExit("Missing dependencies. Install them with: pip install numpy pandas") from exc
SEED, N, NF = 42, 350_000, 25_000
rng=np.random.default_rng(SEED)
districts=np.array(["Khordha","Puri","Cuttack","Jagatsinghpur","Kendrapara","Nayagarh"])
crops=np.array(["Paddy","Maize","Groundnut","Green Gram","Black Gram","Vegetables"])
mandis={"Khordha":["Bhubaneswar Mandi","Jatni Mandi"],"Puri":["Pipili Mandi","Nimapara Mandi","Puri Mandi"],"Cuttack":["Choudwar Mandi","Nischintakoili Mandi","Cuttack Mandi"],"Jagatsinghpur":["Kujang Mandi","Tirtol Mandi"],"Kendrapara":["Pattamundai Mandi","Kendrapara Mandi"],"Nayagarh":["Ranpur Mandi","Nayagarh Mandi"]}
d=rng.choice(districts,N,p=[.17,.19,.18,.15,.16,.15]); c=rng.choice(crops,N,p=[.42,.13,.12,.12,.10,.11])
fids=np.array([f"F{i:06d}" for i in range(1,NF+1)]); w=rng.pareto(1.6,NF)+.25; w/=w.sum()
farmer=fids[rng.choice(NF,N,p=w)]
start,end=np.datetime64("2023-04-01"),np.datetime64("2026-03-31")
days=int((end-start).astype(int))+1; dt=pd.to_datetime(start+rng.integers(0,days,N).astype("timedelta64[D]")); m=dt.month.to_numpy()
season=np.where(np.isin(m,[6,7,8,9]),"Kharif",np.where(np.isin(m,[11,12,1,2,3]),"Rabi","Summer"))
by=pd.Series(c).map({"Paddy":2.8,"Maize":2.6,"Groundnut":1.5,"Green Gram":.75,"Black Gram":.7,"Vegetables":8.5}).to_numpy()
dfac=pd.Series(d).map({"Khordha":1.02,"Puri":1.06,"Cuttack":1.04,"Jagatsinghpur":1.08,"Kendrapara":.98,"Nayagarh":.94}).to_numpy()
y=np.maximum(.15,rng.normal(by*dfac,by*.22)); oi=rng.choice(N,int(N*.004),replace=False); y[oi]*=rng.choice([.25,2.8,3.5],len(oi),p=[.25,.6,.15])
area=np.clip(rng.lognormal(.15,.65,N),.1,12); qty=np.clip(area*y*1000*rng.uniform(.55,1.05,N),20,25000)
bp=pd.Series(c).map({"Paddy":22.5,"Maize":20.,"Groundnut":55.,"Green Gram":78.,"Black Gram":72.,"Vegetables":24.}).to_numpy()
p_kg=np.maximum(5,rng.normal(bp*(1+.08*np.sin(2*np.pi*(m-1)/12)),bp*.12))
unit=rng.choice(["INR_PER_KG","INR_PER_QUINTAL"],N,p=[.72,.28]); pv=np.where(unit=="INR_PER_KG",p_kg,p_kg*100)
rm=pd.Series(m).map({1:12,2:18,3:22,4:35,5:75,6:190,7:320,8:340,9:230,10:120,11:28,12:10}).to_numpy()
rf=pd.Series(d).map({"Khordha":1.,"Puri":1.08,"Cuttack":1.03,"Jagatsinghpur":1.12,"Kendrapara":1.15,"Nayagarh":.92}).to_numpy()
rain=np.maximum(0,rng.normal(rm*rf,35)); rain[rng.random(N)<.085]=np.nan
ph=np.clip(rng.normal(6.25,.75,N),4,8.8); sn=np.clip(rng.normal(235,55,N),70,420); sp=np.clip(rng.normal(22,8,N),3,60); sk=np.clip(rng.normal(185,48,N),45,360)
for a,r in [(ph,.035),(sn,.04),(sp,.045),(sk,.04)]: a[rng.random(N)<r]=np.nan
tx=np.array([f"T{i:07d}" for i in range(1,N+1)]); sid=tx.copy(); ix=np.where(rng.random(N)<.012)[0]; sid[ix]=sid[np.maximum(0,ix-1)]
mandi=np.array([rng.choice(mandis[x]) for x in d])
frame=pd.DataFrame({"transaction_id":tx,"source_record_id":sid,"farmer_id":farmer,"district":d,"mandi_name":mandi,
"arrival_date":dt.strftime("%Y-%m-%d"),"season":season,"crop":c,
"crop_grade":rng.choice(["FAQ","Grade A","Grade B","Unspecified"],N,p=[.48,.22,.20,.10]),
"land_area_ha":np.round(area,3),"yield_tonnes_per_ha":np.round(y,3),"arrival_quantity_kg":np.round(qty,2),
"price_value":np.round(pv,2),"price_unit":unit,"rainfall_mm":np.round(rain,2),"soil_ph":np.round(ph,2),
"soil_n_kg_ha":np.round(sn,1),"soil_p_kg_ha":np.round(sp,1),"soil_k_kg_ha":np.round(sk,1),
"irrigation_type":rng.choice(["Canal","Borewell","Rainfed","Mixed"],N,p=[.18,.31,.34,.17]),
"estimated_input_cost_inr":np.round(np.maximum(500,area*rng.normal(14500,2600,N)),2),
"estimated_gross_value_inr":np.round(qty*p_kg,2)})
frame.to_csv("agri_mandi_records_350k.csv",index=False)
print(f"Created {len(frame):,} rows and {len(frame.columns)} columns.")
