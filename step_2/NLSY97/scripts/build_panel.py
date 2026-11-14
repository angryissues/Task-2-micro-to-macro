"""Interview-anchored NLSY97 panel with a lagged observed employment feature."""
import argparse,calendar,json,re,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import openpyxl

def state(v):
    v=np.asarray(v,float);return np.where((v>=9701)&(v<1000000),1.,np.where(np.isin(v,[2,4,5]),0.,np.nan))
def calendar_map(path):
    ws=openpyxl.load_workbook(path,read_only=True,data_only=True).active;rows=list(ws.values)[1:];out=[]
    for m,d,y,cy,cw,continuous,cm in rows:
        if m is None:continue
        out.append(dict(year=int(cy),week=int(cw),continuous=int(continuous),date=pd.Timestamp(int(y),int(m),int(d))))
    w=pd.DataFrame(out).sort_values('continuous');assert w.continuous.is_unique
    # The official sheet has an 8-day step at 2024-09-02. No outcomes after
    # 2023 enter this protocol; validate the entire period actually used.
    used=w[w.date.dt.year<=2023];assert (np.diff(used.date.values[1:]).astype('timedelta64[D]').astype(int)==7).all();return w
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',required=True);ap.add_argument('--crosswalk',required=True);ap.add_argument('--out',required=True);ap.add_argument('--split',choices=['train','development','confirmation'],required=True);a=ap.parse_args();root=Path(a.prepared);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(root/'selected_variables.csv',dtype=str);w=calendar_map(a.crosswalk);lookup={(r.year,r.week):int(r.continuous) for r in w.itertuples()};dates=dict(zip(w.continuous,w.date));q=meta.set_index('csv_name');weeks=meta[meta.family=='weekly_employment'].copy();weeks['index']=weeks.question.map(lambda s:lookup[tuple(map(int,re.search(r'EMP_STATUS_(\d+)\.(\d+)',s).groups()))]);weeks=weeks.sort_values('index');weekcols=weeks.csv_name.tolist();indices=weeks['index'].to_numpy(int);indexpos={int(k):i for i,k in enumerate(indices)}
    # Only round-specific educational variables, never the across-round CVC versions.
    rounds={}
    for year,g in meta[meta.survey_year.str.fullmatch(r'\d{4}')].groupby('survey_year'):
        names=dict(zip(g.question,g.csv_name));M=next((v for k,v in names.items() if k in ['CV_INTERVIEW_DATE_M','CV_INTERVIEW_DATE~M']),None);Y=next((v for k,v in names.items() if k in ['CV_INTERVIEW_DATE_Y','CV_INTERVIEW_DATE~Y']),None)
        if M and Y:
            degree=next((v for k,v in names.items() if k=='CV_HIGHEST_DEGREE_EVER_EDT'),names.get('CV_HIGHEST_DEGREE_EVER'));rounds[year]=(M,Y,degree)
    dem={row.question:row.csv_name for row in meta.itertuples() if row.family in ['respondent_id','demographics']};cols=list(dict.fromkeys(weekcols+list(dem.values())+[v for r in rounds.values() for v in r if v]));d=pd.read_csv(root/f'{a.split}_selected.csv.gz',usecols=cols);allrows=[];bad_dates=0
    for _,person in d.iterrows():
        pub=int(person[dem['PUBID']]);by=person[dem['KEY!BDATE_Y']];bm=person[dem['KEY!BDATE_M']];sex=person[dem['KEY!SEX']]
        if not (1980<=by<=1984 and 1<=bm<=12):continue
        values=person[weekcols].to_numpy(float);employment=state(values)
        for roundyear,(M,Y,E) in rounds.items():
            yy,mm=person[Y],person[M]
            if not(1980<=yy<=2024 and 1<=mm<=12):continue
            first=pd.Timestamp(int(yy),int(mm),1);end=first+pd.offsets.MonthEnd(0);past=w[w.date+pd.Timedelta(days=6)<first]
            if past.empty:continue
            lag=int(past.iloc[-1].continuous);nexts=w[w.date>end]
            if nexts.empty or lag not in indexpos:continue
            origin=int(nexts.iloc[0].continuous);origin_date=dates[origin];age=((origin_date.year-int(by))*12+origin_date.month-int(bm))/12
            if not 18<=age<45:continue
            current=employment[indexpos[lag]]
            if not np.isfinite(current):continue
            # Feature employment week ends before the interview MONTH starts.
            assert dates[lag]+pd.Timedelta(days=6)<first<=end<origin_date
            row=dict(idind=pub,split=a.split,round_year=int(roundyear),interview_month=int(mm),interview_year=int(yy),origin_week=origin,start_date=str(origin_date.date()),start_year=origin_date.year,age=age,sex=sex if sex in (1,2) else np.nan,education=person[E] if E and 0<=person[E]<=7 else np.nan,current_employment=current,lag_week=lag,lag_days=(origin_date-dates[lag]).days)
            complete=True
            for h in (1,2,3):
                target=origin+52*h;pos=indexpos.get(target);year=(dates[target]+pd.Timedelta(days=6)).year if target in dates else 9999
                row[f'target_year_{h}']=year;row[f'y{h}']=float(employment[pos]) if pos is not None else np.nan;row[f'raw_status_{h}']=float(values[pos]) if pos is not None else np.nan
            # Preserve paths for retrospective H4; no path outcomes enter predictors.
            path_indices=[origin+j*13 for j in range(13)];row['path_quarterly']=json.dumps([float(employment[indexpos[j]]) if j in indexpos and np.isfinite(employment[indexpos[j]]) else None for j in path_indices])
            allrows.append(row)
    panel=pd.DataFrame(allrows).sort_values(['idind','origin_week','round_year']).drop_duplicates(['idind','origin_week'],keep='last')
    lo,hi={'train':(1980,2013),'development':(2014,2017),'confirmation':(2018,2023)}[a.split]
    single=panel[panel.target_year_1.ge(lo)&panel.target_year_1.le(hi)].reset_index(drop=True)
    single.to_csv(out/f'{a.split}_single_horizon_panel.csv.gz',index=False)
    # A common three-horizon cohort keeps all future control dates in the same block.
    panel=panel[panel.target_year_1.ge(lo)&panel.target_year_3.le(hi)].reset_index(drop=True)
    assert not panel.duplicated(['idind','origin_week']).any()
    panel.to_csv(out/f'{a.split}_panel.csv.gz',index=False)
    manifest={'split':a.split,'people_in_extract':len(d),'eligible_windows':len(panel),'eligible_people':int(panel.idind.nunique()),'calendar_target_years':[lo,hi],'origin_year_min':int(panel.start_year.min()) if len(panel) else None,'origin_year_max':int(panel.start_year.max()) if len(panel) else None,'rows_selected_using_outcomes':False,'outcome_metrics_inspected':False,'features_employment_week_ends_before_interview_month':True,'education':'round-specific archival interview value; no CVC cross-round feature; not a historical vintage guarantee','crosswalk_sha256':hashlib.sha256(Path(a.crosswalk).read_bytes()).hexdigest()}
    if a.split!='confirmation':manifest.update({'known_h1':int(panel.y1.notna().sum()),'known_complete_path_h123':int(panel[['y1','y2','y3']].notna().all(axis=1).sum())})
    (out/f'{a.split}_panel_manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':main()
