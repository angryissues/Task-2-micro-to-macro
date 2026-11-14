"""Metadata-driven NLSY97 extraction; no outcome summaries or label-based selection."""
import argparse,csv,datetime,gzip,hashlib,json,re,sys
from pathlib import Path
csv.field_size_limit(sys.maxsize)
def canonical(s):return s.strip().replace('.','')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--protocol',required=True);ap.add_argument('--metadata-only',action='store_true');a=ap.parse_args();source=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    sdf=source/'nlsy97_all_1997-2023.sdf';columns=[];qstart=None
    with sdf.open(encoding='utf-8-sig') as f:
        for line in f:
            if 'Variable Description' in line and 'Question Name' in line:qstart=line.index('Question Name');continue
            if not re.match(r'^[A-Z]\d{5}\.\d{2}\s',line):continue
            ref=line[:9];year=line[11:15].strip();question=line[qstart:].strip();description=line[17:qstart].strip();family=None
            if question=='PUBID':family='respondent_id'
            elif question in ['KEY!SEX','KEY!BDATE_M','KEY!BDATE_Y']:family='demographics'
            elif question.startswith('EMP_STATUS_'):family='weekly_employment'
            elif question.startswith('EMP_'):family='employment_auxiliary_and_quality'
            elif question.startswith(('CV_HGC_EVER','CV_HIGHEST_DEGREE_EVER')):family='education_at_interview'
            elif question.startswith(('CV_INTERVIEW','CV_AGE','CV_INTV','CV_YR_NON_INTERVIEW')):family='interview_timing_and_response'
            if family:columns.append(dict(reference=ref,csv_name=canonical(ref),survey_year=year,question=question,family=family,description=description))
    assert sum(c['family']=='respondent_id' for c in columns)==1;assert len({c['csv_name'] for c in columns})==len(columns)
    with (out/'selected_variables.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(columns[0]));w.writeheader();w.writerows(columns)
    counts={g:sum(c['family']==g for c in columns) for g in sorted({c['family'] for c in columns})};weeks=[re.search(r'EMP_STATUS_(\d{4})\.(\d+)',c['question']).groups() for c in columns if c['family']=='weekly_employment'];summary={'selected_columns':len(columns),'families':counts,'weekly_year_min':min(int(y) for y,w in weeks),'weekly_year_max':max(int(y) for y,w in weeks),'protocol_sha256':hashlib.sha256(Path(a.protocol).read_bytes()).hexdigest(),'holdout_performance_inspected':False,'holdout_outcome_frequencies_inspected':False};print(json.dumps(summary,indent=2),flush=True)
    if a.metadata_only:(out/'metadata_summary.json').write_text(json.dumps(summary,indent=2));return
    src=source/'nlsy97_all_1997-2023.csv';wanted={c['csv_name'] for c in columns};idname=next(c['csv_name'] for c in columns if c['family']=='respondent_id');files={g:gzip.open(out/f'{g}_selected.csv.gz','wt',newline='') for g in ['train','development','confirmation']};writers={g:csv.writer(f) for g,f in files.items()};totals={g:0 for g in files};seen=set();n=0
    try:
      with src.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.reader(f);header=next(reader);index=[i for i,c in enumerate(header) if canonical(c) in wanted];names=[canonical(header[i]) for i in index];assert set(names)==wanted;idpos=header.index(idname) if idname in header else next(i for i,c in enumerate(header) if canonical(c)==idname)
        for w in writers.values():w.writerow(names)
        for row in reader:
            assert len(row)==len(header);person=str(int(float(row[idpos])));assert person not in seen;seen.add(person);bucket=int(hashlib.sha256(('NLSY97-replication-v1:'+person).encode()).hexdigest()[:8],16)%10;group='train' if bucket<6 else ('development' if bucket<8 else 'confirmation');writers[group].writerow([row[i] for i in index]);totals[group]+=1;n+=1
            if n%1000==0:print('extracted respondents',n,flush=True)
    finally:
      for f in files.values():f.close()
    summary.update({'respondents':n,'source_columns':len(header),'person_counts':totals,'split_person_disjoint':len(seen)==n,'extraction_rule':'all rows, metadata-selected columns, no label-dependent filtering','protocol_fixed_before_person_values':True,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()});(out/'extraction_manifest.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
