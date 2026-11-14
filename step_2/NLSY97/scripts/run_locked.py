"""Sequential confirmation access with code/config integrity checks and append-only log."""
import argparse,datetime,hashlib,json,subprocess,sys
from pathlib import Path
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',required=True);ap.add_argument('--predictions',required=True);ap.add_argument('--config',required=True);ap.add_argument('--lock',required=True);ap.add_argument('--out',required=True);ap.add_argument('--local',required=True);a=ap.parse_args();repo=Path(a.repo);code=Path(__file__).parent;out=Path(a.out);out.mkdir(parents=True,exist_ok=True);local=Path(a.local);local.mkdir(parents=True,exist_ok=True);lock=json.loads(Path(a.lock).read_text());config=Path(a.config)
    def verify():
        assert hashlib.sha256(config.read_bytes()).hexdigest()==lock['experiment_config_sha256']
        for file,digest in lock['code_sha256'].items():assert hashlib.sha256((code/file).read_bytes()).hexdigest()==digest,file
        for file,digest in lock['reference_code_sha256'].items():assert hashlib.sha256((repo/file).read_bytes()).hexdigest()==digest,file
    def event(h,status):
        with (out/'confirmation_access.jsonl').open('a') as f:f.write(json.dumps({'hypothesis':h,'status':status,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lock_sha256':hashlib.sha256(Path(a.lock).read_bytes()).hexdigest()})+'\n')
    common=['--repo',str(repo),'--predictions',a.predictions,'--config',str(config),'--out',str(out),'--split','confirmation']
    for h in ['H1','H2','H3','H4','H5','H6']:
        verify();event(h,'started');print('CONFIRMATION',h,flush=True)
        try:
            if h in ['H1','H2','H3']:cmd=[sys.executable,str(code/'replicate.py'),*common,'--hypotheses',h]
            elif h in ['H4','H5']:cmd=[sys.executable,str(code/'paths_groups.py'),*common,'--hypotheses',h]
            else:
                cmd=[sys.executable,str(code/'transfer.py'),'--repo',str(repo),'--predictions',a.predictions,'--config',str(config),'--local',str(local),'--out',str(out/'H6')]
            subprocess.run(cmd,check=True)
            if h=='H6':subprocess.run([sys.executable,str(repo/'step_2/H6-RLMS/scripts/complete.py'),'--input',str(local/'H6_input.csv'),'--config',str(local/'H6_config.json'),'--repo',str(repo),'--out',str(out/'H6'),'--reps','100'],check=True)
            event(h,'completed')
        except Exception:
            event(h,'failed');raise
if __name__=='__main__':main()
