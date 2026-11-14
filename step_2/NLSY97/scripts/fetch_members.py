"""Fetch selected official ZIP members using validated byte ranges; verify CRC."""
import argparse,concurrent.futures,json,struct,time,urllib.request,zlib
from pathlib import Path
URL='https://nlsinfo.org/cohort-data/nlsy97_all_1997-2023.zip';TOTAL=475462343
def fetch(start,end):
    last=None
    for retry in range(4):
      try:
        req=urllib.request.Request(URL,headers={'Range':f'bytes={start}-{end}','User-Agent':'ResearchDataDownload/1.0'})
        with urllib.request.urlopen(req,timeout=90) as r:
            if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/'):raise ValueError('Server did not honor exact byte range')
            data=r.read()
            if len(data)!=end-start+1:raise ValueError('Incomplete byte range')
            return data
      except Exception as e:last=e;time.sleep(min(retry+1,3))
    raise last
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tail',required=True);ap.add_argument('--members',nargs='+',required=True);ap.add_argument('--out',required=True);ap.add_argument('--workers',type=int,default=8);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    b=Path(a.tail).read_bytes();i=b.find(b'PK\x01\x02');entries=[]
    while i>=0 and b[i:i+4]==b'PK\x01\x02':
        h=struct.unpack_from('<4s6H3I5H2I',b,i);name=b[i+46:i+46+h[10]].decode();entries.append({'name':name,'compressed':h[8],'crc32':h[7],'offset':h[16],'method':h[4]});i+=46+h[10]+h[11]+h[12]
    (out/'archive_members.json').write_text(json.dumps(entries,indent=2))
    for e in entries:
        if e['name'].rsplit('.',1)[-1] not in a.members:continue
        name=e['name'];header=fetch(e['offset'],e['offset']+511);hl=struct.unpack_from('<4s5H3I2H',header);data_start=e['offset']+30+hl[-2]+hl[-1];chunks=out/(name+'.compressed_parts');chunks.mkdir(exist_ok=True);size=512*1024;ranges=[(j,data_start+j*size,min(data_start+e['compressed']-1,data_start+(j+1)*size-1)) for j in range((e['compressed']+size-1)//size)]
        def job(item):
            j,s,t=item;p=chunks/f'{j:06d}.part'
            if p.exists() and p.stat().st_size==t-s+1:return j
            data=fetch(s,t);p.write_bytes(data);return j
        print('fetch',name,e['compressed'],'bytes',len(ranges),'chunks',flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as ex:
            for done,j in enumerate(ex.map(job,ranges),1):
                if done%10==0 or done==len(ranges):print(name,done,'/',len(ranges),flush=True)
        target=out/name;tmp=out/(name+'.extracting');crc=0;count=0;decomp=zlib.decompressobj(-15)
        with tmp.open('wb') as f:
            for j,_,_ in ranges:
                data=decomp.decompress((chunks/f'{j:06d}.part').read_bytes());f.write(data);crc=zlib.crc32(data,crc);count+=len(data)
            data=decomp.flush();f.write(data);crc=zlib.crc32(data,crc);count+=len(data)
        if not decomp.eof or crc!=e['crc32']:raise ValueError('CRC or stream verification failed')
        tmp.replace(target);print('VERIFIED',target.name,count,'bytes',flush=True)
if __name__=='__main__':main()
