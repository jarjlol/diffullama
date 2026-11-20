"""Prevent duplicate team lane/queue consumers. No GPU access or process signals."""
import argparse,json,os,pathlib,sys

def existing(kind,gpu,proc=pathlib.Path('/proc'),exclude=()):
 kinds={'lane.sh','queue.sh'} if kind=='lane' else {'queue.sh'}
 found=[]
 for directory in proc.iterdir():
  if not directory.name.isdigit() or int(directory.name) in exclude:continue
  try:
   if directory.stat().st_uid!=os.getuid():continue
   args=directory.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\x00').split('\x00')
  except (FileNotFoundError,PermissionError,ProcessLookupError):continue
  for index,arg in enumerate(args[:2]):
   if pathlib.Path(arg).name in kinds and len(args)>index+1 and args[index+1]==str(gpu):
    found.append({'pid':int(directory.name),'consumer':pathlib.Path(arg).name,'gpu':gpu});break
 return found

def main():
 ap=argparse.ArgumentParser();ap.add_argument('kind',choices=('lane','queue'));ap.add_argument('gpu',type=int,choices=(0,1));ap.add_argument('--exclude',type=int,required=True);a=ap.parse_args()
 matches=existing(a.kind,a.gpu,exclude=(a.exclude,os.getpid()))
 if matches:print(json.dumps({'duplicate_start_refused':matches}));return 3
 return 0
if __name__=='__main__':raise SystemExit(main())
