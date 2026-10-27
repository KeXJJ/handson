"""Download a pinned HandsOn task subset or the official pi05 base checkpoint."""
import argparse
import base64
from collections import Counter
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import os

from paths import ROOT, DATASET_DIR, REPO_ROOT

DATASET = 'Cyan666/HandsOn-Datasets'
REVISION = '7ad38776d91e4e13c02dad4e6ad892e8b8a1d187'
SOURCE_TASK = 'b03_peg_in_socket'
METADATA = ['meta/info.json','meta/tasks.jsonl','meta/episodes.jsonl','meta/splits.json',
            'meta/handson_policy_schema.json','source_manifest.jsonl','FILE_MANIFEST.jsonl',
            'language/prompt_paraphrases.json','LICENSE']


def request_bytes(url, *, headers=None, expected_bytes=None):
    for attempt in range(6):
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'handson-pi05-portable/1.0',**(headers or {})})
            with urllib.request.urlopen(request,timeout=60) as response:
                if headers and 'Range' in headers:
                    if response.status != 206:
                        raise RuntimeError('Server did not honor byte range')
                content=response.read() if expected_bytes is None else response.read(expected_bytes+1)
            if expected_bytes is not None and len(content)!=expected_bytes:
                raise RuntimeError(f'Incomplete download: {len(content)} / {expected_bytes}')
            return content
        except (urllib.error.URLError,TimeoutError,RuntimeError) as error:
            if attempt==5:raise
            delay=min(60,2**attempt)
            if isinstance(error,urllib.error.HTTPError) and error.code==429:
                try:delay=max(delay,float(error.headers.get('Retry-After','30')))
                except ValueError:delay=max(delay,30)
            time.sleep(min(delay,300))


def atomic_bytes(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    part=path.with_name(path.name+'.tmp')
    part.write_bytes(data)
    part.replace(path)


def file_sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while block:=stream.read(4*1024*1024):digest.update(block)
    return digest.hexdigest()


def download_dataset(args):
    endpoint=os.environ.get('HF_ENDPOINT','https://huggingface.co').rstrip('/')
    metadata=ROOT/'metadata'
    for name in METADATA:
        destination=metadata/name
        if not destination.exists():
            url=f'{endpoint}/datasets/{DATASET}/resolve/{REVISION}/{name}'
            atomic_bytes(destination,request_bytes(url))
    manifest={row['path']:row for line in (metadata/'FILE_MANIFEST.jsonl').open() if (row:=json.loads(line))}
    for name in METADATA:
        if name in manifest and file_sha256(metadata/name)!=manifest[name]['sha256']:
            raise RuntimeError(f'Metadata checksum mismatch: {name}; remove cached file and retry')
    rows=[json.loads(line) for line in (metadata/'source_manifest.jsonl').open()]
    candidates=[row for row in rows if row['source_task']==SOURCE_TASK]
    selected=[]
    for split,count in [('train',args.train_episodes),('validation',args.validation_episodes)]:
        split_rows=[row for row in candidates if row['split']==split]
        if not args.all_task and count>len(split_rows):
            raise ValueError(f'Only {len(split_rows)} {split} episodes available')
        selected.extend(split_rows if args.all_task else split_rows[:count])
    paths=[]
    for row in selected:
        paths.extend([row['data_path'],*row['video_paths'].values()])
    entries=[manifest[name] for name in paths]
    summary={'repo_id':DATASET,'revision':REVISION,'task':'t09_insert_peg','source_task':SOURCE_TASK,
             'local_directory':str(DATASET_DIR),'metadata_cache_directory':str(metadata),
             'upstream_url':f'{endpoint}/datasets/{DATASET}/tree/{REVISION}',
             'episodes':len(selected),'split_episodes':dict(Counter(row['split'] for row in selected)),
             'split_frames':{s:sum(row['frames'] for row in selected if row['split']==s) for s in ('train','validation')},
             'files':len(entries),'bytes':sum(row['bytes'] for row in entries),
             'complete_for_selected_episodes':False,'selection':'full task' if args.all_task else 'small task subset',
             'episode_indices':{s:[row['episode_index'] for row in selected if row['split']==s] for s in ('train','validation')}}
    print(json.dumps(summary,indent=2),flush=True)
    if args.plan_only:return
    DATASET_DIR.mkdir(parents=True,exist_ok=True)
    receipt=DATASET_DIR/'subset_receipt.json'
    # Mark the current selection incomplete before updating any reused output.
    receipt.write_text(json.dumps(summary,indent=2)+'\n')
    for name in METADATA:
        if name in ('source_manifest.jsonl','FILE_MANIFEST.jsonl','meta/episodes.jsonl'):continue
        destination=DATASET_DIR/name
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.resolve()!=(metadata/name).resolve():shutil.copy2(metadata/name,destination)
    (DATASET_DIR/'source_manifest.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in selected))
    (DATASET_DIR/'FILE_MANIFEST.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in entries))
    ids={row['episode_index'] for row in selected}
    (DATASET_DIR/'meta/episodes.jsonl').write_text(''.join(line for line in (metadata/'meta/episodes.jsonl').open() if json.loads(line)['episode_index'] in ids))

    def fetch(entry):
        destination=DATASET_DIR/entry['path']
        if destination.exists() and destination.stat().st_size==entry['bytes'] and file_sha256(destination)==entry['sha256']:return
        content=request_bytes(f"{endpoint}/datasets/{DATASET}/resolve/{REVISION}/{entry['path']}?download=true",expected_bytes=entry['bytes'])
        if hashlib.sha256(content).hexdigest()!=entry['sha256']:raise RuntimeError(f"SHA256 mismatch: {entry['path']}")
        atomic_bytes(destination,content)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count,_ in enumerate(pool.map(fetch,entries),1):
            if count%10==0:print(f'Verified {count}/{len(entries)} files',flush=True)
    summary.update(complete_for_selected_episodes=True,sha256_verified_files=len(entries))
    receipt.write_text(json.dumps(summary,indent=2)+'\n')
    print('DATASET_COMPLETE',flush=True)


def download_base(args):
    import google_crc32c
    objects=json.loads((REPO_ROOT/'BASE_CHECKPOINT.json').read_text())['objects']
    if not objects:raise RuntimeError('Official bucket listing was empty')
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'base_objects.json').write_text(json.dumps(objects,indent=2)+'\n')
    directory=ROOT/'models/pi05_base'
    plans=[]
    tasks=[]
    chunk=args.chunk_mb*1024*1024
    for obj in objects:
        destination=directory/obj['name'].removeprefix('checkpoints/pi05_base/')
        size=int(obj['size'])
        if destination.exists() and destination.stat().st_size==size:continue
        destination.parent.mkdir(parents=True,exist_ok=True)
        if size==0:
            destination.touch()
            continue
        partial=destination.with_name(destination.name+'.part')
        prefix=partial.stat().st_size if partial.exists() else 0
        if prefix>size:raise RuntimeError(f'Oversized partial file: {partial}')
        segments=[]
        for start in range(prefix,size,chunk):
            stop=min(size,start+chunk)-1
            segment=partial.with_name(partial.name+'.ranges')/str(start)
            segments.append(segment)
            tasks.append((obj['name'],start,stop,segment))
        plans.append((destination,partial,segments,size))

    generations={obj['name']:obj['generation'] for obj in objects}

    def fetch_range(task):
        name,start,stop,destination=task
        expected=stop-start+1
        if destination.exists() and destination.stat().st_size==expected:return
        content=request_bytes('https://storage.googleapis.com/openpi-assets/'+urllib.parse.quote(name)+'?generation='+generations[name],
                              headers={'Range':f'bytes={start}-{stop}'},expected_bytes=expected)
        atomic_bytes(destination,content)

    print(f'Official base: {sum(int(obj["size"]) for obj in objects)} bytes; {len(tasks)} pending ranges',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for count,_ in enumerate(pool.map(fetch_range,tasks),1):
            if count%100==0:print(f'Ranges ready: {count}/{len(tasks)}',flush=True)
    for destination,partial,segments,size in plans:
        with partial.open('ab') as stream:
            for segment in segments:
                with segment.open('rb') as source:shutil.copyfileobj(source,stream)
                segment.unlink()
        if partial.stat().st_size!=size:raise RuntimeError('Incorrect assembled size')
        partial.replace(destination)
    for obj in objects:
        destination=directory/obj['name'].removeprefix('checkpoints/pi05_base/')
        crc=google_crc32c.Checksum()
        md5=hashlib.md5()
        with destination.open('rb') as stream:
            while block:=stream.read(8*1024*1024):
                crc.update(block)
                md5.update(block)
        valid=base64.b64encode(crc.digest()).decode()==obj['crc32c']
        if 'md5Hash' in obj:valid=valid and base64.b64encode(md5.digest()).decode()==obj['md5Hash']
        if not valid:
            destination.replace(destination.with_name(destination.name+'.corrupt'))
            raise RuntimeError(f"Base checksum mismatch: {obj['name']}; rerun to download again")
    receipt={'source':'gs://openpi-assets/checkpoints/pi05_base','files':len(objects),
             'bytes':sum(int(obj['size']) for obj in objects),'verification':'GCS CRC32C for all objects; MD5 where supplied',
             'checkpoint_type':'official pretrained base, not HandsOn-finetuned'}
    (ROOT/'base_download_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('BASE_COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='kind',required=True)
    dataset=sub.add_parser('dataset')
    dataset.add_argument('--train-episodes',type=int,default=8)
    dataset.add_argument('--validation-episodes',type=int,default=2)
    dataset.add_argument('--all-task',action='store_true')
    dataset.add_argument('--plan-only',action='store_true')
    dataset.add_argument('--workers',type=int,default=3)
    base=sub.add_parser('base')
    base.add_argument('--workers',type=int,default=16)
    base.add_argument('--chunk-mb',type=int,default=2)
    args=parser.parse_args()
    if args.workers<=0:parser.error('--workers must be positive')
    if args.kind=='dataset':
        if args.train_episodes<=0 or args.validation_episodes<=0:parser.error('Episode counts must be positive')
        download_dataset(args)
    else:
        if args.chunk_mb<=0:parser.error('--chunk-mb must be positive')
        download_base(args)


if __name__=='__main__':main()
