"""Import downloaded v8 evaluation history into the existing local report."""
import argparse,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-dir',type=Path,required=True);parser.add_argument('--report-data',type=Path,default=ROOT/'docs/report/report-data.json');args=parser.parse_args()
    history=json.loads((args.run_dir/'landing_evaluation/boundary_history.json').read_text())
    if not history['evaluations']:parser.error('No completed v8 evaluation')
    checkpoint_metrics={}
    for name in ['best','latest']:
        path=args.run_dir/'models'/f'{name}.model'
        if path.exists():
            with zipfile.ZipFile(path) as z:checkpoint_metrics[name]=json.loads(z.read('checkpoint_metrics.json'))
    result=dict(source=str(args.run_dir),latest=max(history['evaluations'],key=lambda x:x['step']),
                first_pass=history['first_pass'],evaluations=history['evaluations'],checkpoints=checkpoint_metrics)
    p=args.report_data;data=json.loads(p.read_text());data['v8_results']=result;p.write_text(json.dumps(data,ensure_ascii=False,indent=2))
    print('Imported',len(history['evaluations']),'evaluations, last step',result['latest']['step'])
if __name__=='__main__':main()
