#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path

from hgai_core import BASE_DIR, HGAIEngine, PRESETS, checkpoint_load, train_model
from build_dataset import expand_pairs, parse_pairs


def main():
    ap=argparse.ArgumentParser(description="HGAI v10 standalone practical hybrid model")
    ap.add_argument("--train-file",default=str(BASE_DIR/"train.txt"))
    ap.add_argument("--model-file",default=str(BASE_DIR/"hgai_model_v10.pth"))
    ap.add_argument("--preset",default="main",choices=sorted(PRESETS))
    ap.add_argument("--retrain",action="store_true",help="새 v10 체크포인트 학습")
    ap.add_argument("--resume",action="store_true",help="기존 v10 체크포인트에서 이어 학습")
    ap.add_argument("--steps",type=int,default=3000)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--seed",type=int,default=None,help="재현이 필요할 때만 고정. 기본은 매 실행 랜덤")
    ap.add_argument("--threads",type=int,default=0)
    ap.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    ap.add_argument("--once")
    ap.add_argument("--mode",default="hybrid",choices=["hybrid","neural","raw-neural","search"])
    ap.add_argument("--temperature",type=float,default=0.72)
    ap.add_argument("--top-k",type=int,default=40)
    ap.add_argument("--top-p",type=float,default=0.92)
    ap.add_argument("--max-new-tokens",type=int,default=96)
    ap.add_argument("--info",action="store_true")
    ap.add_argument("--benchmark",action="store_true")
    ap.add_argument("--eval",action="store_true")
    ap.add_argument("--export-dataset",type=str,default=None)
    args=ap.parse_args()

    train_file=Path(args.train_file); model_file=Path(args.model_file)
    if args.export_dataset:
        base=parse_pairs(train_file.read_text(encoding="utf-8")); pairs=expand_pairs(base,max_variants=12)
        Path(args.export_dataset).write_text("\n".join(f"{q}={a}" for q,a in pairs)+"\n",encoding="utf-8")
        print(f"exported {len(pairs):,} pairs -> {args.export_dataset}")
        return 0

    trained_now=False
    if args.retrain or args.resume:
        if args.seed is None: args.seed=1337
        _,_,_,report=train_model(train_file,model_file,preset=args.preset,steps=args.steps,lr=args.lr,seed=args.seed,device=args.device,resume=args.resume,threads=args.threads,status=print)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        trained_now=True

    engine=HGAIEngine(train_file,model_file,args.preset,args.device,args.seed,load_model=True)
    if trained_now and not any([args.info,args.benchmark,args.eval,args.once is not None]):
        print(json.dumps(engine.model_info(),ensure_ascii=False,indent=2)); return 0
    if args.info:
        info=engine.model_info(); ckpt=checkpoint_load(model_file)
        if ckpt:
            info.update({"checkpoint_step":ckpt.get("step"),"best_val":ckpt.get("best_val"),"last_val":ckpt.get("last_val")})
        print(json.dumps(info,ensure_ascii=False,indent=2)); return 0

    tests=[
        ("Casualties: Unknown 게임이 뭐야?",["하드코어 생존 게임","덥듀"]),
        ("캐주얼티즈 언노운은 어떤 게임이야?",["하드코어 생존 게임"]),
        ("TCP랑 UDP는 뭐가 달라?",["TCP","UDP"]),
        ("깃허브 쉽게 설명해줘",["Git"]),
        ("10000×10000",["100000000"]),
        ("너는 뭐야?",["HGAI"]),
    ]
    if args.eval or args.benchmark:
        ok=0
        for q,must in tests:
            a=engine.reply(q,mode=args.mode,temperature=args.temperature,top_k=args.top_k,top_p=args.top_p,max_new_tokens=args.max_new_tokens)
            passed=all(x.lower() in a.lower() for x in must); ok+=passed
            print(("OK" if passed else "FAIL"),"Q:",q,"\nA:",a,"\n")
        print(f"eval {ok}/{len(tests)}")
        return 0 if (not args.eval or ok==len(tests)) else 1

    if args.once is not None:
        print(engine.reply(args.once,mode=args.mode,temperature=args.temperature,top_k=args.top_k,top_p=args.top_p,max_new_tokens=args.max_new_tokens)); return 0

    print("HGAI v10 대화 시작! 종료: exit")
    print(json.dumps(engine.model_info(),ensure_ascii=False))
    while True:
        try: q=input("너: ").strip()
        except (EOFError,KeyboardInterrupt): print(); break
        if q.lower() in {"exit","quit","종료"}: break
        print("HGAI:",engine.reply(q,mode=args.mode,temperature=args.temperature,top_k=args.top_k,top_p=args.top_p,max_new_tokens=args.max_new_tokens))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
