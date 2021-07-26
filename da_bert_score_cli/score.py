#!/usr/bin/env python
import os
import argparse
import numpy as np
import pickle
import sys
import ipdb
import pandas as pd
import torch

import da_bert_score

def ori_bert_score_output(args, all_preds, hash_code):
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    msg = hash_code + f" P: {P:.6f} R: {R:.6f} F1: {F1:.6f}"
    print(msg)
    if args.seg_level:
        ps, rs, fs = all_preds
        for p, r, f in zip(ps, rs, fs):
            print("{:.6f}\t{:.6f}\t{:.6f}".format(p, r, f))

def main():
    torch.multiprocessing.set_sharing_strategy("file_system")

    parser = argparse.ArgumentParser("Calculate BERTScore w/o Difficulty")
    parser.add_argument(
        "--lang",
        type=str,
        default=None,
        help='two-letter abbreviation of the language (e.g., en) or "en-sci" for scientific text',
    )
    parser.add_argument(
        "-m", "--model", default=None, help="BERT model name (default: bert-base-uncased) or path to a pretrain model"
    )
    parser.add_argument("-l", "--num_layers", type=int, default=None, help="use first N layer in BERT (default: 8)")
    parser.add_argument("-b", "--batch_size", type=int, default=64, help="batch size (default: 64)")
    parser.add_argument("--nthreads", type=int, default=4, help="number of cpu workers (default: 4)")
    parser.add_argument("--idf", action="store_true", help="BERT Score with IDF scaling")
    parser.add_argument(
        "--rescale_with_baseline", action="store_true", help="Rescaling the numerical score with precomputed baselines"
    )
    parser.add_argument("--baseline_path", default=None, type=str, help="path of custom baseline csv file")
    parser.add_argument("-s", "--seg_level", action="store_true", help="show individual score of each pair")
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument("-r", "--ref", type=str, nargs="+", required=True, help="reference file path(s) or a string")
    parser.add_argument(
        "-c", "--cand", type=str, help="candidate (system outputs) file path or a string"
    )

    # DA-BERTScore parameters
    parser.add_argument("--with_diff", action="store_true", help="BERT Score with difficulty weighting", required=True)
    parser.add_argument("--cand_list", type=str, nargs="+", help="Multiple candidate files")

    parser.add_argument('--softmax_norm',default=False, action="store_true")
    parser.add_argument('--max_norm',default=False, action="store_true")
    parser.add_argument('--range_one',default=False, action="store_true")
    parser.add_argument('--ref_diff',default=False, action="store_true")
    parser.add_argument("--save_path", default=None, type=str, help="Path for saving the results", required=True)

    args = parser.parse_args()

    if args.with_diff:
        cands_list = []
        for cand_file in args.cand_list:
            with open(cand_file) as f:
                cands = [line.strip() for line in f]
            cands_list.append(cands)

        refs = []
        for ref_file in args.ref:
            assert os.path.exists(ref_file), f"reference file {ref_file} doesn't exist"
            with open(ref_file) as f:
                curr_ref = [line.strip() for line in f]
                assert len(curr_ref) == len(cands), f"# of sentences in {ref_file} doesn't match the # of candidates"
                refs.append(curr_ref)
        refs = list(zip(*refs))


        # diff_dict = bert_score.score()
        outs, sent_diff_dict = da_bert_score.compute_diff(
            cands_list,
            refs,
            model_type=args.model,
            num_layers=args.num_layers,
            verbose=args.verbose,
            idf=args.idf,
            batch_size=args.batch_size,
            lang=args.lang,
            return_hash=True,
            rescale_with_baseline=True,
            baseline_path=args.baseline_path,
        )
        # Recalculate scores with BERT
        all_P, all_R, all_F, weight_P, weight_R, weight_F = da_bert_score.bert_score_with_diff(
            sent_diff_dict,
            softmax_norm = args.softmax_norm, 
            max_norm = args.max_norm, 
            range_one = args.range_one, 
            ref_diff = args.ref_diff
        )
        ori_P = []
        ori_R = []
        ori_F = []
        for sys_no, sys_name in enumerate(args.cand_list):
            print('System: {}'.format(sys_name))
            all_pred, hash_code = outs[sys_no]
            # ori_bert_score_output(args, all_pred, hash_code)
            avg_scores = [s.mean(dim=0) for s in all_pred]
            ori_P.append(avg_scores[0].cpu().item())
            ori_R.append(avg_scores[1].cpu().item())
            ori_F.append(avg_scores[2].cpu().item())
        # Export
        hypo_names = ['.'.join(i.split('/')[-1].split('.')[1:-1]) for i in args.cand_list]
        testset = [i.split('/')[-1].split('.')[0] for i in args.cand_list]
        language_pair = [i.split('/')[-1].split('.')[-1] for i in args.cand_list]
        res_df = pd.DataFrame({
            "SYSTEM": hypo_names,
            'DA_BERT_P': weight_P,
            'DA_BERT_R': weight_R,
            'DA_BERT_F': weight_F,
            'BERT_P': ori_P,
            'BERT_R': ori_R,
            'BERT_F': ori_F,
        })
        res_df.to_csv(os.path.join(args.save_path, 'results.csv'))
        print("Save results to file: ", os.path.join(args.save_path, 'results.csv'))
        # print(res_df)
    sys.exit(0)

    if os.path.isfile(args.cand):
        with open(args.cand) as f:
            cands = [line.strip() for line in f]

        refs = []
        for ref_file in args.ref:
            assert os.path.exists(ref_file), f"reference file {ref_file} doesn't exist"
            with open(ref_file) as f:
                curr_ref = [line.strip() for line in f]
                assert len(curr_ref) == len(cands), f"# of sentences in {ref_file} doesn't match the # of candidates"
                refs.append(curr_ref)
        refs = list(zip(*refs))
    elif os.path.isfile(args.ref[0]):
        assert os.path.exists(args.cand), f"candidate file {args.cand} doesn't exist"
    else:
        cands = [args.cand]
        refs = [args.ref]
        assert not args.idf, "do not support idf mode for a single pair of sentences"

    all_preds, hash_code = da_bert_score.score(
        cands,
        refs,
        model_type=args.model,
        num_layers=args.num_layers,
        verbose=args.verbose,
        idf=args.idf,
        batch_size=args.batch_size,
        lang=args.lang,
        return_hash=True,
        rescale_with_baseline=args.rescale_with_baseline,
        baseline_path=args.baseline_path,
    )
    avg_scores = [s.mean(dim=0) for s in all_preds]
    P = avg_scores[0].cpu().item()
    R = avg_scores[1].cpu().item()
    F1 = avg_scores[2].cpu().item()
    msg = hash_code + f" P: {P:.6f} R: {R:.6f} F1: {F1:.6f}"
    print(msg)
    if args.seg_level:
        ps, rs, fs = all_preds
        for p, r, f in zip(ps, rs, fs):
            print("{:.6f}\t{:.6f}\t{:.6f}".format(p, r, f))


if __name__ == "__main__":
    main()
