# Reweighting

Code for paper: [Towards Less Generic Responses in Neural Conversation Models:
A Statistical Re-weighting Method](https://www.aclweb.org/anthology/D18-1297), EMNLP 2018 (short, oral). [[paper]](./paper/Reweighting-EMNLP-2018.pdf) | [[code]](https://github.com/yhlleo/Reweighting) | [[dataset]](https://drive.google.com/file/d/1r8B2psuncz2hoSvz4dIXXeOL7opItvN9/view?usp=sharing) | [[presentation]](https://vimeo.com/305942945)

## Usage

Run the command:

```
python reweighting.py \
  --src_file /path/to/source_input \
  --tgt_file /path/to/target_input \
  --src_output /path/to/source_output \
  --tgt_output /path/to/source_output \
  --wt_output /path/to/weight_output
```

## Dataset Format

We have provided two paired samples in the `examples` folder, including a [`source.train`](./examples/source.train) and [`target.train`](./examples/target.train).

## Dataset Download

**Please note that this dialogue corpus is RESTRICTED to non-commercial research and educational purposes**.

## Reference

If you use this research/dataset, please cite our paper:

```
@inproceedings{liu2018towards,
  title={Towards Less Generic Responses in Neural Conversation Models: A Statistical Re-weighting Method},
  author={Liu, Yahui and Bi, Wei and Gao, Jun and Liu, Xiaojiang and Yao, Jian and Shi, Shuming},
  booktitle={Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing},
  pages={2769--2774},
  year={2018}
}
```
