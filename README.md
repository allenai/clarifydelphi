# ClarifyDelphi
This repository is for data and code accompanying the paper:

ClarifyDelphi: Reinforced Clarification Questions with Defeasibility Rewards for Social and Moral Situations

Valentina Pyatkin, Jena D. Hwang, Vivek Srikumar, Ximing Lu, Liwei Jiang, Yejin Choi, Chandra Bhagavatula

ACL 2023

# Data Format
In delta-Clarify we provide the crowdsourced clarification questions.

    0. **id** Enumeration of the instances.
    1. **source** Whether the questions have been crowdsourced or come from a LLM.
    2. **situation** The social or moral situation.
    3. **question** The clarification question.

In delta-Clarify-silver we provide the davinci-002 generated questions, given the defeasible SocialChemistry data.

    0. **DataSource** Source of the data.
    1. **Hypothesis** The social or moral situation together with a judgment.
    2. **Update** A weakening or strengthening update.
    3. **UpdateType** Whether the update weakens or strengthens the hypothesis.
    4. **question_davinci** The question generated by GPT3.
    5. **situation** The social or moral situation without the judgment (automatically removed).

# Demo
We provide a [demo](https://clarify-delphi.apps.allenai.org/) of our clarification question generation system.

# Citing the Data and/or Code
- Resources on this page are licensed CC-BY 4.0, a Creative Commons license requiring Attribution (https://creativecommons.org/licenses/by/4.0/).
- Please cite the following paper if you use the data: 
```
@inproceedings{pyatkin2023clarifydelphi,
  title={clarifydelphi: Reinforced Clarification Questions with Defeasibility Rewards for Social and Moral Situations},
  author={Pyatkin, Valentina and 
    Hwang, Jena D. and
    Srikumar, Vivek and
    Lu, Ximing and
    Jiang, Liwei and
    Choi, Yejin and
    Bhagavatula, Chandra
    },
  booktitle={Proceedings of the Association for Computational Linguistics: ACL 2023},
  address = "Toronto",
  publisher = "Association for Computational Linguistics",
  year={2023}
}
```

If you use the data, please also be sure to cite all of the original datasets on which we built our dataset.

Forbes et al., 2020. [Social Chemistry 101: Learning to Reason about Social and Moral Norms](https://aclanthology.org/2020.emnlp-main.48/)
Rudinger et al., 2020. [Thinking Like a Skeptic: Defeasible Inference in Natural Language](https://aclanthology.org/2020.findings-emnlp.418/)

