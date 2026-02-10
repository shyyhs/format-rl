# FormatRL: Structured Document Translation via Format Reinforcement Learning

[![Paper](https://img.shields.io/badge/IJCNLP--AACL%202025_Main-b31b1b.svg)](https://aclanthology.org/2025.ijcnlp-long.38/) [![Oral](https://img.shields.io/badge/Oral-ff6f00.svg)]()
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of the paper:

> **Structured Document Translation via Format Reinforcement Learning**
>
> [Haiyue Song](https://shyyhs.github.io/), Johannes Eschbach-Dymanus, Hour Kaing, Sumire Honda, Hideki Tanaka, Bianka Buschbeck, Masao Utiyama
>
> *National Institute of Information and Communications Technology (NICT), Japan & SAP SE, Germany*

## Overview

FormatRL addresses the challenge of translating structured documents (XML/HTML) where preserving document structure is as important as translation quality. Traditional approaches either operate at the sentence level or rely on detag-and-project pipelines that suffer from error propagation. FormatRL employs **Group Relative Policy Optimization (GRPO)** on top of a supervised fine-tuning model to directly optimize novel structure-aware rewards.

<p align="center">
  <img src="assets/pipeline.png" width="60%" alt="FormatRL Pipeline"/>
</p>

**Key contributions:**

- **FormatRL**, a two-stage framework (SFT → GRPO) for structured document translation
- **TreeSim**, a reward measuring structural similarity between predicted and reference XML trees via tree edit distance
- **Node-chrF**, a reward measuring translation quality at the XML node level
- Additional reward functions including Tag-F1, XML-Match, BLEU, and XML-BLEU
- Application of **StrucAUC**, a fine-grained metric distinguishing minor errors from major structural failures

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{song-etal-2025-structured,
    title = "Structured Document Translation via Format Reinforcement Learning",
    author = "Song, Haiyue  and
      Eschbach-Dymanus, Johannes  and
      Kaing, Hour  and
      Honda, Sumire  and
      Tanaka, Hideki  and
      Buschbeck, Bianka  and
      Utiyama, Masao",
    editor = "Inui, Kentaro  and
      Sakti, Sakriani  and
      Wang, Haofen  and
      Wong, Derek F.  and
      Bhattacharyya, Pushpak  and
      Banerjee, Biplab  and
      Ekbal, Asif  and
      Chakraborty, Tanmoy  and
      Singh, Dhirendra Pratap",
    booktitle = "Proceedings of the 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics",
    month = dec,
    year = "2025",
    address = "Mumbai, India",
    publisher = "The Asian Federation of Natural Language Processing and The Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.ijcnlp-long.38/",
    pages = "677--697",
    ISBN = "979-8-89176-298-5",
}
```
