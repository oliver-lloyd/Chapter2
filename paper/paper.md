# Comparing Link Prediction Approaches for Polypharmacy Side Effect Modelling

## Abstract

## Introduction

## Methods

Raw data, processed in [cite decagon], was downloaded from the Stanford Network Analysis Project (SNAP) [cite snap]. The task then was to prepare the data for LibKGE [cite libkge], which reads graphs as a list of edges in triple (head, relation, tail) format. Multiple different approaches could be taken to perform this conversion. In our pilot work we focused on three such methods, which we have named ‘selfloops’, ‘non-naïve’, and ‘multidrug’ – the difference between them being the way in which mono-/polypharmacy side effect data is structured in the resulting graph. The selfloops approach treats side effects as edges, either between pairs of drug nodes for polypharmacy, or from one drug back to itself (hence the name) in the monopharmacy case. The non-naïve construction method is similar, but monopharmacy data is instead modelled as n-hot node feature-vectors, with ‘hot’ columns indicating which side effects are associated with a given drug. Dimensionality reduction via principal component analysis (PCA) was performed on these features to create a smaller matrix for each possible dimensional size of embeddings. Then LibKGE was modified to load these vectors from disk, using them as the starting point for learning node embeddings for this dataset rather than any of its usual stochastic initialisation techniques. Finally, the multidrug approach was inspired by work in [cite kim and shin 23] and involves the creation of drug-pair nodes which occupy the same embedding space as regular drug nodes. This way, side effects can be modelled as nodes rather than edges, allowing simple ‘monopharmacy side effect’ edges from drugs to side effects and ‘polypharmacy side effect’ from drug-pair nodes to side effects. In order to have LibKGE recognise the association between multidrugs and their constituent drugs, two ‘multidrug contains’ edges were created between every drug-pair node and their corresponding singular drugs. Graph statistics for the three networks are listed in [table X] below.

[TODO: May be worth adding other stats e.g. density, component count, diameter, etc]

Graph	Meta nodes	Nodes	Meta edges	Edges
Selfloops	2	19734	11148	5485566
 
Multidrug	4	94353	5	5612510
Non-naive	2	19734	964	5310589


To assess the broad viability of dataset/model combinations, a pilot study was conducted which employed one geometric model (TransE [cite]), one deep learning model (ConvE [cite]), and one matrix factorisation method (ComplEx [cite]) on each of the three datasets. Results indicated that the multidrug graph was a poor input for obtaining quality embeddings, so we decided to exclude it from the main analysis to reduce computational burden by approximately one third. Keeping computational efficiency in mind, LibKGE contains 11 KGE method,  many of which have been shown to produce comparable results when run with the same training methods and loss functions. Therefore to avoid redundancy, we created some exclusion criteria which whittled down our method count to just 5. The criteria included such things as models being direct enhancements of one another, or inefficient performance in the pilot study. A list of all methods and reasons for their in-/exclusion are given in [table X] below.


LibKGE model	Selected	Reason

ComplEx	Yes	Reasonable performance in pilot, often used as 'SOTA' comparator (e.g. for rel tucker3, cp)
ConvE	No	Inefficient performance in pilot
CP	No	SimplE is an 'enhancement' of this model
DistMult	Yes	No reason to exclude, but keep an eye on performance
Relational Tucker 3	Yes	'inspired by' RESCAL which itself was used as a baseline in decagon paper
Rescal	No	Used as 'baseline'  in Decagon paper so already compared
RotatE	No	Prev. work (chapter 1) had trouble running this on smaller graphs (fb15k-237 and wn18rr)
SimplE	Yes	No reason to exclude, but keep an eye on performance
TransE	Yes	Achieved best performance in pilot. Also good to have one non-matrix-factorisation method.
Transformer	No	Unable to be compared to others in '_po' querying
TransH	No	Same as RotatE


[Add paragraph giving overview of chosen methods]

[add paragraph describing config parameter choices]

A portion of both test graphs was removed, prior to the main analysis, for out-of-sample validation. Following the methodology used in the Decagon paper, the holdout data was created by randomly removing 10% of the edges belonging to each polypharmacy side effect. Since the graphs include the same polypharmacy data, the number of holdout edges   is the same for both: 458,061. The number of training edges does differ, however, with selfloops having 5,027,505 and non-naïve having 5,310,589. We measure performance using standard link prediction metrics employed in LibKGE, namely mean reciprocal rank (MRR) and hits@k. We also manually assess with the metrics used by the Decagon authors to allow direct comparison with their model - area under the receiver-operating characteristic (AUROC), area under the precision-recall curve (AUPRC), and average precision at 50 (AP@50). These are calculated individually per side effect type.

This work was carried out using the computational facilities of the Advanced Computing Research Centre, University of Bristol—http://www.bristol.ac.uk/acrc/. The specific environment was CentOS-7 running Python 3.8.12 with PyTorch 1.7.1, accelerated with CUDA 11.4 on 4 × NVIDIA GeForce RTX 2080 Ti.

## Results

## Discussion

## References
