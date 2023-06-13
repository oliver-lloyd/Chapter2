# Description of contents of raw files

bio-decagon-ppi.csv:	Protein-protein interaction network (n=715612)
bio-decagon-targets.csv: 	Drug-target protein associations. Is a subset of 'bio-decagon-targets-all' (n=18690)
bio-decagon-targets-all.csv: 	Drug-target protein associations culled from several curated databases (n=131034). Note that including this file in analysis introduces ~1400 drug nodes and ~112k drug-target edges MORE than are reported in the paper. For this reason it will be excluded from all analyses.
bio-decagon-combo.csv:	Polypharmacy side effects in the form of (drug A, side effect type, drug B) triples (n=4649441)
bio-decagon-mono.csv: 	Side effects of individual drugs in the form of (drug A, side effect type) tuples (n=174977)
bio-decagon-effectcategories.csv: 	Side effect categories (n=561)


Drug IDs: 'STITCH' column header, form is 'CID' followed by 8 numerics. E.g. CID003062316. (n=645)
Protein IDs: 'Gene' column header, form is integer ID up to 6 digits. E.g. 375519. (n=19089)
Side effect IDs: 'Side Effect' column, form is 'C' followed by 7 numerics. E.g. C1096328. (n=11501)
