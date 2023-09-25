# Thesis chapter 2 - Decagon vs LibKGE

All code and data used in analysis for the second chapter of my thesis, an assessment of Decagon's strength on polypharmic side effect modelling vs general knowledge graph embedding methods.

Decagon paper: https://doi.org/10.1093/bioinformatics/bty294

LibKGE: https://github.com/uma-pi1/kge

Repository structure:

- Analysis
    - Assessment
        - False_edges:

            Directory for storing the false edges that are created during the assessment process.

        - assessment.py:

            Script for assessing a trained LibKGE model (stored as a .pt file) following the process described in the Decagon paper. The script should be called from the command line with the following arguments:

                - model_checkpoint - path to the LibKGE model.
                - holdout_edges - path to the holdout edges in TSV format.
                - libkge_data_dir - path to the relevant dataset in LibKGE. This is necessary so that the entity and relation lists can be loaded from file and don't have to be parsed from edge data.
                - (optional) out_name - name to give final results file. By default the name will be parsed from libkge_data_dir.
                - (optional) partial_results - path to a partial result file in CSV format. The script can take a while to run, so 'results_temp.csv' is saved after each side effect type is assessed. Pass the path to that file with this argument and the script will pick up from the next side effect. If this argument isn't used, the script will assess all 964 types of side effects from the beginning. 

        - decagon_rank_metrics.py:

            'Average precision at K' functions created by the Decagon authors to assess their model. Taken directly from their GitHub repo https://github.com/mims-harvard/decagon/blob/master/decagon/utility/rank_metrics.py

        - metric_validation.py:

            Script testing the behaviour of the AUROC, AUPRC, and AP@50 metrics, to ensure they are working as intended.

    - Experiments
        - config_templates:

            A template LibKGE configuration file (.yaml) for each model being tested in the experiment. The only things that need changing to run the experiments are the dataset name (ctrl + f "dataset.name: replace_me"), and for the 'non-naive' dataset the path to directory of the embedding initialisation vectors should be passed as "user: pretrained_path: {path}" at the end of the file. 

        - non-naive + selfloops:

            Ready-to-run configuration files, for the corresponding datasets. Begin experiments with `kge start {path_to_config}`.

        - pilot:

            Config and results files from the first round of testing with very short LibKGE experiments.
- Data
    - Figures/EDA:

        Script and figures exploring various aspects of the per-meta-edge graph attributes.

    - graphs/*: 

        For each of the two datasets used, plus the 'multidrug' dataset which was excluded during testing, a directory containing their full edgelist and generating script. Graph stats and schema are also included here. For proper dataset descriptions please see the methods section of the draft paper.

    - processed:

        Note that the TSV files in this directory are too large to store with git so should be re-generated with the corresponding scripts. 

        - polypharmacy_split:
            
            Script (and its outputs) to split 10% of the edges from each side effect type from '../polypharmacy_edges.tsv' and store these as holdout data. Note that to avoid data leakage these edges, stored as 'holdout_polypharmacy.tsv', should never be included in any edgelist passed to LibKGE itself and should be reserved for assessment. The edges in 'train_polypharmacy.tsv' are the ones that should be used. 

        - core_network_ppi_drugtarget.tsv
            
            Drug-target and protein-protein-interaction edges. These edges are the core network used in all versions of the graph.

        - monopharmacy_edges.tsv

            Edgelist of single drug nodes to side effect nodes. These edges need further processing (done in '../graphs') to be used in the 'selfloops' or 'non-naive' datasets, and were only used raw in the now redundant 'multidrug' graph.

        - polypharmacy_edges.tsv

            Raw polypharmacy side effect edges, in triple form drugA -> side effect -> drug B. The file should not be used raw and only be passed to the script in 'polypharmacy_split'.

        - process_raw_data.py

            Script that takes raw data from '../raw' and produces the TSV files described above.

    - raw:
    
        Copies of the relevant raw Decagon data provided at http://snap.stanford.edu/decagon/. Again, the files are too large to store with git but can be redownloaded with `sh download_and_unpack.sh`. 

- Paper
    - paper.docx
        
        Unfinished manuscript for the project. Contains a fully drafted intro and methods section, as well as supplementary text describing the various deep learning optimizers that are included in the LibKGE grid searches.

Contact: oliver.lloyd@bristol.ac.uk
