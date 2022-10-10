Example execution of Final_Script.py:

python Final_Script.py -f 'FILE_MAF_20210727T231032Z_GMO_00000000001650_01' -p 'Final-NBCU-Contextual-IAB-Taxonomy-Mapping-2022-Copy.csv' -t
'IAB_Tech_Lab_Content_Taxonomy_V2_Final_2017-11.xlsx' -l 'http://ac3448e420fce11eaaa4b0a458c10dab-684955813.us-east-1.elb.amazonaws.com/MafData/rs/db/mafdb/analysis/'

The code for Creating the pre-processed input that the Testing files take resides in Hadi_Segment_datacreation.ipynb and the first half of the cells in HadiEvaluation_segment.ipynb

The NRS and MAP code is in the cells of the Testing_NRS_... notebook files. The workflow is: Generate the relevance scores for each segment per file and save it in the Relevance_...pkl files. Then reload them to calculate the NRS and MAP

TopN Sentence generation is done by the TopSentencesGenerator.py script. It currenlty works on all of the IAB Taxonomy.

The Unsupervised Learning approaches are in their own folder and are broken into training and testing. In GPL we do the preprocessing first and then the bi-encoder training.

Several intermediate data files were larger than Github's allowed limit of 100MB so they have been uploaded in the One_Drive Link here: https://comcastcorp-my.sharepoint.com/:f:/g/personal/eyoune200_cable_comcast_com/EqnUapOHYJlGuZniE5qfonYBlJKne-97__cARdBUWHJ-kA?e=bAeXm0

