import sys
import os
from tqdm import tqdm
import re

root_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.path.join(root_folder, "utils"))

import parser_creator
if sys.version_info[0] == 3 and sys.version_info[1] == 7:
    import transformation_ADN
else:
    import transformation_ADN2 as transformation_ADN
import hdfs_functions as hdfs

if __name__ == "__main__":

    ###################################
    # ------ Script's Parameters ------#
    ###################################

    parser = parser_creator.ParserCreator()
    args = parser.parser_genome_kmerization()

    path_data = args.path_data
    path_save = args.path_save
    hdfs.create_dir(os.path.dirname(path_save), "local")
    k = args.k_mer_size
    s = args.step

    ###################################
    # ------------- Run --------------#
    ###################################

    line_to_write = ""
    n_lines = sum(1 for line in open(path_data, 'r'))
    with open(path_data, 'r') as genome:
        with open(path_save, 'w') as f_res:
            for line in tqdm(genome, total=n_lines):
                if line[0] == ">" or line == "\n":
                    continue
                line = re.sub("NN+", "", line).replace("\n", "")
                transformation_ADN.cut_and_write_read(f_res, line, k, s)
