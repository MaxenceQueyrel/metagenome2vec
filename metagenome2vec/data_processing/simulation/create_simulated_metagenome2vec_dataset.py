import os
import subprocess
import csv
import shutil

from metagenome2vec.utils import parser_creator


if __name__ == "__main__":
    args = parser_creator.ParserCreator().parser_create_simulated_metagenome2vec_dataset()
    path_save = args.path_save

    if args.overwrite and os.path.exists(path_save):
        shutil.rmtree(path_save)
    os.makedirs(path_save, exist_ok=True)
    csvfile = open(os.path.join(path_save, 'metadata.csv'), 'w', newline='')
    writter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    writter.writerow(["id.fasta", "group", "id.subject"])
    cpt = 0

    merge_file_name = "metagenome.fq.gz"
    for path in args.path_data.split(","):
        for root, dirs, files in os.walk(path):
            if os.path.basename(root) == "reads":
                class_name = root.split("/")[-3]
                sample_name = "sample_%s" % cpt
                dir_dst = os.path.join(path_save, sample_name)
                os.makedirs(dir_dst, exist_ok=True)
                if args.to_merge:
                    # files = list(filter(lambda x: re.match("^genome_.*\.fq\.gz", x), files))
                    # if len(files) == 0:
                    #     continue
                    if os.path.isfile(os.path.join(dir_dst, merge_file_name)):
                        subprocess.call("rm %s" % os.path.join(dir_dst, merge_file_name), shell=True)
                    subprocess.call("cat %s/*.fq.gz > %s" % (root, os.path.join(dir_dst, merge_file_name)),
                                    shell=True)
                    writter.writerow([sample_name, class_name, sample_name])
                    cpt += 1
                else:
                    for file in files:
                        if "fq.gz" in file:
                            if os.path.exists(os.path.join(dir_dst, file)):
                                os.remove(os.path.join(dir_dst, file))
                            shutil.copy(os.path.join(root, file), os.path.join(dir_dst, file))
                            writter.writerow([sample_name, class_name, sample_name])
                            cpt += 1
    csvfile.close()
