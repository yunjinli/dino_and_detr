bsub -Is -q gpu -gpu "num=1:j_exclusive=yes:gmem=8G" -R "select[ui==aiml_python && osrel==70 && type==X64LIN]" python evaluate_single.py --coco_path /home/phd_li/dataset/RDD2022/ --batch_size 4 --output_dir ./output_dino --num_queries 16 --dim_feedforward 512 --enc_layers 3 --dec_layers 3 --no_aux_loss --eval --resume ./output_dino/checkpoint.pth --dino

