python main.py --train_img_dir data_dev/celeba_hq/train \
               --val_img_dir data_dev/celeba_hq/val \
               --w_hpf 0 \
               --total_iters 3 \
               --print_every 1 \
               --sample_every 1 \
               --save_every 1 \
               --whichgpu 0 \
               --latent_dim 16 \
               --hidden_dim 512 \
               --style_dim 64 \
               --batch_size 8 \
               --val_batch_size 32 \
               --num_outs_per_domain 4

# todo: y_org [0, 1], y_trg [1, 2]
