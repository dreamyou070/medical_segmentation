import os
import torch
from data.dataset_multi import TrainDataset_Seg
from model.tokenizer import load_tokenizer

def call_dataset(args) :

    # [1] load tokenizer
    tokenizer = load_tokenizer(args)

    # [2] train & test dataset
    train_dataset = TrainDataset_Seg(root_dir=args.train_data_path,
                                     resize_shape=[args.resize_shape,args.resize_shape],
                                     tokenizer=tokenizer,
                                     caption=args.trigger_word,
                                     latent_res=args.latent_res,
                                     n_classes = args.n_classes,
                                     mask_res = args.mask_res,
                                     use_patch = args.use_patch,
                                     patch_size = args.patch_size,)
    test_dataset = TrainDataset_Seg(root_dir=args.test_data_path,
                                    resize_shape=[args.resize_shape,args.resize_shape],
                                    tokenizer=tokenizer,
                                    caption=args.trigger_word,
                                    latent_res=args.latent_res,
                                    n_classes=args.n_classes,
                                    mask_res = args.mask_res,
                                    use_patch = args.use_patch,
                                    patch_size = args.patch_size,)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=args.batch_size,
                                                  shuffle=False)
    patch_num = None
    if args.use_patch :
        patch_num = train_dataset.patch_num
    return train_dataloader, test_dataloader, patch_num