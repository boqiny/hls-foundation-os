import os 
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True
custom_imports = dict(imports=['geospatial_fm'])
num_frames = 3
img_size = 224
num_workers = 2

# model
# TO BE DEFINED BY USER: model path
pretrained_weights_path = '/home/charles/Desktop/FM/hls-foundation-os/Prithvi_100M.pt'
num_layers = 6
patch_size = 16
embed_dim = 768
num_heads = 8
tubelet_size = 1
max_epochs = 80
eval_epoch_interval = 5

loss_weights_multi = [
    0.386375, 0.661126, 0.548184, 0.640482, 0.876862, 0.925186, 3.249462,
    1.542289, 2.175141, 2.272419, 3.062762, 3.626097, 1.198702
]

loss_func = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,
    class_weight=loss_weights_multi,
    avg_non_ignore=True)
output_embed_dim = embed_dim*num_frames


# TO BE DEFINED BY USER: Save directory
experiment ='classification_test'
project_dir = '../output/'
work_dir = os.path.join(project_dir, experiment)
save_path = work_dir


gpu_ids = range(0, 1)
dataset_type = 'GeospatialDataset'

# TO BE DEFINED BY USER: data directory
data_root = '/home/charles/Desktop/FM/hls-foundation-os/data_splits/multi_temporal_crop_classification/'

splits = dict(
    train=os.path.join(data_root, 'training_data.txt'),
    val=os.path.join(data_root, 'validation_data.txt'),
    test=os.path.join(data_root, 'validation_data.txt') 
)



img_norm_cfg = dict(
    means=[
        494.905781, 815.239594, 924.335066, 2968.881459, 2634.621962,
        1739.579917, 494.905781, 815.239594, 924.335066, 2968.881459,
        2634.621962, 1739.579917, 494.905781, 815.239594, 924.335066,
        2968.881459, 2634.621962, 1739.579917
    ],
    stds=[
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808,
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808,
        284.925432, 357.84876, 575.566823, 896.601013, 951.900334, 921.407808
    ])

bands = [0, 1, 2, 3, 4, 5]

tile_size = 224
orig_nsize = 512
crop_size = (tile_size, tile_size)
train_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
    dict(type='LoadGeospatialAnnotations', reduce_zero_label=True),
    dict(type='RandomFlip', prob=0.5),

    #Add more augmentation techniques 
    dict(type='RandomRotation', degrees=10), # random rotation between -10 and 10 degrees
    dict(type='RandomResizedCrop', height=tile_size, width=tile_size, scale=(0.8, 1.0)), # random scaling
    dict(type='ColorJitter', brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # color jittering

    dict(type='ToTensor', keys=['img', 'gt_semantic_seg']),
     # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='TorchRandomCrop', crop_size=crop_size),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, tile_size, tile_size)),
    dict(type='Reshape', keys=['gt_semantic_seg'], new_shape=(1, tile_size, tile_size)),
    dict(type='CastTensor', keys=['gt_semantic_seg'], new_type="torch.LongTensor"),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadGeospatialImageFromFile', to_float32=True, channels_last=True),
    dict(type='ToTensor', keys=['img']),
     # to channels first
    dict(type="TorchPermute", keys=["img"], order=(2, 0, 1)),
    dict(type='TorchNormalize', **img_norm_cfg),
    dict(type='Reshape', keys=['img'], new_shape=(len(bands), num_frames, -1, -1), look_up = {'2': 1, '3': 2}),
    dict(type='CastTensor', keys=['img'], new_type="torch.FloatTensor"),
    dict(type='CollectTestList', keys=['img'],
         meta_keys=['img_info', 'seg_fields', 'img_prefix', 'seg_prefix', 'filename', 'ori_filename', 'img',
                    'img_shape', 'ori_shape', 'pad_shape', 'scale_factor', 'img_norm_cfg']),
]

CLASSES = ('Natural Vegetation', 
           'Forest', 
           'Corn', 
           'Soybeans', 
           'Wetlands', 
           'Developed/Barren', 
           'Open Water', 
           'Winter Wheat', 
           'Alfalfa', 
           'Fallow/Idle Cropland', 
           'Cotton', 
           'Sorghum', 
           'Other')

dataset = 'GeospatialDataset'

data = dict(
    #change double batch size, also change lr
    # samples_per_gpu=16,

    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir='training_chips',
        ann_dir='training_chips',
        pipeline=train_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=splits['train']),
    val=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir='validation_chips',
        ann_dir='validation_chips',
        pipeline=test_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=splits['val']
    ),
    test=dict(
        type=dataset,
        CLASSES=CLASSES,
        reduce_zero_label=True,
        data_root=data_root,
        img_dir='validation_chips',
        ann_dir='validation_chips',
        pipeline=test_pipeline,
        img_suffix='_merged.tif',
        seg_map_suffix='.mask.tif',
        split=splits['val']
    ))

#change
# double learning rate to 3e-05
optimizer = dict(
    type='Adam', lr=1.5e-05, betas=(0.9, 0.999), weight_decay=0.05)
optimizer_config = dict(grad_clip=None)

# If you're using ReduceLROnPlateau:
lr_config = dict(
    policy='ReduceLROnPlateau',
    metric='validation_loss',  # assuming validation loss is your metric; change if different
    patience=5,
    factor=0.1,
    min_lr=1e-6,
    by_epoch=False
)


# # If you're using OneCycleLR:
# lr_config = dict(
#     policy='OneCycleLR',
#     max_lr=0.01,
#     total_steps=max_epochs*len(data['train']),  # assuming data['train'] gives your training dataset
#     epochs=max_epochs,
#     steps_per_epoch=len(data['train']),
#     anneal_strategy='cos',
#     final_div_factor=100,
#     three_phase=False,
#     by_epoch=False
# )

# #original
# lr_config = dict(
#     policy='poly',
#     warmup='linear',
#     warmup_iters=1500,
#     warmup_ratio=1e-06,
#     power=1.0,
#     min_lr=0.0,
#     by_epoch=False)

log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])

checkpoint_config = dict(
    by_epoch=True,
    interval=100,
    out_dir=save_path)

evaluation = dict(interval=eval_epoch_interval, metric='mIoU', pre_eval=True, save_best='mIoU', by_epoch=True)
reduce_train_set = dict(reduce_train_set=False)
reduce_factor = dict(reduce_factor=1)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
workflow = [('train', 1)]
norm_cfg = dict(type='BN', requires_grad=True)

model = dict(
    type='TemporalEncoderDecoder',
    frozen_backbone=False,
    backbone=dict(
        #change
        # type='ResNet',
        type='TemporalViTEncoder',
        pretrained=pretrained_weights_path,
        img_size=img_size,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=1,
        in_chans=len(bands),
        embed_dim=embed_dim,
        depth=6,

        #change
        # depth=10,
        num_heads=num_heads,
        mlp_ratio=4.0,
        norm_pix_loss=False),
    neck=dict(
        type='ConvTransformerTokensToEmbeddingNeck',
        embed_dim=embed_dim*num_frames,
        output_embed_dim=output_embed_dim,
        drop_cls_token=True,
        Hp=14,
        Wp=14),
    decode_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=1,

        #change
        # num_convs = 2,

        concat_input=False,

        #change
        # dropout_ratio=0.1,
        dropout_ratio=0.3,

        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=loss_func),
    auxiliary_head=dict(
        num_classes=len(CLASSES),
        in_channels=output_embed_dim,
        type='FCNHead',
        in_index=-1,
        channels=256,
        num_convs=2,
        concat_input=False,

        # dropout_ratio=0.1,
        dropout_ratio=0.3,

        norm_cfg=dict(type='BN', requires_grad=True),
        align_corners=False,
        loss_decode=loss_func),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', stride=(int(tile_size/2), int(tile_size/2)), crop_size=(tile_size, tile_size)))
auto_resume = False
