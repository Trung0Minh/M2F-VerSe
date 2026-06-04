
_base_ = [
    '../upernet/upernet_r50_4xb4-160k_ade20k-512x512.py'
]

model = dict(
    decode_head=dict(num_classes=4),
    auxiliary_head=dict(num_classes=4))

dataset_type = 'BaseSegDataset'
data_root = 'dataset_verse_2d/ade20k'
classes = ('background', 'cervical', 'thoracic', 'lumbar')
palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255]]

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=False),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='train/images', 
            seg_map_path='train/annotations_semantic'),
        pipeline=train_pipeline,
        metainfo=dict(classes=classes, palette=palette)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        img_suffix='.png',
        seg_map_suffix='.png',
        reduce_zero_label=False,
        data_prefix=dict(
            img_path='val/images',
            seg_map_path='val/annotations_semantic'),
        pipeline=test_pipeline,
        metainfo=dict(classes=classes, palette=palette)))

test_dataloader = val_dataloader
val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
test_evaluator = val_evaluator

load_from = 'semantic/mmsegmentation/weights/upernet_r50_512x512_160k_ade20k_20200615_184328-8534de8d.pth'

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1)}),
    clip_grad=None)

# Standardized Schedule with _delete_=True to clear base epochs
train_cfg = dict(_delete_=True, type='IterBasedTrainLoop', max_iters=20000, val_interval=5000)
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=1000),
    dict(type='PolyLR', eta_min=1e-5, power=0.9, begin=1000, end=20000, by_epoch=False)
]

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=5000, by_epoch=False, max_keep_ckpts=1),
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50))

work_dir = './output/verse_semantic_upernet'
