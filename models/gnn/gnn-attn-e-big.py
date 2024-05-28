# model settings
model_cfg = dict(
    backbone=dict(type='GNNAttnE', version='big'),
    neck=[
        dict(type='HRFuseScales', in_channels=(64, 128, 256, 512)),
        dict(type='GlobalAveragePooling'),
    ],
    head=dict(
        type='LinearClsHead',
        in_channels=2048,
        num_classes=2,
        loss=dict(type='CrossEntropyLoss', loss_weight=1.0),
        topk=(1, 5),
    ))

img_norm_cfg = dict(
    mean=[40.95984366, 37.61124436, 35.80594691], std=[48.51076895, 45.51812755, 44.62745192], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFileE'),
    # dict(type='RandomResizedCrop', size=224, backend='pillow'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'img_mask']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'img_mask', 'gt_label'])
]
val_pipeline = [
    dict(type='LoadImageFromFileE'),
    dict(type='Resize', size=(256, -1), backend='pillow'),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img', 'img_mask']),
    dict(type='Collect', keys=['img', 'img_mask'])
]

# train
data_cfg = dict(
    batch_size=16,
    num_workers=4,
    train=dict(
        pretrained_flag=False,
        pretrained_weights='',
        freeze_flag=False,
        freeze_layers=('backbone',),
        epoches=100,
    ),
    test=dict(
        ckpt='',
        metrics=['accuracy', 'precision', 'recall', 'f1_score', 'confusion'],
        metric_options=dict(
            topk=(1, 5),
            thrs=None,
            average_mode='none'
        )
    )
)

paramwise_cfg = dict(
    norm_decay_mult=0.0,
    bias_decay_mult=0.0,
    custom_keys={
        '.absolute_pos_embed': dict(decay_mult=0.0),
        '.relative_position_bias_table': dict(decay_mult=0.0)
    })

# batch 16
# lr = 0.1 * 16 /256
# optimizer
optimizer_cfg = dict(
    type='SGD',
    lr=0.1 * 16 / 256,
    momentum=0.9,
    weight_decay=1e-4)

# learning
lr_config = dict(type='CosineAnnealingLrUpdater', min_lr=0)
# optimizer_cfg = dict(
#     type='SGD',
#     lr=0.01,
#     momentum=0.9,
#     weight_decay=1e-4)
#
# # learning
# lr_config = dict(type='StepLrUpdater', step=20, min_lr=1e-8)