# Config for StyleGANv1 training and generation

_base_ = [
    '../_base_/models/base_styleganv1.py',
    '../_base_/datasets/grow_scale_imgs_ffhq_styleganv1.py',
    '../_base_/gen_default_runtime.py',
]

# Model configuration
model = dict(
    generator=dict(
        out_size=1024,   # Output image size
        latent_dim=512,  # Latent vector dimension
        n_layer=18,      # Number of layers in the generator
        channel_base=32768,  # Channel base for network architecture
    ),
    discriminator=dict(
        in_size=1024,    # Input image size
        n_layer=18,      # Number of layers in the discriminator
        channel_base=32768,  # Channel base for network architecture
    ),
    ema=dict(
        enable=True,     # Enable Exponential Moving Average
        decay=0.9999,    # EMA decay rate
    ),
    loss=dict(
        r1_weight=10.0,  # Weight for R1 regularization
    )
)

# TRAIN
train_cfg = dict(max_iters=60000)


# Optimizer
optim_wrapper = dict(
    generator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * g_reg_ratio, betas=(0,
                                                        0.99**g_reg_ratio))),
    discriminator=dict(
        optimizer=dict(
            type='Adam', lr=0.002 * d_reg_ratio, betas=(0,
                                                        0.99**d_reg_ratio))))

# Data loader
data_root = './data/dataset'
train_dataloader = dict(
    batch_size=train_cfg['batch_size'],
    dataset=dict(data_root=data_root),
)

# Custom hooks
custom_hooks = [
    dict(
        type='VisualizationHook',
        interval=1000,
        fixed_input=True,
        vis_kwargs=dict(type='GAN', name='generated_image')
    )
]

# Metrics
metrics = [
    dict(
        type='FrechetInceptionDistance',
        prefix='FID',
        inception_style='StyleGAN',
    ),
    dict(
        type='InceptionScore',
        prefix='IS',
    ),
]



# Evaluation
val_evaluator = dict(
    metrics=metrics,
)

test_evaluator = dict(
    metrics=metrics,
)
