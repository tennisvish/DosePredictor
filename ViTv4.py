"""
Vision Transformer with Convolutional Patch Embedding
for Single-Cell γH2AX Foci-Based Radiation Dose Estimation

Architecture: ViT with convolutional patch stem, depthwise conv MLP,
cell-cycle conditioning via nuclear area soft quantisation, and
multi-task auxiliary learning (foci count, nuclear area).

Requirements:
    tensorflow >= 2.12
    scikit-learn
    opencv-python
    pandas, numpy, matplotlib, scipy
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

CSV_PATH   = '/kaggle/input/datasets/tennisvish/cleaned-mentor-counts-finaljan17-auxiliarytaskvit/cleaned_mentor_counts_finaljan17.csv'
IMAGES_DIR = '/kaggle/input/traincnn/train copy/'

IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE            = 32
MAX_EPOCHS            = 100
EARLY_STOP_PATIENCE   = 15

CONFIG = {
    'patch':        14,
    'dim':          256,
    'layers':       6,
    'heads':        8,
    'dropout':      0.15,
    'lr':           1e-4,
    'cc_embed_dim': 32,
    'aux_weights': {
        'foci':      0.5,
        'spot_mean': 0.0,
        'nuc_area':  0.3,
    }
}

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_image(filepath):
    """
    Load a single-channel fluorescence microscopy image.
    Applies CLAHE normalisation (clipLimit=2.0, tileGridSize=8×8).
    Returns a (H, W, 1) float32 array in [0, 1], or None on failure.
    """
    try:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img)
        return np.expand_dims(img.astype(np.float32) / 255.0, axis=-1)
    except Exception:
        return None


def compute_normalization_stats(train_meta):
    """
    Compute per-target mean and standard deviation from the training split.
    Stats are computed on training data only to prevent data leakage.
    """
    return {
        'dose':     {'mean': train_meta['dose_Gy'].mean(),
                     'std':  train_meta['dose_Gy'].std()},
        'foci':     {'mean': train_meta['nfoci'].mean(),
                     'std':  train_meta['nfoci'].std()},
        'nuc_area': {'mean': train_meta['nuc_area'].mean(),
                     'std':  train_meta['nuc_area'].std()},
    }


def mixup_batch(images, labels_dict):
    """
    Mixup augmentation (Zhang et al., 2018).
    Blends image pairs and regression targets with lambda ~ Uniform(0.3, 0.7).
    Creates soft interpolated (image, dose) pairs between discrete dose levels,
    encouraging the model to learn a continuous dose manifold.
    """
    bs      = tf.shape(images)[0]
    lam     = tf.random.uniform([bs], 0.3, 0.7)
    lam_img = tf.reshape(lam, [bs, 1, 1, 1])
    lam_lbl = tf.reshape(lam, [bs, 1])
    indices = tf.random.shuffle(tf.range(bs))
    mixed_images = lam_img * images + (1 - lam_img) * tf.gather(images, indices)
    mixed_labels = {
        k: lam_lbl * v + (1 - lam_lbl) * tf.gather(v, indices)
        for k, v in labels_dict.items()
    }
    return mixed_images, mixed_labels


def create_dataset(metadata, batch_size, stats, augment=False, mixup=False):
    """
    Build a tf.data pipeline yielding (image, labels) batches.
    All targets are z-score normalised using training-set statistics.
    Augmentation applies random flips, 90° rotations, and ±20% brightness jitter.
    """
    def generator():
        indices = np.arange(len(metadata))
        if augment:
            np.random.shuffle(indices)
        for idx in indices:
            row = metadata.iloc[idx]
            img = load_image(row['filepath'])
            if img is None:
                continue
            if augment:
                if np.random.rand() > 0.5: img = np.fliplr(img)
                if np.random.rand() > 0.5: img = np.flipud(img)
                if np.random.rand() > 0.5:
                    img = np.rot90(img, np.random.randint(1, 4))
                if np.random.rand() > 0.5:
                    img = np.clip(img * np.random.uniform(0.8, 1.2), 0, 1)

            yield img, {
                'dose':      np.array([(row['dose_Gy']  - stats['dose']['mean'])     / stats['dose']['std']],     dtype=np.float32),
                'foci':      np.array([(row['nfoci']    - stats['foci']['mean'])      / stats['foci']['std']],     dtype=np.float32),
                'spot_mean': np.array([0.0], dtype=np.float32),
                'nuc_area':  np.array([(row['nuc_area'] - stats['nuc_area']['mean'])  / stats['nuc_area']['std']], dtype=np.float32),
            }

    label_sig = {k: tf.TensorSpec(shape=(1,), dtype=tf.float32)
                 for k in ['dose', 'foci', 'spot_mean', 'nuc_area']}
    ds = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(IMG_HEIGHT, IMG_WIDTH, 1), dtype=tf.float32),
            label_sig
        )
    )
    ds = ds.batch(batch_size)
    if mixup:
        ds = ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class ConvPatchEncoder(layers.Layer):
    """
    Convolutional patch embedding stem.

    Replaces the standard single linear projection with a three-layer
    convolutional stem, providing local spatial inductive bias from the
    first layer. The final convolution uses patch_size stride to produce
    exactly (img_size / patch_size)² tokens, matching the standard ViT
    token count while preserving learned local features within each patch.

    Architecture:
        Conv(kernel=3, stride=1, channels=dim//4) → BN → GELU  [edge/texture]
        Conv(kernel=3, stride=1, channels=dim//2) → BN → GELU  [mid-level]
        Conv(kernel=P,  stride=P, channels=dim)   → BN → GELU  [tokenisation]
    """
    def __init__(self, patch_size, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_size     = patch_size
        self.projection_dim = projection_dim

    def build(self, input_shape):
        h, w = input_shape[1], input_shape[2]
        self.num_patches = (h // self.patch_size) * (w // self.patch_size)
        mid = self.projection_dim // 2
        self.conv1 = layers.Conv2D(self.projection_dim // 4, 3, 1, 'same', use_bias=False)
        self.norm1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(mid, 3, 1, 'same', use_bias=False)
        self.norm2 = layers.BatchNormalization()
        self.conv3 = layers.Conv2D(self.projection_dim, self.patch_size,
                                   self.patch_size, 'valid', use_bias=False)
        self.norm3     = layers.BatchNormalization()
        self.pos_embed = layers.Embedding(self.num_patches + 1, self.projection_dim)
        self.cls_token = self.add_weight(
            name='cls', shape=(1, 1, self.projection_dim),
            initializer='random_normal', trainable=True)
        super().build(input_shape)

    def call(self, images, training=False):
        x   = tf.nn.gelu(self.norm1(self.conv1(images), training=training))
        x   = tf.nn.gelu(self.norm2(self.conv2(x),      training=training))
        x   = tf.nn.gelu(self.norm3(self.conv3(x),      training=training))
        b   = tf.shape(images)[0]
        enc = tf.reshape(x, [b, tf.shape(x)[1] * tf.shape(x)[2], self.projection_dim])
        cls = tf.tile(self.cls_token, [b, 1, 1])
        enc = tf.concat([cls, enc], axis=1)
        return enc + self.pos_embed(tf.range(self.num_patches + 1))

    def get_config(self):
        return {**super().get_config(),
                'patch_size': self.patch_size,
                'projection_dim': self.projection_dim}


class TransformerBlock(layers.Layer):
    """
    Post-normalisation transformer block with depthwise convolutional MLP.

    Standard ViT MLP is augmented with a depthwise 3×3 convolution between
    the two dense projections. Patch tokens are reshaped to a spatial grid,
    convolved (mixing spatially neighbouring tokens), then flattened back.
    This introduces local spatial mixing within the feedforward pass,
    complementing the global mixing from multi-head self-attention.
    The CLS token bypasses the spatial mixing step.
    """
    def __init__(self, projection_dim, num_heads, dropout, **kwargs):
        super().__init__(**kwargs)
        self.projection_dim = projection_dim
        self.num_heads      = num_heads
        self.dropout_rate   = dropout

    def build(self, input_shape):
        self.att        = layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.projection_dim // self.num_heads)
        self.mlp_dense1 = layers.Dense(self.projection_dim * 2, activation=tf.nn.gelu)
        self.dw_conv    = layers.DepthwiseConv2D(3, padding='same', use_bias=False,
                                                  depthwise_initializer='glorot_uniform')
        self.mlp_drop   = layers.Dropout(self.dropout_rate)
        self.mlp_dense2 = layers.Dense(self.projection_dim)
        self.norm1      = layers.LayerNormalization(epsilon=1e-6)
        self.norm2      = layers.LayerNormalization(epsilon=1e-6)
        self.drop       = layers.Dropout(self.dropout_rate)
        super().build(input_shape)

    def call(self, x, training=False):
        attn = self.att(self.norm1(x), self.norm1(x))
        x    = x + self.drop(attn, training=training)

        h    = self.mlp_dense1(self.norm2(x))
        cls  = h[:, :1, :]
        ptch = h[:, 1:, :]
        n    = tf.shape(ptch)[1]
        g    = tf.cast(tf.math.round(tf.sqrt(tf.cast(n, tf.float32))), tf.int32)
        pg   = tf.reshape(ptch, [tf.shape(x)[0], g, g, self.projection_dim * 2])
        pg   = self.dw_conv(pg)
        ptch = tf.reshape(pg, [tf.shape(x)[0], -1, self.projection_dim * 2])
        h    = tf.concat([cls, ptch], axis=1)
        h    = self.mlp_drop(h, training=training)
        h    = self.mlp_dense2(h)
        return x + h

    def get_config(self):
        return {**super().get_config(),
                'projection_dim': self.projection_dim,
                'num_heads':      self.num_heads,
                'dropout_rate':   self.dropout_rate}


class NucAreaConditioner(layers.Layer):
    """
    Cell-cycle phase conditioning via soft nuclear area quantisation.

    Nuclear area is a well-established proxy for cell cycle phase:
    G1 (small nuclei), S (intermediate), G2/M (large nuclei). γH2AX
    foci accumulation and repair kinetics vary ~2–3× across phases,
    so conditioning the dose head on predicted cell cycle phase
    provides a biologically informative signal.

    This layer converts a normalised nuclear area prediction into a
    learned 32-dimensional cell-cycle embedding. Soft (differentiable)
    quantisation is used: a softmax over negative squared distances to
    three tertile centres computes a convex combination of three learnable
    phase embeddings (G1, S, G2/M). A learned temperature parameter
    controls assignment sharpness.

    Args:
        embed_dim (int): Dimensionality of the output embedding.
        tertile_boundaries_norm (list): [t33, t67] — 33rd and 67th percentile
            of training-set nuclear area in normalised (z-score) units.
    """
    def __init__(self, embed_dim, tertile_boundaries_norm, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        t0 = tertile_boundaries_norm[0]
        t1 = tertile_boundaries_norm[1]
        self.tertile_centres = tf.constant([
            t0 / 2.0,
            (t0 + t1) / 2.0,
            t1 + (1.0 - t1) / 2.0,
        ], dtype=tf.float32)

    def build(self, input_shape):
        self.embeddings = self.add_weight(
            name='cc_embeddings',
            shape=(3, self.embed_dim),
            initializer='random_normal',
            trainable=True)
        self.temperature = self.add_weight(
            name='temperature',
            shape=(),
            initializer=keras.initializers.Constant(1.0),
            trainable=True,
            constraint=keras.constraints.NonNeg())
        super().build(input_shape)

    def call(self, nuc_pred_norm):
        centres = tf.reshape(self.tertile_centres, [1, 3])
        pred    = tf.reshape(nuc_pred_norm, [-1, 1])
        dists   = tf.square(pred - centres)
        temp    = tf.maximum(self.temperature, 0.1)
        weights = tf.nn.softmax(-dists / temp, axis=1)
        return tf.matmul(weights, self.embeddings)

    def get_config(self):
        return {**super().get_config(),
                'embed_dim': self.embed_dim,
                'tertile_boundaries_norm': [0.0, 0.0]}


def build_model(config, tertile_boundaries_norm):
    """
    Build the full multi-task ViT with cell-cycle conditioning.

    Architecture summary:
        Input (224×224×1)
        → ConvPatchEncoder  → 257 tokens × 256 dim
        → 6 × TransformerBlock
        → LayerNorm → CLS token (256 dim)
        → Shared Dense(512) → Dense(256)
        ├── Nuclear area head → Dense(128) → scalar
        │   └── NucAreaConditioner → 32-dim CC embedding
        ├── Dose head: concat(shared[256], CC[32]) → Dense(128) → scalar
        ├── Foci count head → Dense(128) → scalar
        └── Spot mean head  → Dense(128) → scalar  [loss weight = 0]

    The dose head receives both the shared visual representation and an
    explicit cell-cycle conditioning signal derived from the model's own
    nuclear area prediction, making cell cycle phase information
    available at inference without requiring additional inputs.
    """
    inputs = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 1))
    x      = ConvPatchEncoder(config['patch'], config['dim'])(inputs)

    for i in range(config['layers']):
        x = TransformerBlock(config['dim'], config['heads'],
                             config['dropout'], name=f'transformer_{i}')(x)

    x   = layers.LayerNormalization(epsilon=1e-6)(x)
    cls = x[:, 0, :]

    shared = layers.Dense(512, activation=tf.nn.gelu)(cls)
    shared = layers.Dropout(config['dropout'])(shared)
    shared = layers.Dense(256, activation=tf.nn.gelu)(shared)
    shared = layers.Dropout(config['dropout'])(shared)

    # Nuclear area head
    nuc_h      = layers.Dense(128, activation=tf.nn.gelu)(shared)
    nuc_h      = layers.Dropout(config['dropout'] * 0.5)(nuc_h)
    nuc_output = layers.Dense(1, activation='linear', name='nuc_area')(nuc_h)

    # Cell-cycle conditioning
    cc_embed = NucAreaConditioner(
        embed_dim=config['cc_embed_dim'],
        tertile_boundaries_norm=tertile_boundaries_norm,
        name='cc_conditioner'
    )(nuc_output)

    # Dose head (conditioned on cell-cycle embedding)
    dose_h      = layers.Dense(128, activation=tf.nn.gelu)(
                      layers.Concatenate()([shared, cc_embed]))
    dose_h      = layers.Dropout(config['dropout'] * 0.5)(dose_h)
    dose_output = layers.Dense(1, activation='linear', name='dose')(dose_h)

    # Foci count head
    foci_h      = layers.Dense(128, activation=tf.nn.gelu)(shared)
    foci_h      = layers.Dropout(config['dropout'] * 0.5)(foci_h)
    foci_output = layers.Dense(1, activation='linear', name='foci')(foci_h)

    # Spot mean head (auxiliary; loss weight = 0)
    spot_h      = layers.Dense(128, activation=tf.nn.gelu)(shared)
    spot_output = layers.Dense(1, activation='linear', name='spot_mean')(spot_h)

    return keras.Model(
        inputs=inputs,
        outputs={'dose': dose_output, 'foci': foci_output,
                 'spot_mean': spot_output, 'nuc_area': nuc_output},
        name='vit_convpatch_cc'
    )


# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================

def predict_with_tta(model, img, stats, n_aug=8):
    """
    Test-time augmentation: average dose predictions over geometric transforms.
    Transforms: identity, horizontal flip, vertical flip, 90°/180°/270° rotation,
    and two flip+rotation combinations (total 8 augmentations).
    """
    augments = [
        lambda x: x,
        lambda x: np.fliplr(x),
        lambda x: np.flipud(x),
        lambda x: np.rot90(x, 1),
        lambda x: np.rot90(x, 2),
        lambda x: np.rot90(x, 3),
        lambda x: np.fliplr(np.rot90(x, 1)),
        lambda x: np.flipud(np.rot90(x, 1)),
    ]
    preds = [
        model.predict(np.expand_dims(fn(img), 0), verbose=0)['dose'][0][0]
        * stats['dose']['std'] + stats['dose']['mean']
        for fn in augments[:n_aug]
    ]
    return float(np.mean(preds))


# ============================================================================
# TRAINING
# ============================================================================

def train():
    # ── Data loading ──────────────────────────────────────────────────────────
    df = pd.read_csv(CSV_PATH)
    df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGES_DIR, x))
    df = df[df['filename'].str.endswith('.tif')]
    df = df[df['hr_post_exposure'] == 4]

    spot_med = df[df['nfoci'] > 0]['spot_mean'].median()
    df['spot_mean'] = df['spot_mean'].fillna(
        df['nfoci'].apply(lambda x: 0.0 if x == 0 else spot_med))
    df['nuc_area'] = df['nuc_area'].fillna(df['nuc_area'].median())
    df = df.dropna(subset=['dose_Gy', 'nfoci', 'spot_mean', 'nuc_area'])

    # ── Stratified 70/20/10 split ─────────────────────────────────────────────
    df['dose_group'] = df['dose_Gy'].astype(str)
    train_val, test_meta = train_test_split(
        df, test_size=0.10, random_state=42, stratify=df['dose_group'])
    train_meta, val_meta = train_test_split(
        train_val, test_size=0.222, random_state=42,
        stratify=train_val['dose_group'])
    train_meta = train_meta.reset_index(drop=True)
    val_meta   = val_meta.reset_index(drop=True)
    test_meta  = test_meta.reset_index(drop=True)

    # ── Normalisation ─────────────────────────────────────────────────────────
    stats = compute_normalization_stats(train_meta)

    # ── Cell-cycle tertile boundaries (training set only) ─────────────────────
    nuc_norm_train = ((train_meta['nuc_area'] - stats['nuc_area']['mean'])
                      / stats['nuc_area']['std']).values
    t33 = float(np.percentile(nuc_norm_train, 33))
    t67 = float(np.percentile(nuc_norm_train, 67))

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds    = create_dataset(train_meta, BATCH_SIZE, stats, augment=True,  mixup=True)
    val_ds      = create_dataset(val_meta,   BATCH_SIZE, stats, augment=False, mixup=False)
    train_steps = len(train_meta) // BATCH_SIZE
    val_steps   = len(val_meta)   // BATCH_SIZE

    # ── Learning rate schedule: linear warmup + cosine decay with restarts ────
    class WarmupCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, warmup_steps, base_lr, first_decay_steps):
            super().__init__()
            self.warmup_steps = warmup_steps
            self.base_lr      = base_lr
            self.cosine       = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=base_lr,
                first_decay_steps=first_decay_steps,
                t_mul=1.5, m_mul=0.9, alpha=1e-6)

        def __call__(self, step):
            s = tf.cast(step, tf.float32)
            w = tf.cast(self.warmup_steps, tf.float32)
            return tf.cond(s < w,
                           lambda: self.base_lr * (s / w),
                           lambda: self.cosine(s))

        def get_config(self):
            return {'warmup_steps': self.warmup_steps, 'base_lr': self.base_lr}

    lr_schedule = WarmupCosineDecay(
        warmup_steps=5 * train_steps,
        base_lr=CONFIG['lr'],
        first_decay_steps=20 * train_steps)

    # ── Build and compile ─────────────────────────────────────────────────────
    model = build_model(CONFIG, tertile_boundaries_norm=[t33, t67])
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=lr_schedule, weight_decay=0.02, clipnorm=1.0),
        loss={
            'dose':      'mean_squared_error',
            'foci':      'mean_absolute_error',
            'spot_mean': 'mean_absolute_error',
            'nuc_area':  'mean_absolute_error',
        },
        loss_weights={
            'dose':      1.0,
            'foci':      CONFIG['aux_weights']['foci'],
            'spot_mean': CONFIG['aux_weights']['spot_mean'],
            'nuc_area':  CONFIG['aux_weights']['nuc_area'],
        },
        metrics={'dose': ['mae']}
    )

    # ── Training ──────────────────────────────────────────────────────────────
    start   = datetime.now()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        callbacks=[EarlyStopping(
            monitor='val_dose_loss', patience=EARLY_STOP_PATIENCE,
            mode='min', restore_best_weights=True, verbose=1)],
        verbose=1
    )
    elapsed = (datetime.now() - start).total_seconds() / 3600

    # ── Evaluation ────────────────────────────────────────────────────────────
    dose_true, dose_pred_std, dose_pred_tta = [], [], []
    foci_true, foci_pred, nuc_true, nuc_pred = [], [], [], []

    for _, row in test_meta.iterrows():
        img = load_image(row['filepath'])
        if img is None:
            continue
        p = model.predict(np.expand_dims(img, 0), verbose=0)
        dose_pred_std.append(p['dose'][0][0]     * stats['dose']['std']     + stats['dose']['mean'])
        dose_pred_tta.append(predict_with_tta(model, img, stats, n_aug=8))
        foci_pred.append(    p['foci'][0][0]     * stats['foci']['std']     + stats['foci']['mean'])
        nuc_pred.append(     p['nuc_area'][0][0] * stats['nuc_area']['std'] + stats['nuc_area']['mean'])
        dose_true.append(row['dose_Gy'])
        foci_true.append(row['nfoci'])
        nuc_true.append( row['nuc_area'])

    dose_true     = np.array(dose_true)
    dose_pred_std = np.array(dose_pred_std)
    dose_pred_tta = np.array(dose_pred_tta)
    foci_true = np.array(foci_true); foci_pred = np.array(foci_pred)
    nuc_true  = np.array(nuc_true);  nuc_pred  = np.array(nuc_pred)

    mae_std = mean_absolute_error(dose_true, dose_pred_std)
    r2_std  = r2_score(dose_true, dose_pred_std)
    mae_tta = mean_absolute_error(dose_true, dose_pred_tta)
    r2_tta  = r2_score(dose_true, dose_pred_tta)
    foci_r2 = r2_score(foci_true, foci_pred)
    nuc_r2  = r2_score(nuc_true,  nuc_pred)

    print(f"\n{'Method':<25} {'MAE (Gy)':>10} {'R²':>10}")
    print(f"{'─'*47}")
    print(f"{'Standard':25} {mae_std:>10.4f} {r2_std:>10.4f}")
    print(f"{'TTA (8 augmentations)':25} {mae_tta:>10.4f} {r2_tta:>10.4f}")
    print(f"\nAuxiliary — foci R²={foci_r2:.4f}  nuc_area R²={nuc_r2:.4f}")
    print(f"Training time: {elapsed:.2f} h")

    print("\nPer-dose breakdown (TTA):")
    for d in sorted(np.unique(dose_true)):
        m = dose_true == d
        if m.sum() < 3: continue
        print(f"  {d:.2f} Gy  n={m.sum():4d}  "
              f"MAE={mean_absolute_error(dose_true[m], dose_pred_tta[m]):.4f}  "
              f"R²={r2_score(dose_true[m], dose_pred_tta[m]):.4f}")

    # ── Persistence ───────────────────────────────────────────────────────────
    model.save('vit_convpatch_cc.keras')
    pd.DataFrame({
        'dose_true': dose_true,
        'dose_pred': dose_pred_std,
        'dose_pred_tta': dose_pred_tta,
        'foci_true': foci_true, 'foci_pred': foci_pred,
        'nuc_true':  nuc_true,  'nuc_pred':  nuc_pred,
    }).to_csv('results.csv', index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    colors = {0.0: '#4477AA', 0.10: '#66CCEE', 0.30: '#228833',
              0.82: '#CCBB44', 1.0:  '#EE6677'}

    ax = axes[0]
    for d, col in colors.items():
        m = dose_true == d
        if m.sum():
            ax.scatter(dose_true[m], dose_pred_tta[m], alpha=0.45, s=16,
                       c=col, label=f'{d:.2f} Gy')
    lo = min(dose_true.min(), dose_pred_tta.min()) - 0.02
    hi = max(dose_true.max(), dose_pred_tta.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], 'r--', lw=2, label='Identity')
    ax.set_xlabel('True Dose (Gy)'); ax.set_ylabel('Predicted Dose (Gy)')
    ax.set_title(f'Predicted vs True (TTA)\nR² = {r2_tta:.4f}  MAE = {mae_tta:.4f} Gy')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[1]
    for d, col in colors.items():
        m = dose_true == d
        if m.sum():
            ax.hist(dose_pred_tta[m], bins=30, alpha=0.6, color=col,
                    label=f'{d:.2f} Gy (n={m.sum()})')
    ax.set_xlabel('Predicted Dose (Gy)'); ax.set_ylabel('Count')
    ax.set_title('Prediction Distributions by Dose Level (TTA)')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(history.history['dose_loss'],     label='Train', lw=2, color='steelblue')
    ax.plot(history.history['val_dose_loss'], label='Validation', lw=2, color='coral')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dose Loss (MSE, normalised)')
    ax.set_title('Training and Validation Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    plt.show()

    return model, history, r2_std, r2_tta


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    model, history, r2_std, r2_tta = train()
    print(f"\nR² (standard): {r2_std:.4f}")
    print(f"R² (TTA):      {r2_tta:.4f}")
