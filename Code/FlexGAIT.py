import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
print("GPUs:", gpus)
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import sys

print("Python:", sys.executable)
print("TensorFlow:", tf.__version__)
print("Build info:", tf.sysconfig.get_build_info())
print("GPUs:", tf.config.list_physical_devices("GPU"))

def transformer_block(x, dim=64, heads=2, mlp_ratio=2):

    # attention
    attn = layers.MultiHeadAttention(
        num_heads=heads,
        key_dim=dim
    )(x, x)

    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # FFN
    ff = layers.Dense(
        dim * mlp_ratio,
        activation="gelu"
    )(x)

    ff = layers.Dense(dim)(ff)

    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    return x

def build_video_encoder():

    inp = layers.Input(shape=(32, 10, 2), name="video_input")

    # [32,10,2] -> [32,20]
    x = layers.Reshape((32, 20))(inp)

    # embedding
    x = layers.Dense(64)(x)

    # transformer
    x = transformer_block(x)

    # temporal pooling
    x = layers.GlobalAveragePooling1D()(x)

    # projection
    x = layers.Dense(128)(x)

    x = layers.LayerNormalization()(x)

    return Model(inp, x, name="video_encoder")


def build_signal_encoder(signal_dim=6):
    inp = layers.Input(shape=(500, signal_dim), name="signal_input")

    x = layers.Conv1D(32, 7, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(64, 5, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128)(x)
    x = layers.LayerNormalization()(x)
    return Model(inp, x, name="signal_encoder")

def build_video_encoder_cnn(embed_dim=6):
        inp = layers.Input(shape=(32, 10, 2))
        x = layers.Reshape((32, 20))(inp)
        x = layers.Dense(embed_dim)(x)
        x = transformer_block(x, dim=embed_dim, heads=transformer_heads)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(128)(x)
        x = layers.LayerNormalization()(x)
        return Model(inp, x, name="video_encoder")

def cross_attention_fusion(z1, z2):

    z1_expand = layers.Reshape((1, 128))(z1)
    z2_expand = layers.Reshape((1, 128))(z2)

    attn = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=128
    )(z1_expand, z2_expand)

    out = layers.Add()([attn, z1_expand])

    out = layers.LayerNormalization()(out)

    out = layers.Flatten()(out)

    return out

def cross_attention_fusion(
        z_video,
        z_signal
):

    # -----------------------------------
    # expand token
    # [B,128] -> [B,1,128]
    # -----------------------------------
    z_video_expand = layers.Reshape((1, 128))(z_video)

    z_signal_expand = layers.Reshape((1, 128))(z_signal)

    # -----------------------------------
    # cross attention
    # query = video
    # key/value = signal
    # -----------------------------------
    attn = layers.MultiHeadAttention(
        num_heads=2,
        key_dim=128
    )(
        z_video_expand,
        z_signal_expand
    )

    # residual
    out = layers.Add()([
        attn,
        z_video_expand
    ])

    out = layers.LayerNormalization()(out)

    # flatten
    out = layers.Flatten()(out)

    return out

def build_cmcan_model(signal_dim=6, num_classes=2):
    left_video = layers.Input(shape=(32, 10, 2), name="left_video")
    right_video = layers.Input(shape=(32, 10, 2), name="right_video")

    left_signal = layers.Input(shape=(500, signal_dim), name="left_signal")
    right_signal = layers.Input(shape=(500, signal_dim), name="right_signal")

    video_encoder = build_video_encoder()
    signal_encoder = build_signal_encoder(signal_dim=signal_dim)

    # left/right feature
    z_lv = video_encoder(left_video)
    z_rv = video_encoder(right_video)

    z_ls = signal_encoder(left_signal)
    z_rs = signal_encoder(right_signal)

    # bilateral fusion
    z_video = layers.Average(name="z_video")([z_lv, z_rv])
    z_signal = layers.Average(name="z_signal")([z_ls, z_rs])

    # multimodal fusion teacher
    z_fusion = cross_attention_fusion(z_video, z_signal)

    # fusion teacher classifier
    x = layers.Dense(64, activation="gelu")(z_fusion)
    x = layers.Dropout(0.3)(x)
    logits = layers.Dense(num_classes, name="logits")(x)

    # video student classifier
    vx = layers.Dense(64, activation="gelu")(z_video)
    vx = layers.Dropout(0.3)(vx)
    video_logits = layers.Dense(num_classes, name="video_logits")(vx)

    # signal student classifier
    sx = layers.Dense(64, activation="gelu")(z_signal)
    sx = layers.Dropout(0.3)(sx)
    signal_logits = layers.Dense(num_classes, name="signal_logits")(sx)

    model = Model(
        inputs={
            "left_video": left_video,
            "right_video": right_video,
            "left_signal": left_signal,
            "right_signal": right_signal,
        },
        outputs={
            "logits": logits,
            "video_logits": video_logits,
            "signal_logits": signal_logits,

            "z_video": z_video,
            "z_signal": z_signal,
            "z_fusion": z_fusion,

            "z_lv": z_lv,
            "z_rv": z_rv,
            "z_ls": z_ls,
            "z_rs": z_rs,
        },
        name="CMCAN_Dual_Student_Distillation"
    )

    return model

tf.keras.backend.clear_session()

# 1. Loss
# =========================
# def info_nce_loss(z1, z2, temperature=0.2):
#     z1 = tf.nn.l2_normalize(z1, axis=1, epsilon=1e-6)
#     z2 = tf.nn.l2_normalize(z2, axis=1, epsilon=1e-6)

#     logits = tf.matmul(z1, z2, transpose_b=True) / temperature
#     labels = tf.range(tf.shape(z1)[0])

#     loss1 = tf.keras.losses.sparse_categorical_crossentropy(
#         labels, logits, from_logits=True
#     )
#     loss2 = tf.keras.losses.sparse_categorical_crossentropy(
#         labels, tf.transpose(logits), from_logits=True
#     )

#     return tf.reduce_mean(loss1 + loss2)


# def asymmetry_loss(z_l, z_r):
#     return tf.reduce_mean(tf.abs(z_l - z_r))


# def latent_alignment_loss(z_single, z_fusion):
#     return tf.reduce_mean(
#         tf.square(z_single - tf.stop_gradient(z_fusion))
#     )

def info_nce_loss(z1, z2, temperature=0.2):
    z1 = tf.nn.l2_normalize(z1, axis=1, epsilon=1e-6)
    z2 = tf.nn.l2_normalize(z2, axis=1, epsilon=1e-6)

    logits = tf.matmul(z1, z2, transpose_b=True) / temperature
    labels = tf.range(tf.shape(z1)[0])

    loss1 = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )
    loss2 = tf.keras.losses.sparse_categorical_crossentropy(
        labels, tf.transpose(logits), from_logits=True
    )

    return tf.reduce_mean(loss1 + loss2)


def asymmetry_loss(z_l, z_r):
    return tf.reduce_mean(tf.abs(z_l - z_r))


def latent_alignment_loss(z_single, z_fusion):
    return tf.reduce_mean(
        tf.square(z_single - tf.stop_gradient(z_fusion))
    )


def distillation_loss(student_logits, teacher_logits, temperature=4.0):
    teacher_prob = tf.nn.softmax(
        tf.stop_gradient(teacher_logits) / temperature,
        axis=1
    )

    student_log_prob = tf.nn.log_softmax(
        student_logits / temperature,
        axis=1
    )

    loss = -tf.reduce_mean(
        tf.reduce_sum(teacher_prob * student_log_prob, axis=1)
    )

    return loss * (temperature ** 2)

from sklearn.metrics import accuracy_score, roc_auc_score, recall_score, confusion_matrix
def evaluate_numpy(x_data, y_data, batch_size=4, output_key="logits"):
    y_true = []
    y_pred = []
    y_prob = []

    for i in range(0, len(y_data), batch_size):
        xb = {
            k: v[i:i + batch_size]
            for k, v in x_data.items()
        }

        yb = y_data[i:i + batch_size]

        out = model(xb, training=False)
        logits = out[output_key]

        prob = tf.nn.softmax(logits, axis=1)[:, 1].numpy()
        pred = (prob >= 0.5).astype(int)

        y_true.extend(yb)
        y_pred.extend(pred)
        y_prob.extend(prob)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    acc = accuracy_score(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except:
        auc = np.nan

    sen = recall_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    spec = tn / (tn + fp + 1e-8)

    return acc, auc, sen, spec

MAX_RUNS = 100          # 最多重新训练100次，防止无限循环
EPOCHS = 30
BATCH_SIZE = 32
#best_cmcan_target_model.keras 目前最佳
# SAVE_PATH = "/home/huangshuai/骨科任务/智能诊断数据/视频数据/model/butai_model_pose/model_save/best_cmcan_target_model_v1.keras"
# SAVE_PATH = "/home/huangshuai/骨科任务/智能诊断数据/视频数据/model/butai_model_pose/model_save/best_cmcan_target_model_v2.keras"
best_score = -1
best_overall_weights = None

for run in range(1, MAX_RUNS + 1):

    print("\n" + "=" * 80)
    print(f"START TRAINING RUN {run}")
    print("=" * 80)

    # =====================================================
    # 1. 每次循环重新建模
    # =====================================================
    tf.keras.backend.clear_session()

    model = build_cmcan_model(signal_dim=6)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=1.0
    )

    # =====================================================
    # 2. 每次循环内部重新定义 train_step
    # =====================================================
    @tf.function
    def train_step(x, y):
        y = tf.cast(y, tf.int32)

        with tf.GradientTape() as tape:
            out = model(x, training=True)

            logits = out["logits"]
            video_logits = out["video_logits"]
            signal_logits = out["signal_logits"]

            z_video = out["z_video"]
            z_signal = out["z_signal"]
            z_fusion = out["z_fusion"]

            z_lv = out["z_lv"]
            z_rv = out["z_rv"]

            cls_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, logits, from_logits=True
                )
            )

            video_cls_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, video_logits, from_logits=True
                )
            )

            signal_cls_loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y, signal_logits, from_logits=True
                )
            )

            video_distill = distillation_loss(
                video_logits,
                logits,
                temperature=4.0
            )

            signal_distill = distillation_loss(
                signal_logits,
                logits,
                temperature=4.0
            )

            ctr_loss = info_nce_loss(z_video, z_signal)

            align_v = latent_alignment_loss(z_video, z_fusion)
            align_s = latent_alignment_loss(z_signal, z_fusion)

            asym_loss = asymmetry_loss(z_lv, z_rv)

            loss = (
                1.0 * cls_loss
                + 0.2 * video_cls_loss
                + 0.4 * signal_cls_loss
                + 0.2 * video_distill
                + 0.4 * signal_distill
                + 0.1 * align_v
                + 0.2 * align_s
                + 0.01 * asym_loss
            )

        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))

        return loss, cls_loss, video_cls_loss, signal_cls_loss, video_distill, signal_distill, ctr_loss, align_v, align_s, asym_loss, acc

    # =====================================================
    # 3. 单次训练
    # =====================================================
    best_val_auc = -1
    best_weights = None

    n = len(train_y_new)

    for epoch in range(EPOCHS):
        idx = np.random.permutation(n)

        train_loss = []
        train_acc = []

        for start in range(0, n, BATCH_SIZE):
            batch_idx = idx[start:start + BATCH_SIZE]

            xb = {
                k: v[batch_idx]
                for k, v in train_x_new.items()
            }

            yb = train_y_new[batch_idx].astype(np.int32)

            (
                loss,
                cls,
                v_cls,
                s_cls,
                v_dis,
                s_dis,
                ctr,
                av,
                ass,
                asym,
                acc
            ) = train_step(xb, yb)

            train_loss.append(loss.numpy())
            train_acc.append(acc.numpy())

        val_acc, val_auc, val_sen, val_spec = evaluate_numpy(
            val_x,
            val_y,
            batch_size=BATCH_SIZE,
            output_key="logits"
        )

        print(
            f"Run {run:03d} | "
            f"Epoch {epoch+1:03d} | "
            f"loss={np.mean(train_loss):.4f} | "
            f"acc={np.mean(train_acc):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_auc={val_auc:.4f} | "
            f"val_sen={val_sen:.4f} | "
            f"val_spec={val_spec:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_weights = model.get_weights()

    # =====================================================
    # 4. 加载本轮 val 最佳权重
    # =====================================================
    if best_weights is not None:
        model.set_weights(best_weights)

    # =====================================================
    # 5. 本轮结束后评估 test / external
    # =====================================================
    test_fusion = evaluate_numpy(
        test_x, test_y, BATCH_SIZE, output_key="logits"
    )

    test_video = evaluate_numpy(
        test_x, test_y, BATCH_SIZE, output_key="video_logits"
    )

    test_signal = evaluate_numpy(
        test_x, test_y, BATCH_SIZE, output_key="signal_logits"
    )

    external_fusion = evaluate_numpy(
        external_x, external_y, BATCH_SIZE, output_key="logits"
    )

    external_video = evaluate_numpy(
        external_x, external_y, BATCH_SIZE, output_key="video_logits"
    )

    external_signal = evaluate_numpy(
        external_x, external_y, BATCH_SIZE, output_key="signal_logits"
    )

    test_video_auc = test_video[1]
    test_signal_auc = test_signal[1]

    external_video_auc = external_video[1]
    external_signal_auc = external_signal[1]

    print("\n===== RUN RESULT =====")
    print("Test Fusion   :", test_fusion)
    print("Test Video    :", test_video)
    print("Test Signal   :", test_signal)
    print("External Fusion:", external_fusion)
    print("External Video :", external_video)
    print("External Signal:", external_signal)

