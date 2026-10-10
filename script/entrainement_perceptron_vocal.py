import os
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)


# ============================================================
# CONFIGURATION
# ============================================================

# Chemins conservés comme demandé : tu pourras les rectifier plus tard.
DOSSIER_ENTRAINEMENT = "/home/entrainement"
DOSSIER_TEST = "/home/Test"
MODELE = "/modele/modele_recon_perso.h5"

CLASSES = [
    "Autres",
    "Manda",
    "autresperso"
]

SAMPLE_RATE = 16000
DUREE = 5
N_MFCC = 40

EPOCHS = 500
BATCH_SIZE = 32
SEED = 42

# Reproductibilité partielle des résultats
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# EXTRACTION DES CARACTERISTIQUES MFCC
# ============================================================

def extraire_mfcc(fichier_audio):
    try:
        audio, sr = librosa.load(
            fichier_audio,
            sr=SAMPLE_RATE,
            duration=DUREE,
            mono=True
        )

        if audio.size == 0:
            print(f"Audio vide : {fichier_audio}")
            return None

        longueur = SAMPLE_RATE * DUREE

        if len(audio) < longueur:
            audio = np.pad(audio, (0, longueur - len(audio)))
        else:
            audio = audio[:longueur]

        # Évite les valeurs non finies et protège les audios presque silencieux.
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        ecart_type = np.std(mfcc)
        mfcc = (mfcc - np.mean(mfcc)) / (ecart_type + 1e-8)
        mfcc = mfcc.flatten()

        if not np.all(np.isfinite(mfcc)):
            print(f"Caractéristiques MFCC invalides : {fichier_audio}")
            return None

        return mfcc

    except Exception as e:
        print(f"Erreur avec {fichier_audio} : {e}")
        return None


# ============================================================
# CHARGEMENT DES DONNEES
# ============================================================

EXTENSIONS_AUDIO = (".wav", ".mp3", ".ogg", ".flac", ".m4a")


def charger_donnees(dossier):
    X = []
    y = []

    if not os.path.isdir(dossier):
        print(f"\nERREUR : dossier introuvable : {dossier}")
        return np.empty((0, N_MFCC)), np.array([], dtype=str)

    for classe in CLASSES:
        dossier_classe = os.path.join(dossier, classe)

        if not os.path.isdir(dossier_classe):
            print(f"Dossier introuvable : {dossier_classe}")
            continue

        print(f"\nChargement de la classe : {classe}")

        # Tri pour rendre l'ordre de chargement reproductible.
        fichiers = sorted(os.listdir(dossier_classe))
        nombre_classe = 0

        for fichier in fichiers:
            if not fichier.lower().endswith(EXTENSIONS_AUDIO):
                continue

            chemin = os.path.join(dossier_classe, fichier)

            if not os.path.isfile(chemin):
                continue

            # Avertit si le fichier dépasse 1 Mio, sans le rejeter.
            try:
                taille_mio = os.path.getsize(chemin) / (1024 * 1024)
                if taille_mio > 1:
                    print(f"Attention : fichier supérieur à 1 Mo : {chemin}")
            except OSError:
                pass

            mfcc = extraire_mfcc(chemin)

            if mfcc is not None:
                X.append(mfcc)
                y.append(classe)
                nombre_classe += 1

        print(f"Fichiers audio utilisables pour {classe} : {nombre_classe}")

    if not X:
        return np.empty((0, N_MFCC)), np.array([], dtype=str)

    return np.asarray(X, dtype=np.float32), np.asarray(y)


# ============================================================
# CHARGEMENT ENTRAINEMENT
# ============================================================

print("\n==============================")
print("CHARGEMENT DES DONNEES")
print("==============================")

X, y = charger_donnees(DOSSIER_ENTRAINEMENT)

print("\nNombre total de fichiers :", len(X))
print("Taille des caractéristiques :", X.shape)

if len(X) == 0:
    raise SystemExit(
        "Aucun fichier audio utilisable. Vérifie DOSSIER_ENTRAINEMENT "
        "et les sous-dossiers de classes."
    )

classes_presentes = set(y.tolist())
classes_manquantes = set(CLASSES) - classes_presentes
if classes_manquantes:
    raise SystemExit(
        "Entraînement interrompu : aucune donnée utilisable pour "
        + ", ".join(sorted(classes_manquantes))
    )

effectifs = {classe: int(np.sum(y == classe)) for classe in CLASSES}
print("\nNombre de fichiers par classe :", effectifs)

if min(effectifs.values()) < 2:
    raise SystemExit(
        "Il faut au moins 2 fichiers audio utilisables dans chaque classe "
        "pour effectuer une séparation stratifiée entraînement/validation."
    )


# ============================================================
# ENCODAGE DES CLASSES
# ============================================================

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("\nOrdre des classes utilisé par LabelEncoder :")
for i, classe in enumerate(encoder.classes_):
    print(i, "=", classe)

y_categorical = to_categorical(
    y_encoded,
    num_classes=len(encoder.classes_)
)


# ============================================================
# DIVISION TRAIN / VALIDATION
# ============================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X,
    y_categorical,
    test_size=0.20,
    random_state=SEED,
    stratify=y_encoded
)

print("\nDonnées entraînement :", X_train.shape)
print("Données validation :", X_validation.shape)


# ============================================================
# CREATION DU RESEAU DE NEURONES
# Architecture conservée : Dense + Dropout + ReLU + Softmax
# ============================================================

input_size = X_train.shape[1]

model = Sequential([
    Input(shape=(input_size,)),

    Dense(512, activation="relu"),
    Dropout(0.30),

    Dense(256, activation="relu"),
    Dropout(0.30),

    Dense(128, activation="relu"),
    Dropout(0.20),

    Dense(64, activation="relu"),

    Dense(len(encoder.classes_), activation="softmax")
])


# ============================================================
# COMPILATION
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==============================")
print("ARCHITECTURE DU MODELE")
print("==============================")
model.summary()


# ============================================================
# CALLBACKS : MEILLEUR MODELE, ARRET ANTICIPE, TAUX D'APPRENTISSAGE
# ============================================================

dossier_modele = os.path.dirname(MODELE)
if dossier_modele:
    os.makedirs(dossier_modele, exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath=MODELE,
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=25,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.5,
    patience=8,
    min_lr=1e-6,
    verbose=1
)


# ============================================================
# ENTRAINEMENT
# ============================================================

print("\n==============================")
print("DEBUT DE L'ENTRAINEMENT")
print("==============================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[checkpoint, reduce_lr, early_stopping],
    verbose=1,
    shuffle=True
)


# ============================================================
# RECHARGER LE MEILLEUR MODELE SAUVEGARDE
# ============================================================

if os.path.isfile(MODELE):
    print("\nRechargement du meilleur modèle sauvegardé...")
    model = tf.keras.models.load_model(MODELE)
else:
    print("\nAttention : aucun checkpoint trouvé ; utilisation du modèle en mémoire.")


# ============================================================
# EVALUATION VALIDATION
# ============================================================

loss, accuracy = model.evaluate(X_validation, y_validation, verbose=0)

print("\n==============================")
print("RESULTAT VALIDATION")
print("==============================")
print(f"Loss validation : {loss:.4f}")
print(f"Accuracy validation : {accuracy:.4f}")
print(f"Accuracy validation : {accuracy * 100:.2f}%")


# ============================================================
# COURBE LOSS
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["loss"], label="Loss entraînement")
plt.plot(history.history["val_loss"], label="Loss validation")
plt.title("Evolution du Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# COURBE ACCURACY
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(history.history["accuracy"], label="Accuracy entraînement")
plt.plot(history.history["val_accuracy"], label="Accuracy validation")
plt.title("Evolution de l'Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ============================================================
# EVALUATION SUR LE DOSSIER TEST
# ============================================================

print("\n==============================")
print("EVALUATION DES DONNEES TEST")
print("==============================")

X_test, y_test = charger_donnees(DOSSIER_TEST)

if len(X_test) == 0:
    print("Aucun fichier de test utilisable : évaluation test ignorée.")
else:
    classes_test_inconnues = sorted(set(y_test.tolist()) - set(encoder.classes_))
    if classes_test_inconnues:
        raise SystemExit(
            "Classes de test absentes de l'entraînement : "
            + ", ".join(classes_test_inconnues)
        )

    y_test_encoded = encoder.transform(y_test)
    y_test_categorical = to_categorical(
        y_test_encoded,
        num_classes=len(encoder.classes_)
    )

    test_loss, test_accuracy = model.evaluate(
        X_test,
        y_test_categorical,
        verbose=0
    )

    print(f"\nTest Loss : {test_loss:.4f}")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

    predictions = model.predict(X_test, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)

    print("\n==============================")
    print("RAPPORT DE CLASSIFICATION")
    print("==============================")
    print(classification_report(
        y_test_encoded,
        predicted_classes,
        labels=np.arange(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))

    cm = confusion_matrix(
        y_test_encoded,
        predicted_classes,
        labels=np.arange(len(encoder.classes_))
    )

    print("\nMatrice de confusion (lignes = vraies classes, colonnes = prédites) :")
    print("Ordre des classes :", list(encoder.classes_))
    print(cm)


# ============================================================
# FIN
# Le meilleur modèle a déjà été sauvegardé par ModelCheckpoint.
# Ne pas écraser ce meilleur modèle avec la dernière époque.
# ============================================================

print(f"\nModèle sauvegardé : {MODELE}")
print("\nTerminé !")
