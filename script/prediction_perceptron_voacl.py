import os
import numpy as np
import librosa
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

# Chemin conservé comme demandé : tu pourras le rectifier plus tard.
MODELE = "/modele/modele_recon_perso.h5"

SAMPLE_RATE = 16000
DUREE = 5
N_MFCC = 40

# Doit correspondre exactement à l'ordre utilisé pendant l'entraînement.
# Avec LabelEncoder, l'ordre est généralement alphabétique.
CLASSES = [
    "Autres",
    "Manda",
    "autresperso"
]

# Fichier audio à reconnaître : remplace ce chemin par ton fichier.
fichier_audio = "/Test/test1.ma4"


# ============================================================
# EXTRACTION MFCC
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
            raise ValueError("Le fichier audio est vide ou illisible.")

        longueur = SAMPLE_RATE * DUREE

        if len(audio) < longueur:
            audio = np.pad(audio, (0, longueur - len(audio)))
        else:
            audio = audio[:longueur]

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)

        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=N_MFCC
        )

        mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-8)
        mfcc = mfcc.flatten()

        if not np.all(np.isfinite(mfcc)):
            raise ValueError("Les caractéristiques MFCC contiennent des valeurs invalides.")

        # Ajouter la dimension batch attendue par le réseau.
        return np.expand_dims(mfcc.astype(np.float32), axis=0)

    except Exception as e:
        raise RuntimeError(
            f"Impossible de traiter le fichier audio '{fichier_audio}'. "
            f"Vérifie le chemin, le format et le décodeur audio. Détail : {e}"
        ) from e


# ============================================================
# CHARGEMENT DU MODELE
# ============================================================

print("Chargement du modèle...")

if not os.path.isfile(MODELE):
    raise FileNotFoundError(
        f"Modèle introuvable : {MODELE}\n"
        "Vérifie le chemin MODELE dans la configuration."
    )

model = tf.keras.models.load_model(MODELE)

print("Modèle chargé !")


# ============================================================
# VERIFICATION DU FICHIER AUDIO
# ============================================================

if not os.path.isfile(fichier_audio):
    raise FileNotFoundError(
        f"Fichier audio introuvable : {fichier_audio}\n"
        "Modifie la variable fichier_audio pour indiquer le bon chemin."
    )

taille_mio = os.path.getsize(fichier_audio) / (1024 * 1024)
if taille_mio > 1:
    print(f"Attention : le fichier audio fait {taille_mio:.2f} Mio, donc plus de 1 Mo.")

print(f"Fichier audio : {fichier_audio}")
print(f"Taille : {taille_mio * 1024:.1f} Ko")


# ============================================================
# EXTRACTION
# ============================================================

X = extraire_mfcc(fichier_audio)

# Vérifier la compatibilité entre les caractéristiques et le modèle.
taille_attendue = model.input_shape[-1]
if X.shape[1] != taille_attendue:
    raise ValueError(
        f"Dimension MFCC incompatible : le fichier produit {X.shape[1]} "
        f"caractéristiques, mais le modèle attend {taille_attendue}. "
        "Utilise les mêmes paramètres MFCC que pendant l'entraînement."
    )


# ============================================================
# PREDICTION
# ============================================================

prediction = model.predict(X, verbose=0)
probabilites = prediction[0]

if len(probabilites) != len(CLASSES):
    raise ValueError(
        f"Le modèle retourne {len(probabilites)} sorties, "
        f"mais {len(CLASSES)} classes sont déclarées."
    )

classe_index = int(np.argmax(probabilites))
classe = CLASSES[classe_index]
confiance = float(probabilites[classe_index]) * 100


# ============================================================
# RESULTAT
# ============================================================

print("\n==============================")
print("RECONNAISSANCE VOCALE")
print("==============================")

print(f"\nVoix reconnue : {classe}")
print(f"Confiance du modèle : {confiance:.2f}%")

print("\nProbabilités :")
for i, nom_classe in enumerate(CLASSES):
    print(f"{nom_classe} : {probabilites[i] * 100:.2f}%")

# Avertissement informatif : un score softmax élevé n'est pas une certitude.
if confiance < 60:
    print(
        "\nAttention : la confiance est relativement faible. "
        "Essaie un enregistrement plus clair, avec moins de bruit."
    )
