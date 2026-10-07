from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import os
import webbrowser
import h5py
import tensorflow as tf
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier


# ============================================================
# GLOBAL VARIABLES
# ============================================================
filename = ""
autoencoder = None
decision_tree = None
dnn = None
encoder_model = None
pca = None

X = None
Y = None
y_numeric = None
dataset = None

accuracy = []
precision = []
recall = []
fscore = []

vector = None

X_train = None
X_test = None
y_train = None
y_test = None
scaler = None

# The supplied dataset contains classes 0-7.
# The original code supplied names for only 0-6, so class 7 is
# intentionally kept as "Class 7 (Unmapped)" rather than inventing
# a threat name that is not present in the supplied project.
labels = [
    'Normal',
    'Naive Malicious Response Injection (NMRI)',
    'Complex Malicious Response Injection (CMRI)',
    'Malicious State Command Injection (MSCI)',
    'Malicious Parameter Command Injection (MPCI)',
    'Malicious Function Code Injection (MFCI)',
    'Denial of Service (DoS)',
    'Class 7 (Unmapped)'
]


# ============================================================
# GUI
# ============================================================
main = tkinter.Tk()
main.title("CyberGuard AI - AI-Powered Cyber Threat Detection System")
main.geometry("1300x1200")


def show_text(value):
    text.insert(END, str(value) + "\n")
    text.see(END)
    text.update_idletasks()


# ============================================================
# METRICS
# ============================================================
def calculateMetrics(algorithm, predict, actual):
    a = accuracy_score(actual, predict) * 100
    p = precision_score(actual, predict, average='macro', zero_division=0) * 100
    r = recall_score(actual, predict, average='macro', zero_division=0) * 100
    f = f1_score(actual, predict, average='macro', zero_division=0) * 100

    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(f)

    show_text(algorithm + " Accuracy : " + str(round(a, 2)))
    show_text(algorithm + " Precision : " + str(round(p, 2)))
    show_text(algorithm + " Recall : " + str(round(r, 2)))
    show_text(algorithm + " FScore : " + str(round(f, 2)))
    show_text("")


# ============================================================
# UPLOAD DATASET
# ============================================================
def uploadDataset():
    global filename, dataset

    text.delete('1.0', END)

    filename = filedialog.askopenfilename(
        initialdir="Dataset",
        title="Select Dataset",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not filename:
        return

    try:
        dataset = pd.read_csv(filename)

        text.insert(END, filename + " loaded\n\n")
        text.insert(END, "Dataset Values\n\n")
        text.insert(END, str(dataset.head()) + "\n\n")

        if 'result' not in dataset.columns:
            text.insert(END, "Error: 'result' column not found in dataset\n")
            return

        unique, count = np.unique(dataset['result'], return_counts=True)

        # FIX: the original code had 7 x-axis labels but the supplied
        # dataset has 8 classes, which caused the matplotlib shape error.
        class_names = []
        for value in unique:
            value = int(value)
            if value < len(labels):
                class_names.append(labels[value])
            else:
                class_names.append("Class " + str(value))

        text.insert(
            END,
            "Class distribution: " +
            str({int(k): int(v) for k, v in zip(unique, count)}) +
            "\n"
        )

        plt.figure(figsize=(12, 6))
        plt.bar(np.arange(len(unique)), count)
        plt.xticks(np.arange(len(unique)), class_names, rotation=90)
        plt.title("Various Cyber-Attacks Found in Dataset")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        text.insert(END, "Error loading dataset: " + str(e) + "\n")


# ============================================================
# PREPROCESSING
# ============================================================
def preprocessing():
    global dataset, scaler
    global X, Y, y_numeric
    global X_train, X_test, y_train, y_test

    text.delete('1.0', END)

    try:
        if not filename:
            text.insert(END, "Please upload the dataset first.\n")
            return

        dataset = pd.read_csv(filename)
        dataset.fillna(0, inplace=True)

        if 'result' not in dataset.columns:
            text.insert(END, "Error: 'result' column not found in dataset\n")
            return

        # Keep the original numeric class values from the supplied dataset.
        y_numeric = pd.to_numeric(
            dataset['result'], errors='coerce'
        ).fillna(0).astype(int).values

        # All columns except result are the 23 features.
        feature_data = dataset.drop(columns=['result'])
        feature_data = feature_data.apply(pd.to_numeric, errors='coerce').fillna(0)

        X_raw = feature_data.values.astype(np.float32)

        scaler = MinMaxScaler()
        X = scaler.fit_transform(X_raw).astype(np.float32)

        # The supplied model has 8 output classes.
        Y = tf.keras.utils.to_categorical(y_numeric, num_classes=8)

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            Y,
            test_size=0.2,
            random_state=42,
            stratify=y_numeric
        )

        text.insert(END, "Dataset after feature normalization:\n\n")
        text.insert(END, str(X[:5]) + "\n\n")
        text.insert(END, "Total records: " + str(X.shape[0]) + "\n")
        text.insert(END, "Total features: " + str(X.shape[1]) + "\n\n")
        text.insert(END, "Dataset Train-Test Split:\n")
        text.insert(END, "Train records: " + str(X_train.shape[0]) + "\n")
        text.insert(END, "Test records: " + str(X_test.shape[0]) + "\n")

    except Exception as e:
        text.insert(END, "Error in preprocessing: " + str(e) + "\n")


# ============================================================
# LOAD THE OLD KERAS 2.3.1 WEIGHTS
# ============================================================
def load_legacy_model():
    """
    The supplied encoder_model.json was created with Keras 2.3.1.
    Keras 3 cannot deserialize that old JSON directly.

    The supplied H5 file contains:
        dense_1: 23 -> 256
        dense_2: 256 -> 8

    So we recreate exactly that architecture using the current
    TensorFlow/Keras and copy the original weights into it.
    """
    # Build the exact original graph as a Functional model.
    # Do not create a Keras Model with outputs=None.
    input_layer = tf.keras.Input(shape=(23,), name='input_1')
    encoded = tf.keras.layers.Dense(
        256,
        activation='relu',
        name='dense_1'
    )(input_layer)

    output = tf.keras.layers.Dense(
        8,
        activation='softmax',
        name='dense_2'
    )(encoded)

    model = tf.keras.Model(
        inputs=input_layer,
        outputs=output,
        name='model_1'
    )

    weights_path = os.path.join(
        "model",
        "encoder_model_weights.h5"
    )

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            "model/encoder_model_weights.h5 was not found."
        )

    with h5py.File(weights_path, 'r') as h5:
        kernel1 = np.array(h5['dense_1/dense_1/kernel:0'])
        bias1 = np.array(h5['dense_1/dense_1/bias:0'])
        kernel2 = np.array(h5['dense_2/dense_2/kernel:0'])
        bias2 = np.array(h5['dense_2/dense_2/bias:0'])

    # Validate against the supplied model before loading.
    if kernel1.shape != (23, 256):
        raise ValueError(
            "dense_1 kernel shape is " + str(kernel1.shape) +
            ", expected (23, 256)"
        )

    if kernel2.shape != (256, 8):
        raise ValueError(
            "dense_2 kernel shape is " + str(kernel2.shape) +
            ", expected (256, 8)"
        )

    model.get_layer('dense_1').set_weights([kernel1, bias1])
    model.get_layer('dense_2').set_weights([kernel2, bias2])

    return model


# ============================================================
# AUTOENCODER / ORIGINAL 23 -> 256 -> 8 MODEL
# ============================================================
def runAutoEncoder():
    global autoencoder, encoder_model
    global accuracy, precision, recall, fscore

    if X is None:
        text.delete('1.0', END)
        text.insert(
            END,
            "Please Upload Dataset and run Preprocess Dataset first.\n"
        )
        return

    text.delete('1.0', END)

    # Start a new comparison run.
    accuracy = []
    precision = []
    recall = []
    fscore = []

    try:
        autoencoder = load_legacy_model()

        text.insert(
            END,
            "Original Keras 2.3.1 model weights loaded successfully.\n"
        )

        # IMPORTANT:
        # The original project calls this AutoEncoder, but the supplied
        # model is actually a 23 -> 256 -> 8 classification model.
        # We preserve that original model and name for compatibility.

        autoencoder.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # The 256-neuron hidden layer is the feature representation.
        encoder_model = tf.keras.Model(
            inputs=autoencoder.input,
            outputs=autoencoder.get_layer('dense_1').output,
            name='encoder_model'
        )

        predict = autoencoder.predict(X_test, verbose=0)
        predict = np.argmax(predict, axis=1)
        testY = np.argmax(y_test, axis=1)

        calculateMetrics(
            "AutoEncoder",
            predict,
            testY
        )

        text.insert(
            END,
            "AutoEncoder stage completed.\n"
        )
        text.insert(
            END,
            "256-dimensional encoded features are ready for "
            "Decision Tree + PCA.\n"
        )

    except Exception as e:
        text.insert(
            END,
            "AutoEncoder error: " +
            type(e).__name__ + ": " +
            str(e) + "\n"
        )


# ============================================================
# DECISION TREE + PCA
# ============================================================
def runDecisionTree():
    global decision_tree, encoder_model, vector, pca
    global X_train, X_test, y_train, y_test

    if encoder_model is None:
        text.delete('1.0', END)
        text.insert(
            END,
            "Please run AutoEncoder first.\n"
        )
        return

    if X is None:
        text.insert(
            END,
            "Please preprocess the dataset first.\n"
        )
        return

    text.delete('1.0', END)

    try:
        # FIX:
        # Extract the actual 256-dimensional hidden features,
        # not the final 8-class softmax output.
        vector = encoder_model.predict(
            X,
            verbose=0
        )

        # Keep the original project's PCA target of 7 components.
        pca = PCA(
            n_components=7,
            random_state=42
        )

        vector_pca = pca.fit_transform(vector)

        Y1 = y_numeric

        X_train_dt, X_test_dt, y_train_dt, y_test_dt = train_test_split(
            vector_pca,
            Y1,
            test_size=0.2,
            random_state=42,
            stratify=Y1
        )

        decision_tree = DecisionTreeClassifier(
            random_state=42
        )

        decision_tree.fit(
            X_train_dt,
            y_train_dt
        )

        predict = decision_tree.predict(
            X_test_dt
        )

        text.insert(
            END,
            "Decision Tree Trained on New Features Extracted "
            "from AutoEncoder\n"
        )
        text.insert(
            END,
            "Encoded features: " + str(vector.shape) + "\n"
        )
        text.insert(
            END,
            "PCA features: " + str(vector_pca.shape) + "\n"
        )
        text.insert(
            END,
            "PCA explained variance: " +
            str(round(pca.explained_variance_ratio_.sum() * 100, 2)) +
            "%\n\n"
        )

        calculateMetrics(
            "Decision Tree",
            predict,
            y_test_dt
        )

    except Exception as e:
        text.insert(
            END,
            "Decision Tree error: " +
            type(e).__name__ + ": " +
            str(e) + "\n"
        )


# ============================================================
# DNN
# ============================================================
def runDNN():
    global dnn

    if encoder_model is None or pca is None or vector is None:
        text.delete('1.0', END)
        text.insert(
            END,
            "Please run AutoEncoder and Decision Tree with PCA first.\n"
        )
        return

    text.delete('1.0', END)

    try:
        vector_pca = pca.transform(vector)

        # FIX:
        # The DNN is trained against the ORIGINAL ground-truth labels.
        # The old code trained it against Decision Tree predictions,
        # which made the DNN merely learn the tree's output.
        X_train_dnn, X_test_dnn, y_train_dnn, y_test_dnn = train_test_split(
            vector_pca,
            y_numeric,
            test_size=0.2,
            random_state=42,
            stratify=y_numeric
        )

        dnn = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation='relu',
            solver='adam',
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )

        dnn.fit(
            X_train_dnn,
            y_train_dnn
        )

        predict = dnn.predict(
            X_test_dnn
        )

        text.insert(
            END,
            "Attack Prediction using DNN\n"
        )
        text.insert(
            END,
            "DNN is trained using the original dataset labels.\n\n"
        )

        calculateMetrics(
            "DNN",
            predict,
            y_test_dnn
        )

    except Exception as e:
        text.insert(
            END,
            "DNN error: " +
            type(e).__name__ + ": " +
            str(e) + "\n"
        )


# ============================================================
# DETECTION / ATTRIBUTION
# ============================================================
def attackAttributeDetection():
    global encoder_model, dnn, pca, scaler

    if (
        encoder_model is None or
        dnn is None or
        pca is None or
        scaler is None
    ):
        text.delete('1.0', END)
        text.insert(
            END,
            "Please run AutoEncoder, Decision Tree with PCA "
            "and DNN first.\n"
        )
        return

    filename_test = filedialog.askopenfilename(
        initialdir="Dataset",
        title="Select Test Dataset",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )

    if not filename_test:
        return

    text.delete('1.0', END)

    try:
        test_dataset = pd.read_csv(filename_test)
        test_dataset.fillna(0, inplace=True)

        # testData.csv supplied with the project has only the 23
        # feature columns and no result column.
        if 'result' in test_dataset.columns:
            values = test_dataset.drop(columns=['result'])
        else:
            values = test_dataset.copy()

        values = values.apply(
            pd.to_numeric,
            errors='coerce'
        ).fillna(0)

        if values.shape[1] != X.shape[1]:
            text.insert(
                END,
                "Error: test data has " +
                str(values.shape[1]) +
                " features, but the model expects " +
                str(X.shape[1]) + ".\n"
            )
            return

        temp = scaler.transform(
            values.values.astype(np.float32)
        )

        # Extract the same 256-dimensional features.
        test_vector = encoder_model.predict(
            temp,
            verbose=0
        )

        # Apply the same PCA fitted during training.
        test_vector = pca.transform(
            test_vector
        )

        predict = dnn.predict(
            test_vector
        )

        for i in range(len(predict)):
            pred = int(predict[i])

            if pred == 0:
                text.insert(
                    END,
                    "New Test Data : " +
                    str(values.iloc[i].values) +
                    " ====> NO CYBER ATTACK DETECTED\n\n"
                )
            else:
                if pred < len(labels):
                    attack_label = labels[pred]
                else:
                    attack_label = "Class " + str(pred)

                text.insert(
                    END,
                    "New Test Data : " +
                    str(values.iloc[i].values) +
                    " ====> CYBER ATTACK DETECTED "
                    "Attribution Label: " +
                    attack_label +
                    "\n\n"
                )

    except Exception as e:
        text.insert(
            END,
            "Detection error: " +
            type(e).__name__ + ": " +
            str(e) + "\n"
        )


# ============================================================
# PERFORMANCE GRAPH
# ============================================================
def graph():
    if len(accuracy) < 3:
        text.insert(
            END,
            "\nPlease run AutoEncoder, Decision Tree with PCA and DNN first.\n"
        )
        return

    algorithms = [
        "AutoEncoder",
        "Decision Tree with PCA",
        "DNN"
    ]

    performance = pd.DataFrame({
        "Algorithm": algorithms,
        "Accuracy": accuracy[:3],
        "Precision": precision[:3],
        "Recall": recall[:3],
        "F1 Score": fscore[:3]
    })

    performance.set_index("Algorithm").plot(
        kind="bar",
        figsize=(10, 6)
    )

    plt.title("Model Performance Comparison")
    plt.xlabel("Algorithms")
    plt.ylabel("Performance (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.legend()
    plt.tight_layout()
    plt.show()
# ============================================================
# COMPARISON TABLE
# ============================================================
def comparisonTable():
    if len(accuracy) < 3:
        text.insert(
            END,
            "\nPlease run all three algorithms first.\n"
        )
        return

    output = (
        "<html><body>"
        "<h2 align=center>CyberGuard AI - AI-Powered Cyber Threat Detection System"
        "Model Comparison</h2>"
        "<table align=center border=1 cellpadding=8>"
        "<tr>"
        "<th>Algorithm Name</th>"
        "<th>Accuracy</th>"
        "<th>Precision</th>"
        "<th>Recall</th>"
        "<th>FSCORE</th>"
        "</tr>"
    )

    output += (
        "<tr><td>AutoEncoder</td>"
        "<td>" + str(round(accuracy[0], 2)) + "</td>"
        "<td>" + str(round(precision[0], 2)) + "</td>"
        "<td>" + str(round(recall[0], 2)) + "</td>"
        "<td>" + str(round(fscore[0], 2)) + "</td></tr>"
    )

    output += (
        "<tr><td>Decision Tree with PCA</td>"
        "<td>" + str(round(accuracy[1], 2)) + "</td>"
        "<td>" + str(round(precision[1], 2)) + "</td>"
        "<td>" + str(round(recall[1], 2)) + "</td>"
        "<td>" + str(round(fscore[1], 2)) + "</td></tr>"
    )

    output += (
        "<tr><td>DNN</td>"
        "<td>" + str(round(accuracy[2], 2)) + "</td>"
        "<td>" + str(round(precision[2], 2)) + "</td>"
        "<td>" + str(round(recall[2], 2)) + "</td>"
        "<td>" + str(round(fscore[2], 2)) + "</td></tr>"
    )

    output += "</table></body></html>"

    with open(
        "table.html",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(output)

    webbrowser.open(
        "table.html",
        new=2
    )


# ============================================================
# GUI ELEMENTS
# ============================================================
font = ('times', 16, 'bold')

title = Label(
    main,
    text='CyberGuard AI - AI-Powered Cyber Threat Detection System'
)
title.config(
    bg='greenyellow',
    fg='dodger blue'
)
title.config(
    font=font
)
title.config(
    height=3,
    width=120
)
title.place(
    x=0,
    y=5
)

font1 = ('times', 12, 'bold')

text = Text(
    main,
    height=20,
    width=150
)

scroll = Scrollbar(
    main,
    command=text.yview
)

text.configure(
    yscrollcommand=scroll.set
)

text.place(
    x=50,
    y=120
)

scroll.place(
    x=1330,
    y=120,
    height=330
)

text.config(
    font=font1
)

font1 = ('times', 13, 'bold')

uploadButton = Button(
    main,
    text="Upload Dataset",
    command=uploadDataset
)
uploadButton.place(
    x=50,
    y=550
)
uploadButton.config(
    font=font1
)

processButton = Button(
    main,
    text="Preprocess Dataset",
    command=preprocessing
)
processButton.place(
    x=330,
    y=550
)
processButton.config(
    font=font1
)

autoButton = Button(
    main,
    text="Run AutoEncoder Algorithm",
    command=runAutoEncoder
)
autoButton.place(
    x=630,
    y=550
)
autoButton.config(
    font=font1
)

dtButton = Button(
    main,
    text="Run Decision Tree with PCA",
    command=runDecisionTree
)
dtButton.place(
    x=920,
    y=550
)
dtButton.config(
    font=font1
)

dnnButton = Button(
    main,
    text="Run DNN Algorithm",
    command=runDNN
)
dnnButton.place(
    x=50,
    y=600
)
dnnButton.config(
    font=font1
)

attributeButton = Button(
    main,
    text="Detection & Attribute Attack Type",
    command=attackAttributeDetection
)
attributeButton.place(
    x=330,
    y=600
)
attributeButton.config(
    font=font1
)

tableButton = Button(
    main,
    text="Comparison Table",
    command=comparisonTable
)
tableButton.place(
    x=630,
    y=600
)
tableButton.config(
    font=font1
)

graphButton = Button(
    main,
    text="Performance Graph",
    command=graph
)
graphButton.place(
    x=850,
    y=600
)
graphButton.config(
    font=font1
)

main.config(
    bg='LightSkyBlue'
)

main.mainloop()
