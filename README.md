{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyP4cJOnWByzFtgtxEY0DyOa",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/yatha1910/Hand-Written-Digit-Prediction/blob/main/Copy_of_main_project.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **Hand Written Digit Prediction - Classification Analysis**"
      ],
      "metadata": {
        "id": "eoWZeZFE0XxH"
      }
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 1. Import Libraries"
      ],
      "metadata": {
        "id": "p08QuNt2y7W1"
      }
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "99g2oLjVx3Gd"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.metrics import classification_report, accuracy_score\n",
        "from tensorflow.keras.datasets import mnist\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, Flatten\n"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 2. Load and Preprocess Data python Copy code New Section"
      ],
      "metadata": {
        "id": "waNRn3Svy7uQ"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Load MNIST data\n",
        "(X_train, y_train), (X_test, y_test) = mnist.load_data()\n",
        "\n",
        "# Scale data\n",
        "scaler = StandardScaler()\n",
        "X_train = X_train.reshape(-1, 28*28) / 255.0\n",
        "X_test = X_test.reshape(-1, 28*28) / 255.0\n"
      ],
      "metadata": {
        "collapsed": true,
        "id": "ob0OqnRex36O"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 3. Build and Compile Model"
      ],
      "metadata": {
        "id": "U-6OKH_Ay8Nn"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "model = Sequential([\n",
        "    Flatten(input_shape=(28*28,)),\n",
        "    Dense(128, activation='relu'),\n",
        "    Dense(64, activation='relu'),\n",
        "    Dense(10, activation='softmax')\n",
        "])\n",
        "\n",
        "model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9y2Jm_dGx57C",
        "outputId": "ae79d7e0-49b0-4a89-b4f8-ce94806efee0"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/layers/reshaping/flatten.py:37: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(**kwargs)\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 4. Train Model"
      ],
      "metadata": {
        "id": "jQCQWz04y9LM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "history = model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=32)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ZWDs8f4ix6mI",
        "outputId": "ebc81c2f-4aca-4f27-97bd-518a63c99d87"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Epoch 1/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 4ms/step - accuracy: 0.8663 - loss: 0.4621 - val_accuracy: 0.9516 - val_loss: 0.1623\n",
            "Epoch 2/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m11s\u001b[0m 4ms/step - accuracy: 0.9611 - loss: 0.1254 - val_accuracy: 0.9668 - val_loss: 0.1096\n",
            "Epoch 3/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m5s\u001b[0m 3ms/step - accuracy: 0.9760 - loss: 0.0772 - val_accuracy: 0.9703 - val_loss: 0.0975\n",
            "Epoch 4/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m6s\u001b[0m 4ms/step - accuracy: 0.9832 - loss: 0.0548 - val_accuracy: 0.9732 - val_loss: 0.0945\n",
            "Epoch 5/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 3ms/step - accuracy: 0.9863 - loss: 0.0426 - val_accuracy: 0.9729 - val_loss: 0.0941\n",
            "Epoch 6/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 4ms/step - accuracy: 0.9893 - loss: 0.0335 - val_accuracy: 0.9758 - val_loss: 0.0902\n",
            "Epoch 7/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 4ms/step - accuracy: 0.9913 - loss: 0.0281 - val_accuracy: 0.9775 - val_loss: 0.0903\n",
            "Epoch 8/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m9s\u001b[0m 3ms/step - accuracy: 0.9916 - loss: 0.0239 - val_accuracy: 0.9751 - val_loss: 0.1046\n",
            "Epoch 9/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 4ms/step - accuracy: 0.9938 - loss: 0.0195 - val_accuracy: 0.9741 - val_loss: 0.1062\n",
            "Epoch 10/10\n",
            "\u001b[1m1500/1500\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 3ms/step - accuracy: 0.9941 - loss: 0.0165 - val_accuracy: 0.9697 - val_loss: 0.1437\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# 5. Evaluate Model"
      ],
      "metadata": {
        "id": "U2rb_LnazK_5"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "y_pred = np.argmax(model.predict(X_test), axis=1)\n",
        "print(classification_report(y_test, y_pred))\n",
        "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "oxnhGybkx6ir",
        "outputId": "d100a7b7-75c9-48bd-db62-4ffef44f0a42"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m313/313\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 1ms/step\n",
            "              precision    recall  f1-score   support\n",
            "\n",
            "           0       0.99      0.98      0.99       980\n",
            "           1       0.99      0.99      0.99      1135\n",
            "           2       0.94      0.99      0.96      1032\n",
            "           3       0.94      0.97      0.96      1010\n",
            "           4       0.99      0.96      0.97       982\n",
            "           5       0.99      0.95      0.97       892\n",
            "           6       0.98      0.99      0.98       958\n",
            "           7       0.96      0.98      0.97      1028\n",
            "           8       0.96      0.96      0.96       974\n",
            "           9       0.98      0.94      0.96      1009\n",
            "\n",
            "    accuracy                           0.97     10000\n",
            "   macro avg       0.97      0.97      0.97     10000\n",
            "weighted avg       0.97      0.97      0.97     10000\n",
            "\n",
            "Accuracy: 0.9709\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "# **This model uses the MNIST dataset for digit recognition, applying scaling, feature extraction, and a neural network to classify digits.**"
      ],
      "metadata": {
        "id": "_OjKDLL5zEAL"
      }
    }
  ]
}
