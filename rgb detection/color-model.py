{
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QV5v-4SWWbVZ"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.svm import SVC\n",
        "from tensorflow.keras.applications import VGG16\n",
        "from tensorflow.keras.preprocessing import image\n",
        "from tensorflow.keras.models import Model\n",
        "from tensorflow.keras.layers import GlobalAveragePooling2D\n",
        "\n",
        "# Load pre-trained CNN model for feature extraction\n",
        "base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))\n",
        "\n",
        "# Remove the reshaping step for now\n",
        "\n",
        "# Extract features using the CNN model\n",
        "features = []\n",
        "for img in resized_images:\n",
        "    img = image.img_to_array(img)\n",
        "    img = np.expand_dims(img, axis=0)\n",
        "    features.append(base_model.predict(img))\n",
        "features = np.vstack(features)\n",
        "\n",
        "# Split the extracted features into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(features, resized_labels, test_size=0.2, random_state=42)\n",
        "\n",
        "# Flatten the extracted features\n",
        "X_train_flattened = X_train.reshape(X_train.shape[0], -1)\n",
        "X_test_flattened = X_test.reshape(X_test.shape[0], -1)\n",
        "\n",
        "# Train a Support Vector Machine (SVM) classifier\n",
        "svm = SVC()\n",
        "svm.fit(X_train_flattened, y_train)\n",
        "\n",
        "# Predict labels for the test set\n",
        "y_pred = svm.predict(X_test_flattened)"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "codemirror_mode": {
        "name": "ipython",
        "version": 3
      },
      "file_extension": ".py",
      "mimetype": "text/x-python",
      "name": "python",
      "nbconvert_exporter": "python",
      "pygments_lexer": "ipython3",
      "version": "3.10.2"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
