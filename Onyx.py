{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/CarolineKwawu/KwawuComputationalGroup/blob/main/Onyx.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "VCd_t5Tkz04U",
        "outputId": "9f47ad9a-c03a-49cb-b6e9-1df07840b48d"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Mounted at /content/gdrive\n"
          ]
        }
      ],
      "source": [
        "from google.colab import drive\n",
        "drive.mount('/content/gdrive')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "NqFmsYKh8yLm"
      },
      "outputs": [],
      "source": [
        "import os\n",
        "import warnings\n",
        "\n",
        "# Tools\n",
        "\n",
        "import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)\n",
        "import numpy as np  # linear algebra\n",
        "from itertools import cycle\n",
        "import matplotlib.pyplot as plt\n",
        "from sklearn import decomposition, datasets\n",
        "\n",
        "# Models\n",
        "from sklearn import svm, linear_model\n",
        "from xgboost import XGBRegressor, plot_importance\n",
        "from sklearn.linear_model import LinearRegression, TheilSenRegressor\n",
        "from sklearn.ensemble import RandomForestRegressor\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from xgboost import XGBRegressor, XGBClassifier\n",
        "from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier\n",
        "from sklearn.svm import SVR, SVC\n",
        "\n",
        "\n",
        "\n",
        "\n",
        "# Utilities\n",
        "from sklearn.model_selection import train_test_split\n",
        "import joblib\n",
        "from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OuAYwQfF7-ma"
      },
      "outputs": [],
      "source": [
        "from sklearn.metrics import accuracy_score, precision_score"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IL02uvf7rBh2"
      },
      "outputs": [],
      "source": [
        "#zero_band_gap_rows = data[data['band_gaps'] <= 1]\n",
        "#data_without_zero = data[data['band_gaps'] >= 1]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "AnbPbVFtrV2p"
      },
      "outputs": [],
      "source": [
        "#zero_band_gap_rows.to_csv('/content/gdrive/MyDrive/Descriptor/data/interaction_data_zero.csv', index=False)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "sml_W1xgrioU"
      },
      "outputs": [],
      "source": [
        "#data_without_zero.to_csv('/content/gdrive/MyDrive/Descriptor/data/interaction_data_re.csv', index=False)"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "r34jMptFC6Hd"
      },
      "source": [
        "# Load and preprocess the data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "HVPixOva9W_J"
      },
      "outputs": [],
      "source": [
        "# data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/lattice_data2.csv')"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/lattice_data2.csv')\n",
        "# data.head()"
      ],
      "metadata": {
        "id": "_oaJX9mXs3Ra"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/lattice_data2.csv')\n",
        "# data.head()"
      ],
      "metadata": {
        "id": "THJdLtZ0tjrm"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "df = pd.read_csv(\"/content/sample_data/california_housing_test.csv\")\n",
        "df.head()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 206
        },
        "id": "yeG5aWJmt2Xi",
        "outputId": "8f235f3a-ec55-4b75-afc9-0dec0d8523a9"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   longitude  latitude  housing_median_age  total_rooms  total_bedrooms  \\\n",
              "0    -122.05     37.37                27.0       3885.0           661.0   \n",
              "1    -118.30     34.26                43.0       1510.0           310.0   \n",
              "2    -117.81     33.78                27.0       3589.0           507.0   \n",
              "3    -118.36     33.82                28.0         67.0            15.0   \n",
              "4    -119.67     36.33                19.0       1241.0           244.0   \n",
              "\n",
              "   population  households  median_income  median_house_value  \n",
              "0      1537.0       606.0         6.6085            344700.0  \n",
              "1       809.0       277.0         3.5990            176500.0  \n",
              "2      1484.0       495.0         5.7934            270500.0  \n",
              "3        49.0        11.0         6.1359            330000.0  \n",
              "4       850.0       237.0         2.9375             81700.0  "
            ],
            "text/html": [
              "\n",
              "  <div id=\"df-cc6c0f3a-d43d-4856-aacc-f514a30213f6\" class=\"colab-df-container\">\n",
              "    <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>longitude</th>\n",
              "      <th>latitude</th>\n",
              "      <th>housing_median_age</th>\n",
              "      <th>total_rooms</th>\n",
              "      <th>total_bedrooms</th>\n",
              "      <th>population</th>\n",
              "      <th>households</th>\n",
              "      <th>median_income</th>\n",
              "      <th>median_house_value</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>-122.05</td>\n",
              "      <td>37.37</td>\n",
              "      <td>27.0</td>\n",
              "      <td>3885.0</td>\n",
              "      <td>661.0</td>\n",
              "      <td>1537.0</td>\n",
              "      <td>606.0</td>\n",
              "      <td>6.6085</td>\n",
              "      <td>344700.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>-118.30</td>\n",
              "      <td>34.26</td>\n",
              "      <td>43.0</td>\n",
              "      <td>1510.0</td>\n",
              "      <td>310.0</td>\n",
              "      <td>809.0</td>\n",
              "      <td>277.0</td>\n",
              "      <td>3.5990</td>\n",
              "      <td>176500.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>-117.81</td>\n",
              "      <td>33.78</td>\n",
              "      <td>27.0</td>\n",
              "      <td>3589.0</td>\n",
              "      <td>507.0</td>\n",
              "      <td>1484.0</td>\n",
              "      <td>495.0</td>\n",
              "      <td>5.7934</td>\n",
              "      <td>270500.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>-118.36</td>\n",
              "      <td>33.82</td>\n",
              "      <td>28.0</td>\n",
              "      <td>67.0</td>\n",
              "      <td>15.0</td>\n",
              "      <td>49.0</td>\n",
              "      <td>11.0</td>\n",
              "      <td>6.1359</td>\n",
              "      <td>330000.0</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>-119.67</td>\n",
              "      <td>36.33</td>\n",
              "      <td>19.0</td>\n",
              "      <td>1241.0</td>\n",
              "      <td>244.0</td>\n",
              "      <td>850.0</td>\n",
              "      <td>237.0</td>\n",
              "      <td>2.9375</td>\n",
              "      <td>81700.0</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "    <div class=\"colab-df-buttons\">\n",
              "\n",
              "  <div class=\"colab-df-container\">\n",
              "    <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-cc6c0f3a-d43d-4856-aacc-f514a30213f6')\"\n",
              "            title=\"Convert this dataframe to an interactive table.\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\" viewBox=\"0 -960 960 960\">\n",
              "    <path d=\"M120-120v-720h720v720H120Zm60-500h600v-160H180v160Zm220 220h160v-160H400v160Zm0 220h160v-160H400v160ZM180-400h160v-160H180v160Zm440 0h160v-160H620v160ZM180-180h160v-160H180v160Zm440 0h160v-160H620v160Z\"/>\n",
              "  </svg>\n",
              "    </button>\n",
              "\n",
              "  <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    .colab-df-buttons div {\n",
              "      margin-bottom: 4px;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "    <script>\n",
              "      const buttonEl =\n",
              "        document.querySelector('#df-cc6c0f3a-d43d-4856-aacc-f514a30213f6 button.colab-df-convert');\n",
              "      buttonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "      async function convertToInteractive(key) {\n",
              "        const element = document.querySelector('#df-cc6c0f3a-d43d-4856-aacc-f514a30213f6');\n",
              "        const dataTable =\n",
              "          await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                    [key], {});\n",
              "        if (!dataTable) return;\n",
              "\n",
              "        const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "          '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "          + ' to learn more about interactive tables.';\n",
              "        element.innerHTML = '';\n",
              "        dataTable['output_type'] = 'display_data';\n",
              "        await google.colab.output.renderOutput(dataTable, element);\n",
              "        const docLink = document.createElement('div');\n",
              "        docLink.innerHTML = docLinkHtml;\n",
              "        element.appendChild(docLink);\n",
              "      }\n",
              "    </script>\n",
              "  </div>\n",
              "\n",
              "\n",
              "<div id=\"df-cf5d308f-20dc-46a3-96df-89a65364e539\">\n",
              "  <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-cf5d308f-20dc-46a3-96df-89a65364e539')\"\n",
              "            title=\"Suggest charts\"\n",
              "            style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "  </button>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "      --bg-color: #E8F0FE;\n",
              "      --fill-color: #1967D2;\n",
              "      --hover-bg-color: #E2EBFA;\n",
              "      --hover-fill-color: #174EA6;\n",
              "      --disabled-fill-color: #AAA;\n",
              "      --disabled-bg-color: #DDD;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "      --bg-color: #3B4455;\n",
              "      --fill-color: #D2E3FC;\n",
              "      --hover-bg-color: #434B5C;\n",
              "      --hover-fill-color: #FFFFFF;\n",
              "      --disabled-bg-color: #3B4455;\n",
              "      --disabled-fill-color: #666;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart {\n",
              "    background-color: var(--bg-color);\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: var(--fill-color);\n",
              "    height: 32px;\n",
              "    padding: 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: var(--hover-bg-color);\n",
              "    box-shadow: 0 1px 2px rgba(60, 64, 67, 0.3), 0 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: var(--button-hover-fill-color);\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart-complete:disabled,\n",
              "  .colab-df-quickchart-complete:disabled:hover {\n",
              "    background-color: var(--disabled-bg-color);\n",
              "    fill: var(--disabled-fill-color);\n",
              "    box-shadow: none;\n",
              "  }\n",
              "\n",
              "  .colab-df-spinner {\n",
              "    border: 2px solid var(--fill-color);\n",
              "    border-color: transparent;\n",
              "    border-bottom-color: var(--fill-color);\n",
              "    animation:\n",
              "      spin 1s steps(1) infinite;\n",
              "  }\n",
              "\n",
              "  @keyframes spin {\n",
              "    0% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "      border-left-color: var(--fill-color);\n",
              "    }\n",
              "    20% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    30% {\n",
              "      border-color: transparent;\n",
              "      border-left-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    40% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-top-color: var(--fill-color);\n",
              "    }\n",
              "    60% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "    }\n",
              "    80% {\n",
              "      border-color: transparent;\n",
              "      border-right-color: var(--fill-color);\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "    90% {\n",
              "      border-color: transparent;\n",
              "      border-bottom-color: var(--fill-color);\n",
              "    }\n",
              "  }\n",
              "</style>\n",
              "\n",
              "  <script>\n",
              "    async function quickchart(key) {\n",
              "      const quickchartButtonEl =\n",
              "        document.querySelector('#' + key + ' button');\n",
              "      quickchartButtonEl.disabled = true;  // To prevent multiple clicks.\n",
              "      quickchartButtonEl.classList.add('colab-df-spinner');\n",
              "      try {\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      } catch (error) {\n",
              "        console.error('Error during call to suggestCharts:', error);\n",
              "      }\n",
              "      quickchartButtonEl.classList.remove('colab-df-spinner');\n",
              "      quickchartButtonEl.classList.add('colab-df-quickchart-complete');\n",
              "    }\n",
              "    (() => {\n",
              "      let quickchartButtonEl =\n",
              "        document.querySelector('#df-cf5d308f-20dc-46a3-96df-89a65364e539 button');\n",
              "      quickchartButtonEl.style.display =\n",
              "        google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "    })();\n",
              "  </script>\n",
              "</div>\n",
              "\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 4
        }
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7azoRCgP96he"
      },
      "outputs": [],
      "source": [
        "y = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/E_above_hull/e_above_hull.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 81
        },
        "id": "pEJvBVdUCpzs",
        "outputId": "6f618353-9dab-4305-e5c7-9de6df5dbbad"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "                 formula  Energy_above_hull\n",
              "0  {'Zr': 1.0, 'B': 6.0}           0.402292"
            ],
            "text/html": [
              "\n",
              "\n",
              "  <div id=\"df-42356825-2417-4aff-a7f3-9761d314943c\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>formula</th>\n",
              "      <th>Energy_above_hull</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>{'Zr': 1.0, 'B': 6.0}</td>\n",
              "      <td>0.402292</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-42356825-2417-4aff-a7f3-9761d314943c')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "\n",
              "\n",
              "\n",
              "    <div id=\"df-61505b8b-f089-47fd-9762-8ae5f4ca6978\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-61505b8b-f089-47fd-9762-8ae5f4ca6978')\"\n",
              "              title=\"Suggest charts.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "    </div>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "    <script>\n",
              "      async function quickchart(key) {\n",
              "        const containerElement = document.querySelector('#' + key);\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      }\n",
              "    </script>\n",
              "\n",
              "      <script>\n",
              "\n",
              "function displayQuickchartButton(domScope) {\n",
              "  let quickchartButtonEl =\n",
              "    domScope.querySelector('#df-61505b8b-f089-47fd-9762-8ae5f4ca6978 button.colab-df-quickchart');\n",
              "  quickchartButtonEl.style.display =\n",
              "    google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "}\n",
              "\n",
              "        displayQuickchartButton(document);\n",
              "      </script>\n",
              "      <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-42356825-2417-4aff-a7f3-9761d314943c button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-42356825-2417-4aff-a7f3-9761d314943c');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 6
        }
      ],
      "source": [
        "y.head(1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "GjQKD8LrLrGT"
      },
      "outputs": [],
      "source": [
        "y1 = y.drop('formula', axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hbjMoVEcMQeQ"
      },
      "outputs": [],
      "source": [
        "datax = data.drop('band_gaps', axis=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EZhLP9V3MDTy"
      },
      "outputs": [],
      "source": [
        "datax['Energy'] = y1['Energy_above_hull']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 456
        },
        "id": "Svip9wlLM1SI",
        "outputId": "ba815416-1842-499f-b012-6990a99c048f"
      },
      "outputs": [
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "   Unnamed: 0.17  Unnamed: 0.16  Unnamed: 0.15  Unnamed: 0.14  Unnamed: 0.13  \\\n",
              "0              0            0.0            0.0            0.0            0.0   \n",
              "1              1            1.0            1.0            1.0            1.0   \n",
              "2              2            2.0            2.0            2.0            2.0   \n",
              "3              3            3.0            3.0            3.0            3.0   \n",
              "4              4            4.0            4.0            4.0            4.0   \n",
              "\n",
              "   Unnamed: 0.12  Unnamed: 0.11  Unnamed: 0.10  Unnamed: 0.9  Unnamed: 0.8  \\\n",
              "0            0.0            0.0            0.0           0.0           0.0   \n",
              "1            1.0            1.0            1.0           1.0           1.0   \n",
              "2            2.0            2.0            2.0           2.0           2.0   \n",
              "3            3.0            3.0            3.0           3.0           3.0   \n",
              "4            4.0            4.0            4.0           4.0           4.0   \n",
              "\n",
              "   ...           592           593           594       595           596  \\\n",
              "0  ...  2.415085e-04  1.438905e-01  6.559436e+00  0.015358  1.505887e-13   \n",
              "1  ...  0.000000e+00  1.073819e+01  2.159403e-14  0.008020  0.000000e+00   \n",
              "2  ...  0.000000e+00  0.000000e+00  9.187543e-51  0.000029  0.000000e+00   \n",
              "3  ...  0.000000e+00  1.332571e-32  7.314057e-03  0.000005  0.000000e+00   \n",
              "4  ...  1.894346e-34  3.948782e+00  6.068417e-01  0.001145  2.562839e-42   \n",
              "\n",
              "            597           598        599                           formula  \\\n",
              "0  5.129394e-07  4.563977e-02  13.434057             {'Zr': 1.0, 'B': 6.0}   \n",
              "1  5.199947e-03  2.364922e-14   6.004678             {'Si': 1.0, 'C': 1.0}   \n",
              "2  0.000000e+00  9.858132e-53   1.705136   {'H': 1.0, 'Pb': 1.0, 'I': 3.0}   \n",
              "3  2.779720e-38  9.476679e-01   0.677029            {'Br': 1.0, 'Cl': 1.0}   \n",
              "4  1.366910e-03  3.870909e+00   2.342516  {'Eu': 1.0, 'Fe': 1.0, 'O': 3.0}   \n",
              "\n",
              "     Energy  \n",
              "0  0.402292  \n",
              "1  0.742811  \n",
              "2  0.660889  \n",
              "3  0.000677  \n",
              "4  0.038578  \n",
              "\n",
              "[5 rows x 620 columns]"
            ],
            "text/html": [
              "\n",
              "\n",
              "  <div id=\"df-f6e852b4-2b0f-4431-9f92-58adec4df804\">\n",
              "    <div class=\"colab-df-container\">\n",
              "      <div>\n",
              "<style scoped>\n",
              "    .dataframe tbody tr th:only-of-type {\n",
              "        vertical-align: middle;\n",
              "    }\n",
              "\n",
              "    .dataframe tbody tr th {\n",
              "        vertical-align: top;\n",
              "    }\n",
              "\n",
              "    .dataframe thead th {\n",
              "        text-align: right;\n",
              "    }\n",
              "</style>\n",
              "<table border=\"1\" class=\"dataframe\">\n",
              "  <thead>\n",
              "    <tr style=\"text-align: right;\">\n",
              "      <th></th>\n",
              "      <th>Unnamed: 0.17</th>\n",
              "      <th>Unnamed: 0.16</th>\n",
              "      <th>Unnamed: 0.15</th>\n",
              "      <th>Unnamed: 0.14</th>\n",
              "      <th>Unnamed: 0.13</th>\n",
              "      <th>Unnamed: 0.12</th>\n",
              "      <th>Unnamed: 0.11</th>\n",
              "      <th>Unnamed: 0.10</th>\n",
              "      <th>Unnamed: 0.9</th>\n",
              "      <th>Unnamed: 0.8</th>\n",
              "      <th>...</th>\n",
              "      <th>592</th>\n",
              "      <th>593</th>\n",
              "      <th>594</th>\n",
              "      <th>595</th>\n",
              "      <th>596</th>\n",
              "      <th>597</th>\n",
              "      <th>598</th>\n",
              "      <th>599</th>\n",
              "      <th>formula</th>\n",
              "      <th>Energy</th>\n",
              "    </tr>\n",
              "  </thead>\n",
              "  <tbody>\n",
              "    <tr>\n",
              "      <th>0</th>\n",
              "      <td>0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>0.0</td>\n",
              "      <td>...</td>\n",
              "      <td>2.415085e-04</td>\n",
              "      <td>1.438905e-01</td>\n",
              "      <td>6.559436e+00</td>\n",
              "      <td>0.015358</td>\n",
              "      <td>1.505887e-13</td>\n",
              "      <td>5.129394e-07</td>\n",
              "      <td>4.563977e-02</td>\n",
              "      <td>13.434057</td>\n",
              "      <td>{'Zr': 1.0, 'B': 6.0}</td>\n",
              "      <td>0.402292</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>1</th>\n",
              "      <td>1</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>1.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>1.073819e+01</td>\n",
              "      <td>2.159403e-14</td>\n",
              "      <td>0.008020</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>5.199947e-03</td>\n",
              "      <td>2.364922e-14</td>\n",
              "      <td>6.004678</td>\n",
              "      <td>{'Si': 1.0, 'C': 1.0}</td>\n",
              "      <td>0.742811</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>2</th>\n",
              "      <td>2</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>2.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>9.187543e-51</td>\n",
              "      <td>0.000029</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>9.858132e-53</td>\n",
              "      <td>1.705136</td>\n",
              "      <td>{'H': 1.0, 'Pb': 1.0, 'I': 3.0}</td>\n",
              "      <td>0.660889</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>3</th>\n",
              "      <td>3</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>3.0</td>\n",
              "      <td>...</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>1.332571e-32</td>\n",
              "      <td>7.314057e-03</td>\n",
              "      <td>0.000005</td>\n",
              "      <td>0.000000e+00</td>\n",
              "      <td>2.779720e-38</td>\n",
              "      <td>9.476679e-01</td>\n",
              "      <td>0.677029</td>\n",
              "      <td>{'Br': 1.0, 'Cl': 1.0}</td>\n",
              "      <td>0.000677</td>\n",
              "    </tr>\n",
              "    <tr>\n",
              "      <th>4</th>\n",
              "      <td>4</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>4.0</td>\n",
              "      <td>...</td>\n",
              "      <td>1.894346e-34</td>\n",
              "      <td>3.948782e+00</td>\n",
              "      <td>6.068417e-01</td>\n",
              "      <td>0.001145</td>\n",
              "      <td>2.562839e-42</td>\n",
              "      <td>1.366910e-03</td>\n",
              "      <td>3.870909e+00</td>\n",
              "      <td>2.342516</td>\n",
              "      <td>{'Eu': 1.0, 'Fe': 1.0, 'O': 3.0}</td>\n",
              "      <td>0.038578</td>\n",
              "    </tr>\n",
              "  </tbody>\n",
              "</table>\n",
              "<p>5 rows × 620 columns</p>\n",
              "</div>\n",
              "      <button class=\"colab-df-convert\" onclick=\"convertToInteractive('df-f6e852b4-2b0f-4431-9f92-58adec4df804')\"\n",
              "              title=\"Convert this dataframe to an interactive table.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "  <svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "       width=\"24px\">\n",
              "    <path d=\"M0 0h24v24H0V0z\" fill=\"none\"/>\n",
              "    <path d=\"M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z\"/><path d=\"M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z\"/>\n",
              "  </svg>\n",
              "      </button>\n",
              "\n",
              "\n",
              "\n",
              "    <div id=\"df-b91086b0-76ec-42dc-96ae-9ced9bba3fcf\">\n",
              "      <button class=\"colab-df-quickchart\" onclick=\"quickchart('df-b91086b0-76ec-42dc-96ae-9ced9bba3fcf')\"\n",
              "              title=\"Suggest charts.\"\n",
              "              style=\"display:none;\">\n",
              "\n",
              "<svg xmlns=\"http://www.w3.org/2000/svg\" height=\"24px\"viewBox=\"0 0 24 24\"\n",
              "     width=\"24px\">\n",
              "    <g>\n",
              "        <path d=\"M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zM9 17H7v-7h2v7zm4 0h-2V7h2v10zm4 0h-2v-4h2v4z\"/>\n",
              "    </g>\n",
              "</svg>\n",
              "      </button>\n",
              "    </div>\n",
              "\n",
              "<style>\n",
              "  .colab-df-quickchart {\n",
              "    background-color: #E8F0FE;\n",
              "    border: none;\n",
              "    border-radius: 50%;\n",
              "    cursor: pointer;\n",
              "    display: none;\n",
              "    fill: #1967D2;\n",
              "    height: 32px;\n",
              "    padding: 0 0 0 0;\n",
              "    width: 32px;\n",
              "  }\n",
              "\n",
              "  .colab-df-quickchart:hover {\n",
              "    background-color: #E2EBFA;\n",
              "    box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "    fill: #174EA6;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart {\n",
              "    background-color: #3B4455;\n",
              "    fill: #D2E3FC;\n",
              "  }\n",
              "\n",
              "  [theme=dark] .colab-df-quickchart:hover {\n",
              "    background-color: #434B5C;\n",
              "    box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "    filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "    fill: #FFFFFF;\n",
              "  }\n",
              "</style>\n",
              "\n",
              "    <script>\n",
              "      async function quickchart(key) {\n",
              "        const containerElement = document.querySelector('#' + key);\n",
              "        const charts = await google.colab.kernel.invokeFunction(\n",
              "            'suggestCharts', [key], {});\n",
              "      }\n",
              "    </script>\n",
              "\n",
              "      <script>\n",
              "\n",
              "function displayQuickchartButton(domScope) {\n",
              "  let quickchartButtonEl =\n",
              "    domScope.querySelector('#df-b91086b0-76ec-42dc-96ae-9ced9bba3fcf button.colab-df-quickchart');\n",
              "  quickchartButtonEl.style.display =\n",
              "    google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "}\n",
              "\n",
              "        displayQuickchartButton(document);\n",
              "      </script>\n",
              "      <style>\n",
              "    .colab-df-container {\n",
              "      display:flex;\n",
              "      flex-wrap:wrap;\n",
              "      gap: 12px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert {\n",
              "      background-color: #E8F0FE;\n",
              "      border: none;\n",
              "      border-radius: 50%;\n",
              "      cursor: pointer;\n",
              "      display: none;\n",
              "      fill: #1967D2;\n",
              "      height: 32px;\n",
              "      padding: 0 0 0 0;\n",
              "      width: 32px;\n",
              "    }\n",
              "\n",
              "    .colab-df-convert:hover {\n",
              "      background-color: #E2EBFA;\n",
              "      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);\n",
              "      fill: #174EA6;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert {\n",
              "      background-color: #3B4455;\n",
              "      fill: #D2E3FC;\n",
              "    }\n",
              "\n",
              "    [theme=dark] .colab-df-convert:hover {\n",
              "      background-color: #434B5C;\n",
              "      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);\n",
              "      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));\n",
              "      fill: #FFFFFF;\n",
              "    }\n",
              "  </style>\n",
              "\n",
              "      <script>\n",
              "        const buttonEl =\n",
              "          document.querySelector('#df-f6e852b4-2b0f-4431-9f92-58adec4df804 button.colab-df-convert');\n",
              "        buttonEl.style.display =\n",
              "          google.colab.kernel.accessAllowed ? 'block' : 'none';\n",
              "\n",
              "        async function convertToInteractive(key) {\n",
              "          const element = document.querySelector('#df-f6e852b4-2b0f-4431-9f92-58adec4df804');\n",
              "          const dataTable =\n",
              "            await google.colab.kernel.invokeFunction('convertToInteractive',\n",
              "                                                     [key], {});\n",
              "          if (!dataTable) return;\n",
              "\n",
              "          const docLinkHtml = 'Like what you see? Visit the ' +\n",
              "            '<a target=\"_blank\" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'\n",
              "            + ' to learn more about interactive tables.';\n",
              "          element.innerHTML = '';\n",
              "          dataTable['output_type'] = 'display_data';\n",
              "          await google.colab.output.renderOutput(dataTable, element);\n",
              "          const docLink = document.createElement('div');\n",
              "          docLink.innerHTML = docLinkHtml;\n",
              "          element.appendChild(docLink);\n",
              "        }\n",
              "      </script>\n",
              "    </div>\n",
              "  </div>\n"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ],
      "source": [
        "datax.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "zkAd4WtoGaLw"
      },
      "outputs": [],
      "source": [
        "no_oxy_data = data_without_oxygen\n",
        "oxy_data = data_with_oxygen\n",
        "li_data = data_with_li\n",
        "p_data = data_with_p\n",
        "mn_data = data_with_mn\n",
        "new_data = data_without_specific_elements"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dZ75Hur0Ivss"
      },
      "outputs": [],
      "source": [
        "\n",
        "# Save the filtered data to separate CSV files\n",
        "data_with_oxygen.to_csv(\"/content/gdrive/MyDrive/Descriptor/data/crystals_with_oxygen.csv\", index=False)\n",
        "data_with_li.to_csv(\"/content/gdrive/MyDrive/Descriptor/data/band_gaps/crystals_with_li.csv\", index=False)\n",
        "data_with_mn.to_csv(\"/content/gdrive/MyDrive/Descriptor/data/band_gaps/crystals_with_mn.csv\", index=False)\n",
        "data_with_p.to_csv(\"/content/gdrive/MyDrive/Descriptor/data/band_gaps/crystals_with_p.csv\", index=False)\n",
        "data_without_specific_elements.to_csv(\"/content/gdrive/MyDrive/Descriptor/data/band_gaps/crystals_without_specific_elements.csv\", index=False)\n",
        "data_without_oxygen.to_csv(\"/content/gdrive/MyDrive/Descriptor/data/band_gaps/crystals_without_oxygen.csv\", index=False)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "72Eg_21_UmRU"
      },
      "outputs": [],
      "source": [
        "oxy_data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/formationE/symmetry G/crystals_with_oxygen.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vAiYVaYgFcSs"
      },
      "outputs": [],
      "source": [
        "no_oxy_data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/formationE/symmetry G/crystals_without_oxygen.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "qs_8VUxEb9qX"
      },
      "outputs": [],
      "source": [
        "li_data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/formationE/symmetry G/crystals_with_li.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "aFlZ_YAGhzrH"
      },
      "outputs": [],
      "source": [
        "mn_data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/formationE/symmetry G/crystals_with_mn.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "v6eoFrAKiEN-"
      },
      "outputs": [],
      "source": [
        "p_data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/formationE/symmetry G/crystals_with_p.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8WUiYB2F9RwY"
      },
      "outputs": [],
      "source": [
        "new_data = pd.read_csv('/content/gdrive/MyDrive/Descriptor/data/formationE/symmetry G/crystals_without_specific_elements.csv')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LG5aFder_Oru"
      },
      "outputs": [],
      "source": [
        "#data = data_without_zero"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "2Ge91fffOtJA"
      },
      "outputs": [],
      "source": [
        "set_data = [data,oxy_data,no_oxy_data,li_data,mn_data,p_data,new_data]\n",
        "for d in set_data:\n",
        "  for c in d.columns:\n",
        "    if 'Unnamed' in c:\n",
        "      print(c)\n",
        "      del d[c]\n",
        "      print(d.head(1))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0289lywljQ0n"
      },
      "outputs": [],
      "source": [
        "oxy_y = oxy_data.pop('Energy')\n",
        "no_oxy_y = no_oxy_data.pop('Energy')\n",
        "li_y = li_data.pop('Energy')\n",
        "mn_y = mn_data.pop('Energy')\n",
        "p_y = p_data.pop('Energy')\n",
        "new_y = new_data.pop('Energy')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "m431TZMs_GxX"
      },
      "outputs": [],
      "source": [
        "formula, formula2,formula3,formula4,formula5,formula6,formula7,formula8 = data.pop('formula'),y.pop('formula'),oxy_data.pop('formula'),no_oxy_data.pop('formula'),li_data.pop('formula'),mn_data.pop('formula'),p_data.pop('formula'),new_data.pop('formula')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_mf9ckZr-GRt",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "761a3eab-d40e-4bcf-eaef-b1203b3d041e"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "<ipython-input-16-65c7d161699d>:4: SettingWithCopyWarning: \n",
            "A value is trying to be set on a copy of a slice from a DataFrame\n",
            "\n",
            "See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy\n",
            "  d = d.fillna(0, inplace=True)\n"
          ]
        }
      ],
      "source": [
        "set_y =[oxy_y,no_oxy_y,li_y,mn_y,p_y,new_y,y]\n",
        "\n",
        "for d in set_y:\n",
        "  d = d.fillna(0, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZV8oiZqP9udD"
      },
      "outputs": [],
      "source": [
        "x = data.drop('band_gaps', axis=1)\n",
        "oxy_x = oxy_data\n",
        "no_oxy_x = no_oxy_data\n",
        "li_x = li_data\n",
        "mn_x = mn_data\n",
        "p_x = p_data\n",
        "new_x = new_data"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yEy0Nh3SpO5p"
      },
      "outputs": [],
      "source": [
        "set_x = [x,oxy_x,no_oxy_x,li_x,mn_x,p_x,new_x]\n",
        "for xs in set_x:\n",
        "  xs.fillna(value=0, inplace=True)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1Vtyx9bEGRwj"
      },
      "outputs": [],
      "source": [
        "set_x = [x,oxy_x,no_oxy_x,li_x,mn_x,p_x,new_x]\n",
        "for xs in set_x:\n",
        "  print(xs.head())"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "rOJYOI5gBngK"
      },
      "outputs": [],
      "source": [
        "scaler = StandardScaler().fit(x)\n",
        "set_x = [x,oxy_x,no_oxy_x,li_x,mn_x,p_x,new_x]\n",
        "for xs in set_x:\n",
        "  xs = scaler.transform(xs)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "eaQnEJlPCFhZ"
      },
      "outputs": [],
      "source": [
        "X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(\n",
        "    x, y, test_size=.2, random_state=100)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bmy4r_JNCHSo"
      },
      "outputs": [],
      "source": [
        "X_train_scaled=pd.DataFrame(X_train_scaled)\n",
        "X_test_scaled=pd.DataFrame(X_test_scaled)\n",
        "y_train=pd.DataFrame(y_train)\n",
        "y_test=pd.DataFrame(y_test)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "BlVJWSAroht6"
      },
      "outputs": [],
      "source": [
        "set_x = [oxy_x,no_oxy_x,li_x,mn_x,p_x,new_x]\n",
        "for xs in set_x:\n",
        "  xs = pd.DataFrame(xs)\n",
        "\n",
        "set_y =[oxy_y,no_oxy_y,li_y,mn_y,p_y,new_y]\n",
        "for ys in set_y:\n",
        "  ys = pd.DataFrame(ys)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "BXjr-CV5tYH4"
      },
      "outputs": [],
      "source": [
        "y_train.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LRnKEr9sXccN"
      },
      "outputs": [],
      "source": [
        "y_train = y_train.pop('Energy_above_hull')\n",
        "y_test = y_test.pop('Energy_above_hull')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "i5wceuy7CSqT"
      },
      "outputs": [],
      "source": [
        "\"\"\"X_train_scaled.to_csv(\"/content/gdrive/MyDrive/Descriptor/data_frame/X_train_scaled.csv\")\n",
        "X_test_scaled.to_csv(\"/content/gdrive/MyDrive/Descriptor/data_frame/X_test_scaled.csv\")\n",
        "y_train.to_csv(\"/content/gdrive/MyDrive/Descriptor/data_frame/y_train.csv\")\n",
        "y_test.to_csv(\"/content/gdrive/MyDrive/Descriptor/data_frame/y_test.csv\")\"\"\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Zl6tGTCyGwKd"
      },
      "outputs": [],
      "source": [
        "\"\"\", 'SVR'\n",
        "   SVR(kernel='rbf', epsilon=0.1),\"\"\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4Jmu7QyoDf3i"
      },
      "outputs": [],
      "source": [
        "regr_names = ['XGBoostR', 'RFR']\n",
        "regr_objects = [\n",
        "    XGBRegressor(objective='reg:linear', colsample_bytree=0.3, learning_rate=0.1,\n",
        "                  max_depth=400, alpha=10, n_estimators=400),\n",
        "    RandomForestRegressor(n_estimators=400, max_depth=400, random_state=0),\n",
        "]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "RBTyfts-wBGE"
      },
      "outputs": [],
      "source": [
        "ele = ['O']"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "yT-NgEvJDuXT"
      },
      "outputs": [],
      "source": [
        "for regr_choice in ele: #range(2):\n",
        "    #regr = regr_objects[regr_choice]\n",
        "    #regr_name = regr_names[regr_choice]\n",
        "    regr = RandomForestRegressor(n_estimators=400, max_depth=400, random_state=0)\n",
        "    regr_name = 'RFR'\n",
        "    regr.fit(X_train_scaled, y_train)\n",
        "\n",
        "   #1 Training\n",
        "    y_predicted = regr.predict(X_train_scaled)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/train_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(y_train, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(y_train, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = y_train\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ko')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_train', bbox_inches='tight')\n",
        "\n",
        "\n",
        "\n",
        "    #2 Test\n",
        "    y_predicted = regr.predict(X_test_scaled)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/test_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(y_test, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(y_test, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = y_test\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ko')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_test', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    #3 Oxygen\n",
        "    y_predicted = regr.predict(oxy_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/oxygen_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(oxy_y, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(oxy_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = oxy_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_oxygen', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    #4 No_Oxygen\n",
        "    y_predicted = regr.predict(no_oxy_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/no_oxygen_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(no_oxy_y, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(no_oxy_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = no_oxy_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'go')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_no_oxygen', bbox_inches='tight')\n",
        "\n",
        "\n",
        "\n",
        "    #5 Lithium\n",
        "    y_predicted = regr.predict(li_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/Lithium_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(li_y, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(li_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = li_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'bo')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_Lithium', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    #6 Manganese\n",
        "    y_predicted = regr.predict(mn_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/Manganese_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(mn_y, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(mn_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = mn_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'yo')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_Manganese', bbox_inches='tight')\n",
        "\n",
        "\n",
        "\n",
        "    #7 Phosphorus\n",
        "    y_predicted = regr.predict(p_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/Phosphorus_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(p_y, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(p_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = p_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'mo')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_Phosphorus', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    #8 Without specific elements\n",
        "    y_predicted = regr.predict(new_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/New_Data_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(new_y, y_predicted)))+'\\n')\n",
        "    errors_file.write('r2\\t'+str(r2_score(new_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = new_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'co')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_New_Data', bbox_inches='tight')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6ke8K8RA-xJZ"
      },
      "outputs": [],
      "source": [
        "y_predicted = regr.predict(new_x)\n",
        "\n",
        "errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/no_specific_elements_'+regr_name+'.txt', 'w')\n",
        "errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(new_y, y_predicted)))+'\\n')\n",
        "errors_file.write('r2\\t'+str(r2_score(new_y, y_predicted))+'\\n')\n",
        "errors_file.close()\n",
        "\n",
        "xPlot = new_y\n",
        "yPlot = y_predicted\n",
        "plt.figure(figsize=(10, 10))\n",
        "plt.plot(xPlot, yPlot, 'ro')\n",
        "plt.plot(xPlot, xPlot)\n",
        "plt.ylabel(regr_name)\n",
        "plt.xlabel('DFT')\n",
        "plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_no_specific_elements', bbox_inches='tight')\n",
        "\n",
        "\n",
        "\n",
        "y_predicted = regr.predict(no_oxy_x)\n",
        "\n",
        "errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/no_oxygen_'+regr_name+'.txt', 'w')\n",
        "errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(no_oxy_y, y_predicted)))+'\\n')\n",
        "errors_file.write('r2\\t'+str(r2_score(no_oxy_y, y_predicted))+'\\n')\n",
        "errors_file.close()\n",
        "\n",
        "xPlot = no_oxy_y\n",
        "yPlot = y_predicted\n",
        "plt.figure(figsize=(10, 10))\n",
        "plt.plot(xPlot, yPlot, 'ro')\n",
        "plt.plot(xPlot, xPlot)\n",
        "plt.ylabel(regr_name)\n",
        "plt.xlabel('DFT')\n",
        "plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_no_oxygen', bbox_inches='tight')\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aoOz2h8hH7WV",
        "outputId": "ec38c592-c739-46f7-c098-ac126819b40d"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "101068\n",
            "101068\n"
          ]
        }
      ],
      "source": [
        "print(len(y_predicted))\n",
        "print(len(y_train))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "R7pd9X5VlC4D"
      },
      "outputs": [],
      "source": [
        "threshold = 1.5\n",
        "y_train = (y_train > threshold).astype(int)\n",
        "y_test = (y_test > threshold).astype(int)\n",
        "oxy_y = (oxy_y > threshold).astype(int)\n",
        "no_oxy_y = (no_oxy_y > threshold).astype(int)\n",
        "li_y = (li_y > threshold).astype(int)\n",
        "mn_y = (mn_y > threshold).astype(int)\n",
        "p_y = (p_y > threshold).astype(int)\n",
        "new_y = (new_y > threshold).astype(int)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "m6P_Ea7565SQ"
      },
      "outputs": [],
      "source": [
        "\"\"\"'SVC'\n",
        ",\n",
        "    SVC(kernel='rbf', gamma='auto'),\n",
        "\"\"\"\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "dJy5Xf0qmFMb"
      },
      "outputs": [],
      "source": [
        "regr_names2 = ['XGBoostC', 'RFC']\n",
        "regr_objects2 = [\n",
        "    XGBClassifier(objective='binary:logistic', colsample_bytree=0.3, learning_rate=0.1,\n",
        "                   max_depth=400, alpha=10, n_estimators=400),\n",
        "    RandomForestClassifier(n_estimators=400, max_depth=400, random_state=0)]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 382
        },
        "id": "7QRrgKHck8x0",
        "outputId": "d5d38a93-ed03-40f8-bb82-b301a4e7ec93"
      },
      "outputs": [
        {
          "ename": "KeyboardInterrupt",
          "evalue": "ignored",
          "output_type": "error",
          "traceback": [
            "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m                         Traceback (most recent call last)",
            "\u001b[0;32m<ipython-input-67-a3fe99f00863>\u001b[0m in \u001b[0;36m<cell line: 1>\u001b[0;34m()\u001b[0m\n\u001b[1;32m      2\u001b[0m     \u001b[0mregr\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mregr_objects2\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mregr_choice\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      3\u001b[0m     \u001b[0mregr_name\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0mregr_names2\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mregr_choice\u001b[0m\u001b[0;34m]\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m----> 4\u001b[0;31m     \u001b[0mregr\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mfit\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mX_train_scaled\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0my_train\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m      5\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m      6\u001b[0m    \u001b[0;31m#1 afasdfasfdasfdasfdasfd\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/xgboost/core.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    618\u001b[0m             \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    619\u001b[0m                 \u001b[0mkwargs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mk\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0marg\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 620\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    621\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    622\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/xgboost/sklearn.py\u001b[0m in \u001b[0;36mfit\u001b[0;34m(self, X, y, sample_weight, base_margin, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, base_margin_eval_set, feature_weights, callbacks)\u001b[0m\n\u001b[1;32m   1488\u001b[0m             )\n\u001b[1;32m   1489\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1490\u001b[0;31m             self._Booster = train(\n\u001b[0m\u001b[1;32m   1491\u001b[0m                 \u001b[0mparams\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1492\u001b[0m                 \u001b[0mtrain_dmatrix\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/xgboost/core.py\u001b[0m in \u001b[0;36minner_f\u001b[0;34m(*args, **kwargs)\u001b[0m\n\u001b[1;32m    618\u001b[0m             \u001b[0;32mfor\u001b[0m \u001b[0mk\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0marg\u001b[0m \u001b[0;32min\u001b[0m \u001b[0mzip\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0msig\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mparameters\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0margs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    619\u001b[0m                 \u001b[0mkwargs\u001b[0m\u001b[0;34m[\u001b[0m\u001b[0mk\u001b[0m\u001b[0;34m]\u001b[0m \u001b[0;34m=\u001b[0m \u001b[0marg\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 620\u001b[0;31m             \u001b[0;32mreturn\u001b[0m \u001b[0mfunc\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0;34m**\u001b[0m\u001b[0mkwargs\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    621\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    622\u001b[0m         \u001b[0;32mreturn\u001b[0m \u001b[0minner_f\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/xgboost/training.py\u001b[0m in \u001b[0;36mtrain\u001b[0;34m(params, dtrain, num_boost_round, evals, obj, feval, maximize, early_stopping_rounds, evals_result, verbose_eval, xgb_model, callbacks, custom_metric)\u001b[0m\n\u001b[1;32m    183\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcb_container\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mbefore_iteration\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbst\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtrain\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevals\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    184\u001b[0m             \u001b[0;32mbreak\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m--> 185\u001b[0;31m         \u001b[0mbst\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mupdate\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mdtrain\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mobj\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0m\u001b[1;32m    186\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mcb_container\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mafter_iteration\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0mbst\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mi\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mdtrain\u001b[0m\u001b[0;34m,\u001b[0m \u001b[0mevals\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m    187\u001b[0m             \u001b[0;32mbreak\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n",
            "\u001b[0;32m/usr/local/lib/python3.10/dist-packages/xgboost/core.py\u001b[0m in \u001b[0;36mupdate\u001b[0;34m(self, dtrain, iteration, fobj)\u001b[0m\n\u001b[1;32m   1916\u001b[0m \u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1917\u001b[0m         \u001b[0;32mif\u001b[0m \u001b[0mfobj\u001b[0m \u001b[0;32mis\u001b[0m \u001b[0;32mNone\u001b[0m\u001b[0;34m:\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[0;32m-> 1918\u001b[0;31m             _check_call(_LIB.XGBoosterUpdateOneIter(self.handle,\n\u001b[0m\u001b[1;32m   1919\u001b[0m                                                     \u001b[0mctypes\u001b[0m\u001b[0;34m.\u001b[0m\u001b[0mc_int\u001b[0m\u001b[0;34m(\u001b[0m\u001b[0miteration\u001b[0m\u001b[0;34m)\u001b[0m\u001b[0;34m,\u001b[0m\u001b[0;34m\u001b[0m\u001b[0;34m\u001b[0m\u001b[0m\n\u001b[1;32m   1920\u001b[0m                                                     dtrain.handle))\n",
            "\u001b[0;31mKeyboardInterrupt\u001b[0m: "
          ]
        }
      ],
      "source": [
        "for regr_choice in range(2):\n",
        "    regr = regr_objects2[regr_choice]\n",
        "    regr_name = regr_names2[regr_choice]\n",
        "    regr.fit(X_train_scaled, y_train)\n",
        "\n",
        "   #1 afasdfasfdasfdasfdasfd\n",
        "    y_predicted = regr.predict(X_train_scaled)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/train_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'accuracy\\t'+str(accuracy_score(y_train, y_predicted))+'\\n')\n",
        "    errors_file.write('precision\\t'+str(precision_score(y_train, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = y_train\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_train', bbox_inches='tight')\n",
        "\n",
        "\n",
        "\n",
        "    #2 ajfdskaskjfdakjshfdkjaskfdaksfdlk\n",
        "    y_predicted = regr.predict(X_test_scaled)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/test_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'accuracy\\t'+str(accuracy_score(y_test, y_predicted))+'\\n')\n",
        "    errors_file.write('precision\\t'+str(precision_score(y_test, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = y_test\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_test', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    #3 ajkfdhsfdkjsakjhfdkajshkfdjhaksjfdla\n",
        "    y_predicted = regr.predict(oxy_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/oxygen_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'accuracy\\t'+str(accuracy_score(oxy_y, y_predicted))+'\\n')\n",
        "    errors_file.write('precision\\t'+str(precision_score(oxy_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = oxy_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_oxygen', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    #4 ajfkakjsfljaskfdjhaksjhfdkasdf\n",
        "    y_predicted = regr.predict(no_oxy_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/no_oxygen_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'accuracy\\t'+str(accuracy_score(no_oxy_y, y_predicted))+'\\n')\n",
        "    errors_file.write('precision\\t'+str(precision_score(no_oxy_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = no_oxy_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_no_oxygen', bbox_inches='tight')\n",
        "\n",
        "\n",
        "\n",
        "    #4 afdskjalsjfdlkajsldfkjlajdsflk;asd\n",
        "    y_predicted = regr.predict(li_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/Lithium_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'accuracy\\t'+str(accuracy_score(li_y, y_predicted))+'\\n')\n",
        "    errors_file.write('precision\\t'+str(precision_score(li_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = li_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_Lithium', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    #5BAUFADSFAISFIUAHSDFHAKSDKASDKJF\n",
        "    y_predicted = regr.predict(mn_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/Manganese_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'accuracy\\t'+str(accuracy_score(mn_y, y_predicted))+'\\n')\n",
        "    errors_file.write('precision\\t'+str(precision_score(mn_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = mn_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_Manganese', bbox_inches='tight')\n",
        "\n",
        "\n",
        "    y_predicted = regr.predict(p_x)\n",
        "\n",
        "    errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/Phosphorus_'+regr_name+'.txt', 'w')\n",
        "    errors_file.write(\n",
        "        'accuracy\\t'+str(accuracy_score(p_y, y_predicted))+'\\n')\n",
        "    errors_file.write('precision\\t'+str(precision_score(p_y, y_predicted))+'\\n')\n",
        "    errors_file.close()\n",
        "\n",
        "    xPlot = p_y\n",
        "    yPlot = y_predicted\n",
        "    plt.figure(figsize=(10, 10))\n",
        "    plt.plot(xPlot, yPlot, 'ro')\n",
        "    plt.plot(xPlot, xPlot)\n",
        "    plt.ylabel(regr_name)\n",
        "    plt.xlabel('DFT')\n",
        "    plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'_Phosphorus', bbox_inches='tight')\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Lc8iuR3QkP3g",
        "outputId": "cf2de6cc-f318-4465-921c-8113684961a8"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "0.9883601722315979\n"
          ]
        }
      ],
      "source": [
        "\n",
        "y_predicted = regr.predict(no_oxy_x)\n",
        "print(accuracy_score(no_oxy_y,y_predicted))\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-1BeWqzThy4W",
        "outputId": "1752c181-5b1b-4ac2-b342-5e30231bc198"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "0.9884632128863735\n",
            "0.9805259029364619\n"
          ]
        }
      ],
      "source": [
        "y_predicted = regr.predict(X_train_scaled)\n",
        "print(accuracy_score(y_train, y_predicted))\n",
        "print(precision_score(y_train, y_predicted))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 867
        },
        "id": "2y66f9-5iyRy",
        "outputId": "d3caf908-de78-454c-9bf7-c06c60dc185e"
      },
      "outputs": [
        {
          "data": {
            "text/plain": [
              "Text(0.5, 0, 'DFT')"
            ]
          },
          "execution_count": 59,
          "metadata": {},
          "output_type": "execute_result"
        },
        {
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 1000x1000 with 1 Axes>"
            ]
          },
          "metadata": {},
          "output_type": "display_data"
        }
      ],
      "source": [
        "xPlot = y_train\n",
        "yPlot = y_predicted\n",
        "plt.figure(figsize=(10, 10))\n",
        "plt.plot(xPlot, yPlot, 'ro')\n",
        "plt.plot(xPlot, xPlot)\n",
        "plt.ylabel(regr_name)\n",
        "plt.xlabel('DFT')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "PMAAG-E_Rcbv",
        "outputId": "becd3e6a-3966-4a45-92ec-84b1edbbc496"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "0.19681788951451706\n"
          ]
        }
      ],
      "source": [
        "print(str(np.sqrt(mean_squared_error(y_train, y_predicted))))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "0udS8AJQmF0w"
      },
      "outputs": [],
      "source": [
        "regr_name = 'SVM'\n",
        "regr =  svm.SVR(kernel='rbf', epsilon=0.1, verbose=True)\n",
        "regr.fit(X_train_scaled, y_train)\n",
        "\n",
        "y_predicted = regr.predict(X_test_scaled)\n",
        "\n",
        "print(mean_squared_error(y_test, y_predicted))\n",
        "print(r2_score(y_test, y_predicted))\n",
        "\n",
        "errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'Test_Analysis.txt', 'w')\n",
        "errors_file.write('RMSE\\t'+str(np.sqrt(mean_squared_error(y_test, y_predicted)))+'\\n')\n",
        "errors_file.write('r2\\t'+str(r2_score(y_test, y_predicted))+'\\n')\n",
        "errors_file.close()\n",
        "\n",
        "xPlot = y_test\n",
        "yPlot = y_predicted\n",
        "plt.figure(figsize=(10, 10))\n",
        "plt.plot(xPlot, yPlot, 'ro')\n",
        "plt.plot(xPlot, xPlot)\n",
        "plt.ylabel(regr_name)\n",
        "plt.xlabel('DFT')\n",
        "plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'Correlation_Test', bbox_inches='tight')\n",
        "\n",
        "y_predicted = regr.predict(X_train_scaled)\n",
        "\n",
        "print(mean_squared_error(y_train, y_predicted))\n",
        "print(r2_score(y_train, y_predicted))\n",
        "\n",
        "errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'Train_Analysis.txt', 'w')\n",
        "errors_file.write('RMSE\\t'+str(np.sqrt(mean_squared_error(y_train, y_predicted)))+'\\n')\n",
        "errors_file.write('r2\\t'+str(r2_score(y_train, y_predicted))+'\\n')\n",
        "errors_file.close()\n",
        "\n",
        "xPlot = y_train\n",
        "yPlot = y_predicted\n",
        "plt.figure(figsize=(10, 10))\n",
        "plt.plot(xPlot, yPlot, 'ro')\n",
        "plt.plot(xPlot, xPlot)\n",
        "plt.ylabel(regr_name)\n",
        "plt.xlabel('DFT')\n",
        "plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+regr_name+'Correlation_Train', bbox_inches='tight')\n"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "POzXSe7GUInq"
      },
      "source": [
        "# **Testing to see which discriptor gives the best correlation**"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true
        },
        "id": "-bl9SGFTRthv"
      },
      "outputs": [],
      "source": [
        "i = 1\n",
        "X_train_scaled = X_train_scaled[i].values.reshape(-1,1)\n",
        "X_test_scaled = X_test_scaled[i].values.reshape(-1,1)\n",
        "model = RandomForestRegressor(n_estimators=2, max_depth=100, random_state=0)\n",
        "\n",
        "model.fit(X_train_scaled, y_train)\n",
        "\n",
        "y_predicted = model.predict(X_test_scaled)\n",
        "\n",
        "print(mean_squared_error(y_test, y_predicted))\n",
        "print(r2_score(y_test, y_predicted))\n",
        "\n",
        "errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/'+nu+'Test_Analysis.txt', 'w')\n",
        "errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(y_test, y_predicted)))+'\\n')\n",
        "errors_file.write('r2\\t'+str(r2_score(y_test, y_predicted))+'\\n')\n",
        "errors_file.close()\n",
        "\n",
        "xPlot = y_test\n",
        "yPlot = y_predicted\n",
        "plt.figure(figsize=(10, 10))\n",
        "plt.plot(xPlot, yPlot, 'ro')\n",
        "plt.plot(xPlot, xPlot)\n",
        "plt.ylabel('descriptor '+nu+'coloumn')\n",
        "plt.xlabel('DFT')\n",
        "plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+nu+'Correlation_Test', bbox_inches='tight')\n",
        "\n",
        "y_predicted = model.predict(X_train_scaled)\n",
        "\n",
        "print(mean_squared_error(y_train, y_predicted))\n",
        "print(r2_score(y_train, y_predicted))\n",
        "\n",
        "errors_file = open('/content/gdrive/MyDrive/Descriptor/data/models/'+nu+'Train_Analysis.txt', 'w')\n",
        "errors_file.write(\n",
        "        'RMSE\\t'+str(np.sqrt(mean_squared_error(y_train, y_predicted)))+'\\n')\n",
        "errors_file.write('r2\\t'+str(r2_score(y_train, y_predicted))+'\\n')\n",
        "errors_file.close()\n",
        "\n",
        "xPlot = y_train\n",
        "yPlot = y_predicted\n",
        "plt.figure(figsize=(10, 10))\n",
        "plt.plot(xPlot, yPlot, 'ro')\n",
        "plt.plot(xPlot, xPlot)\n",
        "plt.ylabel('descriptor '+nu+'coloumn')\n",
        "plt.xlabel('DFT')\n",
        "plt.savefig('/content/gdrive/MyDrive/Descriptor/data/models/'+nu+'Correlation_Train', bbox_inches='tight')"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "He_73H92ElrW"
      },
      "outputs": [],
      "source": [
        "import pandas as pd\n",
        "\n",
        "# Define the elements to filter\n",
        "elements_to_filter = ['O', 'Li', 'Mn', 'P']\n",
        "\n",
        "# Filter the data to include only crystals containing specified elements\n",
        "data_with_oxygen = datax[datax[\"formula\"].apply(lambda formula: 'O' in formula)]\n",
        "data_with_li = datax[datax[\"formula\"].apply(lambda formula: 'Li' in formula)]\n",
        "data_with_mn = datax[datax[\"formula\"].apply(lambda formula: 'Mn' in formula)]\n",
        "data_with_p = datax[datax[\"formula\"].apply(lambda formula: 'P' in formula)]\n",
        "\n",
        "# Filter the data to remove crystals containing specified elements\n",
        "data_without_oxygen = datax[~datax[\"formula\"].apply(lambda formula: 'O' in formula)]\n",
        "data_without_specific_elements = data_without_oxygen[~data_without_oxygen[\"formula\"].apply(lambda formula: any(element in formula for element in elements_to_filter))]\n"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}