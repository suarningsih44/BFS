{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyNwUDBjc2MDwvDO+G+t2R3W",
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
        "<a href=\"https://colab.research.google.com/github/suarningsih44/BFS/blob/main/Truck.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 8,
      "metadata": {
        "id": "6EMAoKBw2ngU"
      },
      "outputs": [],
      "source": [
        "import numpy as np\n",
        "import random\n",
        "import matplotlib.pyplot as plt"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Inisialisasi lokasi outlet\n",
        "num_locations = 50\n",
        "num_trucks = 10\n",
        "locations = np.random.rand(num_locations, 2) * 100  # 50 lokasi acak dalam area 100x100\n",
        "\n",
        "# Fungsi jarak Euclidean\n",
        "def distance(a, b):\n",
        "    return np.linalg.norm(a - b)\n",
        "\n",
        "# Total jarak rute untuk satu truk\n",
        "def route_distance(route):\n",
        "    dist = 0\n",
        "    for i in range(len(route) - 1):\n",
        "        dist += distance(locations[route[i]], locations[route[i+1]])\n",
        "    return dist"
      ],
      "metadata": {
        "id": "F1R4hYFS3vcX"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Fitness = total jarak semua truk (semakin kecil, semakin baik)\n",
        "def fitness(chromosome):\n",
        "    total_dist = 0\n",
        "    for truck_route in chromosome:\n",
        "        if len(truck_route) > 1:\n",
        "            total_dist += route_distance(truck_route)\n",
        "    return 1 / total_dist\n",
        "\n",
        "# Inisialisasi populasi\n",
        "def create_chromosome():\n",
        "    indices = list(range(num_locations))\n",
        "    random.shuffle(indices)\n",
        "    split = np.array_split(indices, num_trucks)\n",
        "    return [list(s) for s in split]\n",
        "\n",
        "def create_population(size=30):\n",
        "    return [create_chromosome() for _ in range(size)]\n",
        "\n",
        "# Seleksi: Tournament\n",
        "def selection(pop):\n",
        "    return max(random.sample(pop, 5), key=fitness)\n",
        "\n",
        "# Crossover: tukar satu rute antar truk\n",
        "def crossover(p1, p2):\n",
        "    child = []\n",
        "    all_points = set()\n",
        "    for t1, t2 in zip(p1, p2):\n",
        "        if random.random() < 0.5:\n",
        "            route = [x for x in t1 if x not in all_points]\n",
        "        else:\n",
        "            route = [x for x in t2 if x not in all_points]\n",
        "        all_points.update(route)\n",
        "        child.append(route)\n",
        "    return child"
      ],
      "metadata": {
        "id": "zzv8SmxJ37k_"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Mutasi: tukar dua lokasi dalam satu rute\n",
        "def mutate(chromo, rate=0.1):\n",
        "    for route in chromo:\n",
        "        if random.random() < rate and len(route) > 1:\n",
        "            i, j = random.sample(range(len(route)), 2)\n",
        "            route[i], route[j] = route[j], route[i]\n",
        "    return chromo\n",
        "\n",
        "# Genetic Algorithm main loop\n",
        "population = create_population()\n",
        "generations = 100\n",
        "best_fit = 0\n",
        "best_solution = None\n",
        "\n",
        "for gen in range(generations):\n",
        "    new_population = []\n",
        "    for _ in range(len(population)):\n",
        "        parent1 = selection(population)\n",
        "        parent2 = selection(population)\n",
        "        child = crossover(parent1, parent2)\n",
        "        child = mutate(child)\n",
        "        new_population.append(child)\n",
        "\n",
        "        fit = fitness(child)\n",
        "        if fit > best_fit:\n",
        "            best_fit = fit\n",
        "            best_solution = child\n",
        "\n",
        "    population = new_population\n",
        "    print(f\"Gen {gen+1}: Best Fitness = {best_fit:.4f}\")\n",
        "\n",
        "# Visualisasi rute terbaik\n",
        "colors = plt.cm.tab10(np.linspace(0, 1, num_trucks))\n",
        "for idx, route in enumerate(best_solution):\n",
        "    points = locations[route]\n",
        "    plt.plot(points[:, 0], points[:, 1], marker='o', color=colors[idx])\n",
        "plt.title(\"Visualisasi Rute Terbaik (10 Truk - 50 Lokasi)\")\n",
        "plt.xlabel(\"X\")\n",
        "plt.ylabel(\"Y\")\n",
        "plt.grid(True)\n",
        "plt.show()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "s1LQXm9A38_0",
        "outputId": "ef20ef2d-23c0-4f0d-fea9-f173d249dd50"
      },
      "execution_count": 11,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Gen 1: Best Fitness = 0.0009\n",
            "Gen 2: Best Fitness = 0.0015\n",
            "Gen 3: Best Fitness = 0.0018\n",
            "Gen 4: Best Fitness = 0.0023\n",
            "Gen 5: Best Fitness = 0.0035\n",
            "Gen 6: Best Fitness = 0.0036\n",
            "Gen 7: Best Fitness = 0.0043\n",
            "Gen 8: Best Fitness = 0.0047\n",
            "Gen 9: Best Fitness = 0.0047\n",
            "Gen 10: Best Fitness = 0.0047\n",
            "Gen 11: Best Fitness = 0.0047\n",
            "Gen 12: Best Fitness = 0.0052\n",
            "Gen 13: Best Fitness = 0.0052\n",
            "Gen 14: Best Fitness = 0.0053\n",
            "Gen 15: Best Fitness = 0.0053\n",
            "Gen 16: Best Fitness = 0.0053\n",
            "Gen 17: Best Fitness = 0.0053\n",
            "Gen 18: Best Fitness = 0.0053\n",
            "Gen 19: Best Fitness = 0.0053\n",
            "Gen 20: Best Fitness = 0.0053\n",
            "Gen 21: Best Fitness = 0.0053\n",
            "Gen 22: Best Fitness = 0.0053\n",
            "Gen 23: Best Fitness = 0.0053\n",
            "Gen 24: Best Fitness = 0.0053\n",
            "Gen 25: Best Fitness = 0.0053\n",
            "Gen 26: Best Fitness = 0.0053\n",
            "Gen 27: Best Fitness = 0.0053\n",
            "Gen 28: Best Fitness = 0.0053\n",
            "Gen 29: Best Fitness = 0.0053\n",
            "Gen 30: Best Fitness = 0.0053\n",
            "Gen 31: Best Fitness = 0.0053\n",
            "Gen 32: Best Fitness = 0.0053\n",
            "Gen 33: Best Fitness = 0.0053\n",
            "Gen 34: Best Fitness = 0.0053\n",
            "Gen 35: Best Fitness = 0.0053\n",
            "Gen 36: Best Fitness = 0.0053\n",
            "Gen 37: Best Fitness = 0.0053\n",
            "Gen 38: Best Fitness = 0.0053\n",
            "Gen 39: Best Fitness = 0.0053\n",
            "Gen 40: Best Fitness = 0.0053\n",
            "Gen 41: Best Fitness = 0.0053\n",
            "Gen 42: Best Fitness = 0.0053\n",
            "Gen 43: Best Fitness = 0.0053\n",
            "Gen 44: Best Fitness = 0.0053\n",
            "Gen 45: Best Fitness = 0.0053\n",
            "Gen 46: Best Fitness = 0.0053\n",
            "Gen 47: Best Fitness = 0.0053\n",
            "Gen 48: Best Fitness = 0.0053\n",
            "Gen 49: Best Fitness = 0.0053\n",
            "Gen 50: Best Fitness = 0.0053\n",
            "Gen 51: Best Fitness = 0.0053\n",
            "Gen 52: Best Fitness = 0.0053\n",
            "Gen 53: Best Fitness = 0.0053\n",
            "Gen 54: Best Fitness = 0.0053\n",
            "Gen 55: Best Fitness = 0.0053\n",
            "Gen 56: Best Fitness = 0.0053\n",
            "Gen 57: Best Fitness = 0.0053\n",
            "Gen 58: Best Fitness = 0.0053\n",
            "Gen 59: Best Fitness = 0.0053\n",
            "Gen 60: Best Fitness = 0.0053\n",
            "Gen 61: Best Fitness = 0.0053\n",
            "Gen 62: Best Fitness = 0.0053\n",
            "Gen 63: Best Fitness = 0.0053\n",
            "Gen 64: Best Fitness = 0.0053\n",
            "Gen 65: Best Fitness = 0.0053\n",
            "Gen 66: Best Fitness = 0.0053\n",
            "Gen 67: Best Fitness = 0.0053\n",
            "Gen 68: Best Fitness = 0.0053\n",
            "Gen 69: Best Fitness = 0.0053\n",
            "Gen 70: Best Fitness = 0.0053\n",
            "Gen 71: Best Fitness = 0.0053\n",
            "Gen 72: Best Fitness = 0.0053\n",
            "Gen 73: Best Fitness = 0.0053\n",
            "Gen 74: Best Fitness = 0.0053\n",
            "Gen 75: Best Fitness = 0.0053\n",
            "Gen 76: Best Fitness = 0.0053\n",
            "Gen 77: Best Fitness = 0.0053\n",
            "Gen 78: Best Fitness = 0.0053\n",
            "Gen 79: Best Fitness = 0.0053\n",
            "Gen 80: Best Fitness = 0.0053\n",
            "Gen 81: Best Fitness = 0.0053\n",
            "Gen 82: Best Fitness = 0.0053\n",
            "Gen 83: Best Fitness = 0.0053\n",
            "Gen 84: Best Fitness = 0.0053\n",
            "Gen 85: Best Fitness = 0.0053\n",
            "Gen 86: Best Fitness = 0.0053\n",
            "Gen 87: Best Fitness = 0.0053\n",
            "Gen 88: Best Fitness = 0.0053\n",
            "Gen 89: Best Fitness = 0.0053\n",
            "Gen 90: Best Fitness = 0.0053\n",
            "Gen 91: Best Fitness = 0.0053\n",
            "Gen 92: Best Fitness = 0.0053\n",
            "Gen 93: Best Fitness = 0.0053\n",
            "Gen 94: Best Fitness = 0.0053\n",
            "Gen 95: Best Fitness = 0.0053\n",
            "Gen 96: Best Fitness = 0.0053\n",
            "Gen 97: Best Fitness = 0.0053\n",
            "Gen 98: Best Fitness = 0.0053\n",
            "Gen 99: Best Fitness = 0.0053\n",
            "Gen 100: Best Fitness = 0.0053\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<Figure size 640x480 with 1 Axes>"
            ],
            "image/png": "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9SsrK0sAEDt27BBC6NumUqnE0qVLjeecPHlSABB79uyRq8x64+7uLr755hurbWdhYaFo1aqViIuLE/369TMGGWtq75w5c0SnTp2qvM+a2imEEC+99JLo06dPtfdb++fTM888I8LCwoROp7Oq93bEiBHioYceMjk2duxYMXnyZCGE9b+vQgjBoaVGlJycjIyMDMTExBiPubq6okePHtizZ4+MldWP/Px8AICHhwcA4NChQ9BoNCbtDQ8PR7NmzSy6vVqtFr/99huKi4sRHR1tte2cMWMGRowYYdIuwPre1zNnziAgIAAtWrTA5MmTcenSJQDW187Vq1eja9eumDBhAnx8fNC5c2d8/fXXxvut+fOpoqICP//8Mx566CFIkmRV722vXr2wdetWJCUlAQCOHj2KXbt2YdiwYQCs+301sPpNI81JRkYGAMDX19fkuK+vr/E+S6XT6TBz5kz07t0b7du3B6Bvr62tLdzc3EzOtdT2Hj9+HNHR0SgrK4OTkxNWrFiBiIgIHDlyxKraCQC//fYb4uPjceDAgUr3WdP72qNHD3z//fdo06YN0tPTMXfuXNx1111ISEiwqnYCwPnz57FgwQLMmjULr7zyCg4cOICnn34atra2mDJlilV/Pq1cuRJ5eXmYOnUqAOv6Hp49ezYKCgoQHh4OpVIJrVaL+fPnY/LkyQCs+/eOAYMM1YsZM2YgISEBu3btkruUBtOmTRscOXIE+fn5+OOPPzBlyhTs2LFD7rLq3eXLl/HMM88gLi4OdnZ2cpfToAz/agWAjh07okePHggJCcGSJUtgb28vY2X1T6fToWvXrnjnnXcAAJ07d0ZCQgIWLlyIKVOmyFxdw/r2228xbNgwBAQEyF1KvVuyZAl++eUXLF68GO3atcORI0cwc+ZMBAQEWP37asChpUbk5+cHAJWujM/MzDTeZ4mefPJJrF27Ftu2bUNQUJDxuJ+fHyoqKpCXl2dyvqW219bWFi1btkSXLl3w7rvvolOnTvjf//5nde08dOgQsrKyEBUVBRsbG9jY2GDHjh345JNPYGNjA19fX6tq743c3NzQunVrnD171ureV39/f0RERJgca9u2rXEozVo/ny5evIgtW7bgkUceMR6zpvf2hRdewOzZszFp0iR06NAB//d//4dnn30W7777LgDrfV9vxCDTiEJDQ+Hn54etW7cajxUUFGDfvn2Ijo6WsbK6EULgySefxIoVK/Dnn38iNDTU5P4uXbpApVKZtPf06dO4dOmSRbb3ZjqdDuXl5VbXzkGDBuH48eM4cuSI8da1a1dMnjzZ+P/W1N4bFRUV4dy5c/D397e697V3796VlkdISkpCSEgIAOv7fDJYtGgRfHx8MGLECOMxa3pvS0pKoFCY/ipXKpXQ6XQArPd9NSH31cbWprCwUBw+fFgcPnxYABAfffSROHz4sLh48aIQQj8Nzs3NTaxatUocO3ZMjB492mKnwU2fPl24urqK7du3m0xzLCkpMZ7z+OOPi2bNmok///xTHDx4UERHR4vo6GgZq66b2bNnix07dojk5GRx7NgxMXv2bCFJkti8ebMQwnraWZ0bZy0JYT3tfe6558T27dtFcnKy+Pvvv0VMTIzw8vISWVlZQgjraacQ+qn0NjY2Yv78+eLMmTPil19+EQ4ODuLnn382nmNNn09CCKHVakWzZs3ESy+9VOk+a3lvp0yZIgIDA43Tr5cvXy68vLzEiy++aDzH2t7XmzHI1LNt27YJAJVuU6ZMEULop8K9/vrrwtfXV6jVajFo0CBx+vRpeYuuo6raCUAsWrTIeE5paal44oknhLu7u3BwcBD33HOPSE9Pl6/oOnrooYdESEiIsLW1Fd7e3mLQoEHGECOE9bSzOjcHGWtp77333iv8/f2Fra2tCAwMFPfee6/JuirW0k6DNWvWiPbt2wu1Wi3Cw8PFV199ZXK/NX0+CSHEpk2bBIAq22At721BQYF45plnRLNmzYSdnZ1o0aKFePXVV0V5ebnxHGt7X28mCXHD8n9EREREFoTXyBAREZHFYpAhIiIii8UgQ0RERBaLQYaIiIgsFoMMERERWSwGGSIiIrJYDDJERERksRhkiIiIyGIxyBCRRdFqtejVqxfGjh1rcjw/Px/BwcF49dVXZaqMiOTAlX2JyOIkJSUhMjISX3/9NSZPngwAePDBB3H06FEcOHAAtra2MldIRI2FQYaILNInn3yCN998EydOnMD+/fsxYcIEHDhwAJ06dZK7NCJqRAwyRGSRhBAYOHAglEoljh8/jqeeegqvvfaa3GURUSNjkCEii3Xq1Cm0bdsWHTp0QHx8PGxsbOQuiYgaGS/2JSKL9d1338HBwQHJyclISUmRuxwikgF7ZIjIIu3evRv9+vXD5s2bMW/ePADAli1bIEmSzJURUWNijwwRWZySkhJMnToV06dPx4ABA/Dtt99i//79WLhwodylEVEjY48MEVmcZ555BuvXr8fRo0fh4OAAAPjyyy/x/PPP4/jx42jevLm8BRJRo2GQISKLsmPHDgwaNAjbt29Hnz59TO6LjY3FtWvXOMRE1IQwyBAREZHF4jUyREREZLEYZIiIiMhiMcgQERGRxWKQISIiIovFIENEREQWi0GGiIiILBaDDBEREVksBhkiIiKyWAwyREREZLEYZIiIiMhiMcgQERGRxWKQISIiIov1/x4fZyysREwGAAAAAElFTkSuQmCC\n"
          },
          "metadata": {}
        }
      ]
    }
  ]
}