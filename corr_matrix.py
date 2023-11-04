import matplotlib.pyplot as plt
from io import BytesIO
import seaborn as sb
import base64
import numpy as np

def graph(df, cols):
    corr = df[list(cols)].corr(method='pearson')
    fig, axis = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.2)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    sb.heatmap(corr, cmap="YlGnBu", mask=mask, linewidths=.5, square=True)

    pngImage = BytesIO()
    plt.savefig(pngImage, transparent=True, format='png')
    pngImage.seek(0)

    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    image = pngImageB64String

    return image
