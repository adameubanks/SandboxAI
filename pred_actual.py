from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.metrics import r2_score, mean_squared_error
from matplotlib.figure import Figure
from matplotlib import colors
from io import BytesIO
import numpy as np
import base64

def graph(y_train, y_pred):
    #get standard deviation and mean of y
    std_y = np.std(y_train)
    mean_y = np.mean(y_train)

    #plot figure
    fig = Figure(facecolor=(0.0, 0.0, 0.0, 0.0))
    axis = fig.add_subplot(1, 1, 1)
    axis.set_title("Actual vs Predicted Values")
    axis.set_xlabel("Actual Values")
    axis.set_ylabel("Predicted Values")
    axis.grid()
    axis.hist2d(x=y_train, y=y_pred, bins=100, cmap='viridis', norm=colors.LogNorm())

    #set limits within one standard deviation of mean
    axis.set_xlim([mean_y-std_y, mean_y+std_y])
    axis.set_ylim([mean_y-std_y, mean_y+std_y])
    #line to show comparison with perfect prediction
    axis.plot([0, 1], [0, 1], transform=axis.transAxes, color='red', linestyle='--')
    axis.set_facecolor((1, 1, 1, 0.1))
    #save figure as png    
    pngImage = BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    #encode png as base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    image = pngImageB64String

    return image

def metrics(y_train, y_pred):
    r_score = r2_score(y_pred,y_train)
    mse = mean_squared_error(y_pred,y_train)

    return r_score, mse