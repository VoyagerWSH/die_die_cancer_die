import sys
from os.path import dirname, realpath
sys.path.append(dirname(dirname(realpath(__file__))))
from src.dataset import NLST

def main():
    print("Loading data ..")
    datamodule = NLST(**vars(NLST))
    datamodule.setup()
    test_dataloader = datamodule.test_dataloader()
    y = []
    lung_rads = []
    for x in test_dataloader:
        lung_rads.extend(x["lung_rads"].tolist())
        y.extend(x["y"].tolist())

    # Simulating Clinical Utility
    # Rads     Not Rads
    # Cancer   a         b
    # Not Cancer   c     d
    a = 0
    b = 0
    c = 0
    d = 0
    for i in range(len(y)):
        if lung_rads[i] == 0 and y[i] == 0:
            d += 1
        elif lung_rads[i] == 0 and y[i] == 1:
            b += 1
        elif lung_rads[i] == 1 and y[i] == 0:
            c += 1
        elif lung_rads[i] == 1 and y[i] == 1:
            a += 1

    print("Sensitivity of the LungRads criteria on the NLST test set is " + str(a / (a + b)))
    print("Specificity of the LungRads criteria on the NLST test set is " + str(d / (d + c)))
    print("Positive Predictive Value of the LungRads criteria on the NLST test set is " + str(a / (a + c)))

if __name__ == '__main__':
    main()
