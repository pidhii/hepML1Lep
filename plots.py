from ROOT import TCanvas, TH1F, TGraph, TLegend
from math import log, sqrt
import numpy as np

NBINS = 30

LUMI = 35.9E+03

class Classifier:
    def __init__(self, pred, scoreing, multi=False):
        # Create histograms.
        self._sig = TH1F("Classifier._sig", "Classification", NBINS, 0, 1.001)
        self._bg = TH1F("Classifier._bg", "Classification", NBINS, 0, 1.001)

        self._sig_w = TH1F("Classifier._sig_w", "Classification (weighted)", NBINS, 0, 1.001)
        self._bg_w = TH1F("Classifier._bg_w", "Classification (weighted)", NBINS, 0, 1.001)

        # Fill histograms.
        issigs = scoreing.testDF["isSignal"].values
        ws = scoreing.testDF["Finalweight"].values
        cs = pred[3 if multi else 0].values
        for issig, w, x in zip(issigs, ws, cs):
            if issig == (3 if multi else 1):
                self._sig.Fill(x)
                self._sig_w.Fill(x, w * LUMI)
            else:
                self._bg.Fill(x)
                self._bg_w.Fill(x, w * LUMI)

    def signal(self):
        return self._sig_w

    def background(self):
        return self._bg_w

    def prepare(self):
        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Non-weighted histograms.
        #
        self._leg = TLegend(0.7, 0.9, 0.9, 0.7)

        # Draw signal-histogram.
        self._sig.GetXaxis().SetTitle("classification")
        self._sig.SetFillColor(8)
        self._sig.SetLineColor(8)
        self._sig.SetFillStyle(3002)
        self._sig.SetMaximum(max(
            self._sig.GetBinContent(self._sig.GetMaximumBin()),
            self._bg.GetBinContent(self._bg.GetMaximumBin())
        ))

        # Draw background-histogram.
        self._bg.SetFillColor(46)
        self._bg.SetLineColor(46)
        self._bg.SetFillStyle(3002)

        self._leg.AddEntry(self._sig, "signal")
        self._leg.AddEntry(self._bg, "background")

        # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        # Weighted histograms.
        #
        self._leg_w = TLegend(0.7, 0.9, 0.9, 0.7)

        self._sig_w.GetXaxis().SetTitle("classification")
        self._sig_w.SetFillColor(8)
        self._sig_w.SetLineColor(8)
        self._sig_w.SetFillStyle(3002)
        self._sig_w.SetMaximum(max(
            self._sig_w.GetBinContent(self._sig_w.GetMaximumBin()),
            self._bg_w.GetBinContent(self._bg_w.GetMaximumBin())
        ))

        # Draw background-histogram.
        self._bg_w.SetFillColor(46)
        self._bg_w.SetLineColor(46)
        self._bg_w.SetFillStyle(3002)

        self._leg_w.AddEntry(self._sig_w, "signal")
        self._leg_w.AddEntry(self._bg_w, "background")

    def draw(self):
        self._canvas = TCanvas()
        self._canvas.SetLogy()
        self._sig.Draw()
        self._bg.Draw('SAME')
        self._leg.Draw('SAME')

        self._canvas_w = TCanvas()
        self._canvas_w.SetLogy()
        self._sig_w.Draw()
        self._bg_w.Draw('SAME')
        self._leg_w.Draw('SAME')


# Asimov significance
def Z(s,b,sig=None):
    #if sig == None: sig=eps
		try:
			return sqrt(-2.0/(sig*sig)*log(1.0 + b*(sig*sig)*s/(b+(b*b)*(sig*sig)))+ \
						 2.0*(s+b)*log((s+b)*(b+(b*b)*(sig*sig))/( (b*b)+(s+b)*(b*b)*(sig*sig))))
		except Exception as e:
			return 0

class Significance:
    def __init__(self, h_sig, h_bg):
        g_y = []
        g_x = []
        for i in range(1, NBINS+1):
            s = h_sig.Integral(i, NBINS)
            b = h_bg.Integral(i, NBINS)
            g_y.append(Z(s, b, 0.5))
            g_x.append(1./NBINS * i)

        ys = np.array(g_y)
        xs = np.array(g_x)
        self._g = TGraph(len(ys), xs, ys)

    def prepare(self):
        self._g.SetLineColor(9)
        self._g.SetLineWidth(2)
        self._g.GetYaxis().SetTitle("Asimov significance")
        self._g.GetXaxis().SetTitle("Cut on classifier score")

    def draw(self):
        self._c = TCanvas()
        self._g.Draw()

