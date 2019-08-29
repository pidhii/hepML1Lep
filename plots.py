from ROOT import TCanvas, TH1F, TGraph, TLegend
# from math import log, sqrt
import numpy as np
from numpy import log, power, sqrt

NBINS = 30
LUMI = 35.9E+03

# Asimov significance
def Z(s,b,sig=None):
    #if sig == None: sig=eps
    try:
        return sqrt(-2.0/(sig*sig)*log(1.0 + b*(sig*sig)*s/(b+(b*b)*(sig*sig))) + 2.0*(s+b)*log((s+b)*(b+(b*b)*(sig*sig))/( (b*b)+(s+b)*(b*b)*(sig*sig))))
    except Exception as e:
        return 0

# error propagation on Asimov significance
def eZ(s,es,b,eb,sig=None):
    #if sig == None: sig=eps
    #if sig < eps: sig=eps # to avoid stability issue in calculation
    try:
        return power(-(eb*eb)/( 1.0/(sig*sig)*log( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)-( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))))*power( 1.0/( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)/(sig*sig)*( 1.0/( b+(b*b)*(sig*sig))*(sig*sig)*s-b/power( b+(b*b)*(sig*sig),2.0)*(sig*sig)*( 2.0*b*(sig*sig)+1.0)*s)-( ( b+s)*( 2.0*b*(sig*sig)+1.0)/( (b*b)+( b+s)*(b*b)*(sig*sig))+( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))-( b+s)*( 2.0*( b+s)*b*(sig*sig)+2.0*b+(b*b)*(sig*sig))*( b+(b*b)*(sig*sig))/power( (b*b)+( b+s)*(b*b)*(sig*sig),2.0))/( b+(b*b)*(sig*sig))*( (b*b)+( b+s)*(b*b)*(sig*sig))-log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))),2.0)/2.0-1.0/( 1.0/(sig*sig)*log( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)-( b+s)*log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))))*power( log(( b+s)*( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig)))+1.0/( b+(b*b)*(sig*sig))*( ( b+(b*b)*(sig*sig))/( (b*b)+( b+s)*(b*b)*(sig*sig))-( b+s)*(b*b)*( b+(b*b)*(sig*sig))*(sig*sig)/power( (b*b)+( b+s)*(b*b)*(sig*sig),2.0))*( (b*b)+( b+s)*(b*b)*(sig*sig))-1.0/( b/( b+(b*b)*(sig*sig))*(sig*sig)*s+1.0)*b/( b+(b*b)*(sig*sig)),2.0)*(es*es)/2.0,(1.0/2.0))
    except Exception as e:
        return 0

class Classifier:
    def __init__(self, pred, scoreing, multi=False, classidx=3, expsig=None, expbg=None, sigma=0.1, nsig_tot=None, nbg_tot=None, n_tot=None):
        # Create histograms.
        sufix = '' if not multi else '-{}'.format(classidx)
        self._sig = TH1F("Classifier._sig" + sufix, "Classification" + sufix, NBINS, 0, 1.001)
        self._bg = TH1F("Classifier._bg" + sufix, "Classification" + sufix, NBINS, 0, 1.001)

        self._sig_w = TH1F("Classifier._sig_w" + sufix, "Classification (weighted)" + sufix, NBINS, 0, 1.001)
        self._bg_w = TH1F("Classifier._bg_w" + sufix, "Classification (weighted)" + sufix, NBINS, 0, 1.001)

        self._multi = multi
        if multi:
            h_multi = [TH1F("Classifier{}.h[{}]".format(sufix, i), "Class {}/{}".format(i, classidx), NBINS, 0, 1.001) for i in range(4)]

        #
        # Fill histograms and compute "power".
        #
        issigs = scoreing.testDF["isSignal"].values
        ws = scoreing.testDF["Finalweight"].values
        cs = pred[classidx if multi else 0].values

        s_w_sum = b_w_sum = 0.
        s_sum = b_sum = 0.
        nsig = nbg = 0
        if multi: n_multi = [0, 0, 0, 0]
        for issig, w, x in zip(issigs, ws, cs):
            if issig == (classidx if multi else 1):
                self._sig.Fill(x)
                self._sig_w.Fill(x, w * LUMI)

                s_w_sum += w
                s_sum += w * x

                nsig += 1

            else:
                self._bg.Fill(x)
                self._bg_w.Fill(x, w * LUMI)

                b_w_sum += w
                b_sum += w * x

                nbg += 1

            if multi:
                h_multi[issig].Fill(x, w * LUMI)
                n_multi[issig] += 1

        self._sig_w.Scale(float(nsig_tot) / nsig)
        self._bg_w.Scale(float(nbg_tot) / nbg)

        if multi:
          for i in range(4):
              h_multi[i].Scale(float(n_tot[i]) / n_multi[i])
          self._h_multi = h_multi

        s = expsig * s_sum / s_w_sum
        b = expbg * b_sum / b_w_sum
        self.significance = Z(s, b, sigma)

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

        if self._multi:
            self._canvas_multi = TCanvas()
            for h in self._h_multi:
                h.Draw('' if h is self._h_multi[0] else 'SAME')


class Significance:
    def __init__(self, h_sig, h_bg, sigma):
        ys_h = []
        ys_l = []
        xs   = []
        s_acc = b_acc = 0
        s_err_acc = b_err_acc = 0
        self.sigma = sigma

        for i in reversed(range(1, NBINS+1)):
            s_h = b_h = s_l = b_l = 0

            s_val = h_sig.GetBinContent(i)
            s_err = h_sig.GetBinError(i)
            b_val = h_bg.GetBinContent(i)
            b_err = h_bg.GetBinError(i)

            s_acc += s_val
            b_acc += b_val
            s_err_acc += s_err**2
            b_err_acc += b_err**2

            sign = Z(s_acc, b_acc, self.sigma)
            sign_err = eZ(s_acc, sqrt(s_err_acc), b_acc, sqrt(b_err_acc), self.sigma)

            ys_h.append(sign + sign_err)
            ys_l.append(sign - sign_err)
            xs.append(1./NBINS * i)

        ys_h = np.array(list(reversed(ys_h)))
        ys_l = np.array(list(reversed(ys_l)))
        xs   = np.array(list(reversed(xs)))
        n = len(ys_h)
        self._g_h = TGraph(n, xs, ys_h)
        self._g_l = TGraph(n, xs, ys_l)

        shade = TGraph(2*n)
        for i in range(n):
            shade.SetPoint(i, xs[i], ys_l[i])
            shade.SetPoint(n+i, xs[n-i-1], ys_h[n-i-1])
        self._g_hl = shade

    def prepare(self):
        self._g_h.SetLineColor(4)
        self._g_h.SetLineWidth(2)
        self._g_h.GetYaxis().SetTitle("Asimov significance")
        self._g_h.GetXaxis().SetTitle("Cut on classifier score")

        self._g_l.SetLineColor(4)
        self._g_l.SetLineWidth(2)
        self._g_l.GetYaxis().SetTitle("Asimov significance")
        self._g_l.GetXaxis().SetTitle("Cut on classifier score")

        self._g_hl.SetFillStyle(3001)
        self._g_hl.SetFillColor(9)
        self._g_hl.SetTitle("")
        self._g_hl.GetYaxis().SetTitle("Asimov significance")
        self._g_hl.GetXaxis().SetTitle("Cut on classifier score")

    def draw(self):
        self._c = TCanvas()
        self._g_hl.Draw('AF')
        self._g_h.Draw('LSAME')
        self._g_l.Draw('LSAME')

    def save(self, name):
        self._g_hl.SetName(name)
        self._g_hl.Write()
