import tensorflow as tf
import keras
from keras import backend as K

def weight(score, weights):
    score_arr = (score * weights) \
              / K.mean(K.cast(K.not_equal(weights, 0), K.floatx()))
    return K.mean(score_arr)

################################################################################
# Binary classification
#
def binary(name, expsig, expbg, sigma=None, debug=False):
  def loss(y_data, y_pred):
    y_true = y_data[:,0]         # true labels
    y_pred = y_pred[:,0]         # DNN predictions
    sample_weights = y_data[:,1] # per-event-weights
    class_weights = y_data[:,2]  # per-class-weights

    # calculate these summs separately co'z need to check if signal/background
    # is present at all
    sig_sum = K.sum(sample_weights * y_true)
    bg_sum  = K.sum(sample_weights * (1 - y_true))

    s_w = expsig / sig_sum
    b_w = expbg  / bg_sum

    # set `s` and `b` to zero when N_signal or N_background = 0
    #  (otherwize would yeild NaN)
    # s = K.switch(K.equal(sig_sum, 0),
          # lambda: K.constant(0),
          # lambda: s_w * K.sum(sample_weights * y_pred * y_true))
    # b = K.switch(K.equal(bg_sum, 0),
          # lambda: K.constant(2),
          # lambda: b_w * K.sum(sample_weights * y_pred * (1 - y_true)))

    s = s_w * K.sum(sample_weights * y_pred * y_true)
    b = b_w * K.sum(sample_weights * y_pred * (1 - y_true))

    # gets crazy when b = 0
    b = tf.maximum(tf.constant(0.01), b)

    dummy = K.sum(K.print_tensor(y_data, "y_data =")) \
          + K.sum(K.print_tensor(y_true, "y_true =")) \
          + K.print_tensor(K.max(y_true), "max(y_true) =") \
          + K.print_tensor(K.min(y_true), "min(y_true) =") \
          + K.sum(K.print_tensor(y_pred, "y_pred =")) \
          + K.print_tensor(K.max(y_pred), "max(y_pred) =") \
          + K.print_tensor(K.min(y_pred), "min(y_pred) =") \
          + K.print_tensor(b, "b =") \
          + K.print_tensor(s, "s =") \
          + K.print_tensor(K.sum(y_true), "N signal =") \
          + K.print_tensor(K.sum(1-y_true), "N background =")

    sigB = sigma * b

    if debug: s += 0*dummy

    if name == 'crossentropy':
      cc = keras.losses.binary_crossentropy(y_true, y_pred)
      return weight(cc, class_weights)
    elif name == 'significance':
      loss = (s + b) / (s*s + K.epsilon())
      if debug: loss += 0*K.print_tensor(loss, "loss =")
      return loss
    elif name == 'asimov':
      ln1_top = (s + b)*(b + sigB*sigB)
      ln1_bot = b*b + (s + b)*sigB*sigB
      ln1 = K.log(ln1_top / (ln1_bot + K.epsilon()) + K.epsilon())
      ln2 = K.log(1. + sigB*sigB*s / (b*(b + sigB*sigB) + K.epsilon()))
      loss = 1./K.sqrt(2*((s + b)*ln1 - b*b*ln2/(sigB*sigB + K.epsilon())) + K.epsilon())
      if debug: loss += 0*K.print_tensor(loss, "loss =")
      return loss
    else:
      print("\x1b[38;5;1;1mError:\x1b[0m undefined loss name, {}".format(name))
      raise Exception("undefined loss")

  return loss


################################################################################
# Multiclass
#
def multiclass(name, expsig, expbg, sigma=None, debug=False, cc_weight=1E-04):
  def loss(y_data, y_pred):
    y_true = [y_data[:, i] for i in range(4)]
    y_pred = [y_pred[:, i] for i in range(4)]
    sample_weights = y_data[:, 4]
    class_weights  = y_data[:, 5]

    def crossentropy(i):
      true = y_true[i]
      pred = y_pred[i]
      score = keras.losses.categorical_crossentropy(y_true[i], y_pred[i])
      return weight(score, class_weights)

    def significance(i):
      s_w = expsig / K.sum(y_true[i])
      b_w = expbg  / K.sum(1 - y_true[i])
      s = s_w * K.sum(sample_weights * y_pred[i] * y_true[i])
      b = b_w * K.sum(sample_weights * y_pred[i] * (1 - y_true[i]))
      b = tf.cond(b < 0, lambda: tf.constant(0.), lambda: b)
      return (s + b) / (s*s + K.epsilon())

    def asimov(i):
      s_w = expsig / K.sum(sample_weights * y_true[i])
      b_w = expbg / K.sum(sample_weights * (1 - y_true[i]))

      s = s_w * K.sum(sample_weights * y_pred[i] * y_true[i])
      b = b_w * K.sum(sample_weights * y_pred[i] * (1 - y_true[i]))
      b = tf.maximum(tf.constant(2.), b)
      sigB = sigma * b

      ln1_top = (s + b)*(b + sigB*sigB)
      ln1_bot = b*b + (s + b)*sigB*sigB
      ln1 = K.log(ln1_top / (ln1_bot + K.epsilon()) + K.epsilon())
      ln2 = K.log(1. + sigB*sigB*s / (b*(b + sigB*sigB) + K.epsilon()))
      loss = 1./K.sqrt(2*((s + b)*ln1 - b*b*ln2/(sigB*sigB + K.epsilon())) + K.epsilon())
      # loss = weight(loss, sample_weights)
      return loss

    cc = crossentropy(0) \
       + crossentropy(1) \
       + crossentropy(2) \
       + crossentropy(3)

    asi = asimov(3)

    if debug:
      cc = K.print_tensor(cc, "cc =")
      asi = K.print_tensor(asi, "asi =")

    if name == 'crossentropy':
      return cc
    elif name == 'asimov':
      return asi + cc * cc_weight
    else:
      print("\x1b[38;5;1;1mError:\x1b[0m undefined loss name, {}".format(name))
      raise Exception("undefined loss")

  return loss

