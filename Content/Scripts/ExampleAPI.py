import tensorflow as tf
import unreal_engine as ue
from TFPluginAPI import TFPluginAPI
import numpy as np


class ExampleAPI(TFPluginAPI):
  input_data_column_cnt = 3 * 3  # 입력데이터의 컬럼 개수(Variable 개수)
  output_data_column_cnt = 1  # 결과데이터의 컬럼 개수

  seq_length = 100  # 1개 시퀀스의 길이(시계열데이터 입력 개수)
  rnn_cell_hidden_dim = 20  # 각 셀의 (hidden)출력 크기
  forget_bias = 1.0  # 망각편향(기본값 1.0)
  num_stacked_layers = 1  # stacked LSTM layers 개수
  keep_prob = 1.0  # dropout할 때 keep할 비율
  b = np.zeros((seq_length, input_data_column_cnt))
  my_list=[]
  status = []
  index=0

  # expected optional api: setup your model for training
  def onSetup(self):
    print("Good?/")
    ue.log("잘되는건가?/")
    self.X = tf.placeholder(tf.float32, [None, self.seq_length, self.input_data_column_cnt])

    def lstm_cell():
      # LSTM셀을 생성
      # num_units: 각 Cell 출력 크기
      # forget_bias:  to the biases of the forget gate
      #        (default: 1)  in order to reduce the scale of forgetting in the beginning of the training.
      # state_is_tuple: True ==> accepted and returned states are 2-tuples of the c_state and m_state.
      # state_is_tuple: False ==> they are concatenated along the column axis.
      cell = tf.contrib.rnn.BasicLSTMCell(num_units=self.rnn_cell_hidden_dim,
                        forget_bias=self.forget_bias, state_is_tuple=True,
                        activation=tf.nn.softsign)

      if self.keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
      return cell

    stackedRNNs = [lstm_cell() for _ in range(self.num_stacked_layers)]
    multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs,
                          state_is_tuple=True) if self.num_stacked_layers > 1 else lstm_cell()

    # RNN Cell(여기서는 LSTM셀임)들을 연결
    hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, self.X, dtype=tf.float32)

    # [:, -1]를 잘 살펴보자. LSTM RNN의 마지막 (hidden)출력만을 사용했다.
    self.hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], self.output_data_column_cnt,
                              activation_fn=tf.identity)

    # sess = tf.Session()
    # sess.run(tf.global_variables_initializer())

    save_file = 'C:/Users/bearics/Desktop/4gram/4gram-new-tf/Content/Scripts/model/train_model.ckpt'
    saver = tf.train.Saver()
    self.sess = tf.Session()
    saver.restore(self.sess, save_file)
    ue.log("모델 로드 완료")

    # self.sess = tf.InteractiveSession()
    # # self.graph = tf.get_default_graph()
    #
    # self.a = tf.placeholder(tf.float32)
    # self.b = tf.placeholder(tf.float32)
    #
    # # operation
    # self.c = self.a + self.b
    # ue.log("aaa")
    pass

  # expected optional api: parse input object and return a result object, which will be converted to json for UE4
  def onJsonInput(self, jsonInput):
    #print("json : " + str(jsonInput))
    temp_list=[]
    temp_list.append(jsonInput["head"]["pitch"])
    temp_list.append(jsonInput["head"]["yaw"])
    temp_list.append(jsonInput["head"]["roll"])
    temp_list.append(jsonInput["lHand"]["pitch"])
    temp_list.append(jsonInput["lHand"]["yaw"])
    temp_list.append(jsonInput["lHand"]["roll"])
    temp_list.append(jsonInput["rHand"]["pitch"])
    temp_list.append(jsonInput["rHand"]["yaw"])
    temp_list.append(jsonInput["rHand"]["roll"])
    if len(self.my_list)!=self.seq_length:
      self.my_list.append(temp_list)
      return len(self.my_list)
    else:
      self.my_list.pop(0)
      self.my_list.append(temp_list)
      test_predict = self.sess.run(self.hypothesis, feed_dict={self.X: [self.my_list]})
      result = float(test_predict[0][0])
      if result>0.3:
      	if len(self.status) < 10:
          self.status.append(1)
      else:
        if len(self.status) != 0:
          self.status.pop(0)

      if len(self.status) > 2:
        return 1
      else:
        return 0




  # custom function to change the op
  def changeOperation(self, type):
    self.b[0] = [1, 1, 1, 1, 1, 1]
    test_predict = self.sess.run(self.hypothesis, feed_dict={self.X: [self.b]})
    ue.log(test_predict[0][0])

  # expected optional api: start training your network
  def onBeginTraining(self):
    pass


# NOTE: this is a module function, not a class function. Change your CLASSNAME to reflect your class
# required function to get our api
def getApi():
  # return CLASSNAME.getInstance()
  return ExampleAPI.getInstance()