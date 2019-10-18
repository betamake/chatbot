
#!/usr/bin/env Python
# coding=utf-8

from flask import Flask, render_template, request, make_response
from flask import jsonify

import sys
import time  
import threading
import os
from datapreprocess import preprocess
import train_eval
import fire
from QA_data import QA_test
from config import Config


def heartbeat():
    print (time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()
timer = threading.Timer(60, heartbeat)
timer.start()

try:  
    import xml.etree.cElementTree as ET  
except ImportError:  
    import xml.etree.ElementTree as ET


app = Flask(__name__,static_url_path="/static") 

@app.route('/message', methods=['POST'])
def reply():
    opt = Config()
   # for k, v in kwargs.items(): #设置参数
    #    setattr(opt, k, v)   

    searcher, sos, eos, unknown, word2ix, ix2word = train_eval.test(opt)

    if os.path.isfile(opt.corpus_data_path) == False:
        preprocess()

    
    input_sentence =  request.form['msg']
    #if input_sentence == 'q' or input_sentence == 'quit' or input_sentence == 'exit': break
    if opt.use_QA_first:
        query_res = QA_test.match(input_sentence)
        if(query_res == tuple()):
             output_words = train_eval.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word)
        else:
            output_words = "您是不是要找以下问题: " + query_res[1] + '，您可以尝试这样: ' + query_res[2]
    else:
        output_words = train_eval.output_answer(input_sentence, searcher, sos, eos, unknown, opt, word2ix, ix2word)
    print('BOT > ',output_words)
    return jsonify( { 'text': output_words } )

@app.route("/")
def index(): 
    return render_template("index.html")

# 启动APP
if (__name__ == "__main__"): 
    app.run(host = '0.0.0.0', port = 8808) 
