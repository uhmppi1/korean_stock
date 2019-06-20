# korean_stock

----

한국 주식데이터(일자별)를 기반으로 주가 흐름을 예측하는 sequential model을 만들어 본다.  
data source : https://github.com/FinanceData/marcap

# plan


## 1. 데이터셋 만들기 (3가지 버전)  
 - 삼성전기 only. seq_len = 10, columns = 3  
 - 짧게 돌려볼 수 있는 것 (2018-03-01~2018-08-31)  20개 회사, seq_len = 10, columns = 3
 - 성능을 확인할 수 있는 것 (2010-01-01~2018-10-31)  179개 회사, seq_len = 30, columns = all(13)

## 2. 1-step prediction 모델 돌려보기  
* 회사 임베딩 결합 모델 구현  
   model 0) 회사 임베딩 없이. 1company only (baseline)  
   model 1) 회사 임베딩을 input에 concat  
   model 2) 회사 임베딩을 LSTM output에 concat  
* 실행  
~~~
python run_experiment_1step_xxx.py
~~~
   

## 3. seq2seq 모델 구현 및 돌려보기  
   model 0) multi 1-step prediction (baseline for seq2seq) -> 1-step의 가장 성능 좋은 모델
   model 1) 회사 임베딩을 input에 concat  
   model 2) 회사 임베딩을 context에 concat  
   model 3) 회사 임베딩을 decoder output에 concat  
* 실행 
~~~
python run_experiment_seq2seq_xxx.py 
~~~

## 4. attentional seq2seq 모델 구현 및 돌려보기  
* 실행 
~~~
python run_experiment_attn_seq2seq_xxx.py 
~~~

## 5. (가능하면) transformer 모델 적용 및 돌려보기  
* 실행 
~~~
python run_experiment_transformer_xxx.py 
~~~