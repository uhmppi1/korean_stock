# -*- coding: utf-8 -*-
# marcap_utils.py - 시가총액 데이터를 위한 유틸함수
# 참고 : https://github.com/FinanceData/marcap

# marcap Datasets Columns
# Date : 날짜
# Code : 종목코드
# Name : 종명이름
# Open : 시가
# High : 고가
# Low : 저가
# Close : 종가
# Volume : 거래량
# Amount : 거래대금
# Changes : 전일대비
# ChagesRatio : 전일비
# Marcap : 시가총액(백만원)
# Stocks : 상장주식수
# MarcapRatio : 시가총액비중(%)
# ForeignShares : 외국인 보유주식수
# ForeignRatio : 외국인 지분율(%)
# Rank: 시가총액 순위 (당일)

from datetime import datetime
import numpy as np
import pandas as pd
import glob

# for test
krx200_code_list_small=['005930', '009150']

krx200_code_list_full=[
    '000020',
    '000050',
    '000070',
    '000100',
    '000120',
    '000140',
    '000150',
    '000210',
    '000230',
    '000240',
    '000270',
    '000480',
    '000640',
    '000660',
    '000670',
    '000700',
    '000720',
    '000810',
    #'000830', 삼성물산
    '000880',
    '000990',
    '001040',
    '001060',
    '001120',
    '001130',
    '001210',
    '001230',
    #'001300',
    '001430',
    '001440',
    '001520',
    '001630',
    '001680',
    '001740',
    '001790',
    '001800',
    '001940',
    '002020',
    '002030',
    '002240',
    '002270',
    '002300',
    '002350',
    '002380',
    '002390',
    '002790',
    '003000',
    '003030',
    '003120',
    #'003190',
    '003240',
    '003300',
    '003410',
    #'003450',
    '003480',
    '003490',
    '003520',
    '003550',
    '003570',
    #'003600',
    #'003640',
    '003920',
    #'003940',
    '004000',
    '004020',
    '004130',
    '004150',
    '004170',
    '004370',
    '004430',
    '004490',
    '004710',
    '004800',
    #'004940',
    '004990',
    '005090',
    '005180',
    #'005270',
    #'005280',
    '005300',
    '005380',
    '005490',
    '005500',
    '005680',
    '005930',
    '005940',
    '006120',
    '006260',
    '006280',
    '006360',
    '006380',
    '006400',
    '006650',
    '006800',
    '007310',
    '007340',
    '007570',
    #'008000',
    '008060',
    '008730',
    '008930',
    '009150',
    '009200',
    '009290',
    '009540',
    '009580',
    '009680',
    #'009720',
    '009830',
    '010060',
    '010120',
    '010130',
    '010140',
    #'010520',
    '010620',
    '010950',
    '011070',
    '011170',
    '011200',
    '011780',
    '011790',
    '011810',
    '012330',
    '012450',
    '012630',
    '012750',
    '014820',
    '014830',
    '015590',
    '015760',
    '016360',
    '016380',
    '016800',
    '017670',
    '017800',
    '017960',
    '018880',
    '019680',
    '020000',
    '021240',
    '023530',
    '024110',
    '025000',
    '025540',
    #'025850',
    '025860',
    '028050',
    '028670',
    '029530',
    '029780',
    '030000',
    '030200',
    '032640',
    '033780',
    '034020',
    '034120',
    '034220',
    '035250',
    '035420',
    '036460',
    '036570',
    #'037620',
    '042660',
    '042670',
    '047040',
    '047050',
    #'051310',
    '051900',
    '051910',
    #'053000',
    '055550',
    #'064420',
    '064960',
    '066570',
    #'067250',
    #'068870',
    '069260',
    '069620',
    '069960',
    '071050',
    '077970',
    '078930',
    '084010',
    '085310',
    '086280',
    '086790',
    '090430',
    '091090',
    '093050',
    '093370',
    '096770',
    '097230',
    '097950',
    '100840',
    '103140',
    #'103150',
    '103590',
    '104700',
    '105560',
    '108670',
    ]
krx200_code_list = krx200_code_list_full
code2index = {code:index for index, code in enumerate(krx200_code_list)}
index2code = {index:code for code, index in code2index.items()}


def marcap_date(date):
    '''
    지정한 날짜의 시가총액 순위 데이터
    :param datetime theday: 날짜
    :return: DataFrame
    '''
    date = pd.to_datetime(date)
    csv_file = 'dataset/marcap/marcap-%s.csv.gz' % (date.year)

    result = None
    try:
        df = pd.read_csv(csv_file, dtype={'Code':str, 'ChagesRatio':float}, parse_dates=['Date'], index_col='Date')
        result = df[[ 'Date', 'Code', 'Name',
                          'Open', 'High', 'Low', 'Close', 'Volume', 'Amount',
                          'Changes', 'ChagesRatio', 'Marcap', 'Stocks', 'MarcapRatio',
                          'ForeignShares', 'ForeignRatio', 'Rank']]
        result = result[result['Date'] == date]
        result = result.sort_values(['Date','Rank'])
    except Exception as e:
        print(e)
        return None
    result.reset_index(drop=True, inplace=True)
    return result


def marcap_date_range(start, end, code=None):
    '''
    지정한 기간 데이터 가져오기
    :param datetime start: 시작일
    :param datetime end: 종료일
    :param str code: 종목코드 (지정하지 않으면 모든 종목)
        or list codes[] : 종목코드의 리스트 (종목코드 복수개 지정 가능)
    :return: DataFrame
    '''
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    df_list = []
    for year in range(start.year, end.year + 1):
        try:
            cratio_parse = lambda cratio: 0.0 if cratio == '#######' else float(cratio)
            csv_file = 'dataset/marcap/marcap-%s.csv.gz' % (year)
            df = pd.read_csv(csv_file, dtype={'Code': str}, parse_dates=['Date'],
                             converters={'ChagesRatio': cratio_parse})
            df_list.append(df)
        except Exception as e:
            print(e)
    df_merged = pd.concat(df_list)
    df_merged = df_merged[(start <= df_merged['Date']) & (df_merged['Date'] <= end)]
    df_merged = df_merged.sort_values(['Date','Rank'])
    if code:
        if not isinstance(code, list):
            code = [code]
        df_merged = df_merged[df_merged['Code'].isin(code)]
    df_merged.reset_index(drop=True, inplace=True)
    return df_merged


def marcap_date_range_dateindexed(start, end, code=None):
    '''
    지정한 기간 데이터 가져오기
    :param datetime start: 시작일
    :param datetime end: 종료일
    :param str code: 종목코드 (지정하지 않으면 모든 종목)
        or list codes[] : 종목코드의 리스트 (종목코드 복수개 지정 가능)
    :return: DataFrame
    '''
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    df_list = []
    for year in range(start_date.year, end_date.year + 1):
        try:
            cratio_parse = lambda cratio: 0.0 if cratio == '#######' else float(cratio)
            csv_file = 'dataset/marcap/marcap-%s.csv.gz' % (year)
            df = pd.read_csv(csv_file, dtype={'Code':str}, parse_dates=['Date'], index_col='Date',
                             converters={'ChagesRatio': cratio_parse}).sort_index()
            df_list.append(df)
        except Exception as e:
            print(e)
    df_merged = pd.concat(df_list)
    df_merged = df_merged[start:end]
    # df_merged = df_merged.sort_values(['Rank'])
    if code:
        if not isinstance(code, list):
            code = [code]
        df_merged = df_merged[df_merged['Code'].isin(code)]
    return df_merged
