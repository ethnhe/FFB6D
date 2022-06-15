# -*- coding:utf-8 -*-
import sys
import copy
import time
import json
import hashlib
import re
import os
import datetime
import requests
from datetime import date
from collections import Iterable
from bson import ObjectId

PY3 = sys.version_info >= (3,)

if PY3:
    basestring_type = str
    unicode_type = str
    import urllib.request as request
else:
    basestring_type = basestring
    unicode_type = unicode
    import urllib2 as request

default_headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/76.0.3809.87 Safari/537.36',
    'Accept': '*/*',
    'cache-control': "no-cache"
}


def test():
    print('开始进行测试')
    print('format_json >>', format_json({"a": 1, "b": "b", "time": datetime.datetime.now(), "ObjectId": ObjectId()}))

    print('get_string >>', get_string('abcde如果这句话前后没有字母则表示正常abcde', 'abcde', 'abcde'))

    print('remove_dic_key >>', remove_dic_key({'a': 1, 'b': 2, 'c': '如果只看到我则表示正常'}, ['a', 'b']))

    print('str_2_time, 2019年09月08日14:24:01 >>', str_2_time('2019年09月08日14:24:01'))
    print('str_2_time, 2019年09月08日 >>', str_2_time('2019年09月08日'))
    print('str_2_time, 5秒前 >>', str_2_time('5秒前'))

    print('get_str_time >>', get_str_time())

    print('unixtime_2_datetime >>', unixtime_2_datetime(time.time()))
    print('unixtime_2_datetime, millisecond >>', unixtime_2_datetime(time.time() * 1000, True))

    print('datetime_2_unixtime >>', datetime_2_unixtime(datetime.datetime.now()))
    print('datetime_2_unixtime, millisecond >>', datetime_2_unixtime(datetime.datetime.now(), True))

    print('get_html baidu.com >>',
          '\033[32msuccess\033[30m' if get_html('https://www.baidu.com') else '\033[31mfail\033[30m')

    print('get_html_for_requests baidu.com >>',
          '\033[32msuccess\033[30m' if get_html_for_requests('https://www.baidu.com') else '\033[31mfail\033[30m')

    print('post_for_request baidu.com >>',
          '\033[32msuccess\033[30m' if post_for_request('https://www.baidu.com') else '\033[31mfail\033[30m')
    print('post_for_request_return_html_and_cookie baidu.com >>',
          '\033[32msuccess\033[30m' if post_for_request_return_html_and_cookie(
              'https://www.baidu.com') else '\033[31mfail\033[30m')

    print('md5_from_str >>', md5_from_str('123'))
    print('md5_from_file >>', md5_from_file('./cPython.py'))

    print('sha1_from_str >>', sha1_from_str('123'))
    print('sha1_from_file >>', sha1_from_file('./cPython.py'))

    print('版本对比测试, 1.0.1 > 1.0.1 >>', versioncodeCompare('1.0.1', '>', '1.0.1'))

    print('测试完成')


# 将str格式化为json, 并且将其格式化为能被正常接受的json类型(如ObjectId转为str, datetime转为时间戳等)
def format_json(v, encode=None):
    """convert any list, dict, iterable and primitives object to string
    """
    if isinstance(v, basestring_type):
        v = json.loads(v)

    if isinstance(v, dict):
        return __to_dict_str(v, encode)

    if isinstance(v, Iterable):
        return __to_list_str(v, encode)

    if encode:
        return encode(v)
    else:
        return __default_encode(v)


def __to_dict_str(origin_value, encode=None):
    """recursively convert dict content into string
    """
    value = copy.deepcopy(origin_value)
    for k, v in value.items():
        if isinstance(v, dict):
            value[k] = __to_dict_str(v, encode)
            continue

        if isinstance(v, list):
            value[k] = __to_list_str(v, encode)
            continue

        if encode:
            value[k] = encode(v)
        else:
            value[k] = __default_encode(v)

    return value


def __to_list_str(value, encode=None):
    """recursively convert list content into string

    :arg list value: The list that need to be converted.
    :arg function encode: Function used to encode object.
    """
    result = []
    for index, v in enumerate(value):
        if isinstance(v, dict):
            result.append(__to_dict_str(v, encode))
            continue

        if isinstance(v, list):
            result.append(__to_list_str(v, encode))
            continue

        if encode:
            result.append(encode(v))
        else:
            result.append(__default_encode(v))

    return result


def __default_encode(v):
    """convert ObjectId, datetime, date into string
    """
    if isinstance(v, ObjectId):
        return unicode_type(v)

    if isinstance(v, datetime.datetime):
        return __format_time(v)

    if isinstance(v, date):
        return __format_time(v)

    return v


def __format_time(dt):
    """datetime format
    """
    return time.mktime(dt.timetuple())


def get_today_last_datetime():
    now = datetime.datetime.now()
    zeroToday = now - datetime.timedelta(hours=now.hour, minutes=now.minute, seconds=now.second, microseconds=now.microsecond)
    lastToday = zeroToday + datetime.timedelta(hours=23, minutes=59, seconds=59)
    return lastToday


# 截取字符串
def get_string(data, start_str, end_str, contain=False):
    try:
        start_index = 0 if start_str == '' else data.find(start_str)
        start_len = len(start_str)
        if data[start_index:].find(end_str) == -1:
            return ''
        end_index = len(data) if end_str == '' else data[start_index+start_len:].find(end_str) + start_index + start_len + end_str.__len__()
        if not contain:
            start_index += start_str.__len__()
            end_index -= end_str.__len__()
        return data[start_index:end_index]
    except:
        return ''


# 删除字典中指定的key
def remove_dic_key(dic, keyList=[]):
    try:
        for k in keyList:
            if k == '': continue;
            if k in dic.keys():
                dic.pop(k)
        return dic
    except:
        return dic


# 将Str转换为datetime格式
def str_2_time(str, format='auto'):
    try:
        str_bak = str
        str = str.replace('年', '-').replace('月', '-').replace('日', '').replace('时', ':').replace('分', ':').replace(
            '秒', '')
        if str.endswith(':'):
            str = str[0:''.__len__() - 1]
            str = str.replace(' ', '')
        if re.match('\d{4}-\d{1,2}-\d{1,2}\d{1,2}:\d{1,2}:\d{1,2}', str):
            format = '%Y-%m-%d%H:%M:%S'
        elif re.match('\d{4}-\d{1,2}-\d{1,2}\d{1,2}:\d{1,2}', str):
            format = '%Y-%m-%d%H:%M'
        elif re.match('\d{4}-\d{1,2}-\d{1,2}', str):
            format = '%Y-%m-%d'
        else:
            str = str_bak
            num = re.findall('^\d*', str)[0]
            if num != '':
                num = int(num)
                time = datetime.datetime.now()
                str = str.replace('前', '').replace('分钟', '分').replace('小时', '时')
                if str.rfind('秒') > -1:
                    return time + datetime.timedelta(seconds=-num)
                if str.rfind('分') > -1:
                    return time + datetime.timedelta(minutes=-num)
                if str.rfind('时') > -1:
                    return time + datetime.timedelta(hours=-num)
                if str.rfind('天') > -1:
                    return time + datetime.timedelta(days=-num)
                if str.rfind('周') > -1:
                    return time + datetime.timedelta(weeks=-num)
                if str.rfind('月') > -1:
                    return time + datetime.timedelta(days=-(num*30))

        date_time = datetime.datetime.strptime(str, format)
        return date_time
    except:
        return None


# 将datetime转换为指定格式的字符串
def get_str_time(time=datetime.datetime.now(), format='%Y-%m-%d'):
    try:
        return time.__format__(format)
    except:
        return None


# 将时间戳转换为datetime
def unixtime_2_datetime(timestamp, millisecond=False):
    if millisecond:
        timestamp /= 1000
    time_local = time.localtime(timestamp)
    return time.strftime("%Y-%m-%d %H:%M:%S", time_local)


# 将datetime转换为时间戳
def datetime_2_unixtime(_datetime, millisecond=False):
    if not millisecond:
        return int(time.mktime(_datetime.timetuple()))
    else:
        return int(time.mktime(_datetime.timetuple()) * 1000.0 + _datetime.microsecond / 1000.0)


def get_html(url, headers=None, encode=None, maxError=3, timeout=10, proxies=None):
    error = 0
    while error < maxError:
        try:
            if not headers:
                headers = default_headers
                headers.__setitem__('Referer', url)

            _request = request.Request(url)
            for key in headers:
                _request.add_header(key, headers[key])

            response = request.urlopen(_request, timeout=timeout)
            html = response.read()
            if encode:
                return html.decode(encode)
            else:
                try:
                    return html.decode()
                except:
                    return html
        except Exception as e:
            error += 1


# 获取网页源代码
def get_html_for_requests(url, maxError=5, timeout=10, headers=None, encode=None, proxies=None):
    error = 0
    while error <= maxError:
        try:
            if not headers:
                headers = default_headers
                headers.__setitem__('Referer', url)

            response = requests.request("GET", url, headers=headers, timeout=timeout, proxies=proxies)
            html = response.text
            if not html.strip():
                time.sleep(1)
                error += 1
                continue
            if encode:
                return html.decode(encode)
            else:
                try:
                    return html.decode()
                except:
                    return html
        except:
            error += 1


def post_for_request(url='', params='', _data='', headers=None, encode=None, timeout=10, proxies=None):
    if not headers:
        headers = default_headers
        headers.__setitem__('Referer', url)

    response = requests.request('POST', url, data=_data, headers=headers, params=params, timeout=timeout, proxies=proxies)
    html = response.text
    if encode:
        return html.decode(encode)
    else:
        try:
            return html.decode()
        except:
            return html


def post_for_request_return_html_and_cookie(url='', params='', _data='', headers=None, encode=None, timeout=10, proxies=None):
    if not headers:
        headers = default_headers
        headers.__setitem__('Referer', url)

    response = requests.request('POST', url, data=_data, headers=headers, params=params, timeout=timeout, proxies=proxies)
    html = response.text
    if encode:
        return html.decode(encode), ';'.join([(f[0] + '=' + f[1]) for f in response.cookies.items()])
    else:
        return html, ';'.join([(f[0] + '=' + f[1]) for f in response.cookies.items()])


def md5_from_str(str):
    if PY3:
        return hashlib.md5(str.encode("utf-8")).hexdigest()
    else:
        return hashlib.md5(str).hexdigest()


def md5_from_file(file_path):
    if not os.path.isfile(file_path):
        return
    myhash = hashlib.md5()
    f = open(file_path,'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()


def sha1_from_str(str):
    if PY3:
        return hashlib.sha1(str.encode("utf-8")).hexdigest()
    else:
        return hashlib.sha1(str).hexdigest()


def sha1_from_file(file_path):
    if not os.path.isfile(file_path):
        return
    myhash = hashlib.sha1()
    f = open(file_path,'rb')
    while True:
        b = f.read(8096)
        if not b:
            break
        myhash.update(b)
    f.close()
    return myhash.hexdigest()


def versioncodeCompare(version1, way, version2):
    list_version1 = [int(f) for f in version1.split('.') if f]
    list_version2 = [int(f) for f in version2.split('.') if f]
    if len(list_version1) != len(list_version2):
        if len(list_version1) < len(list_version2):
            [list_version1.append(0) for f in range(0, len(list_version2) - len(list_version1))]
        else:
            [list_version2.append(0) for f in range(0, len(list_version1) - len(list_version2))]
    if way in ['==', '>=', '<=']:
        flag = True
        for i in range(0, len(list_version1)):
            if list_version1[i] != list_version2[i]:
                if way == '==':
                    return False
                else:
                    flag = False
                    break
        if way == '==' or flag:
            return True
    if way in ['>', '>=']:
        for i in range(0, len(list_version1)):
            if list_version1[i] == list_version2[i]:
                continue
            if list_version1[i] > list_version2[i]:
                return True
            else:
                return False
        return False
    elif way in ['<', '<=']:
        for i in range(0, len(list_version1)):
            if list_version1[i] == list_version2[i]:
                continue
            if list_version1[i] < list_version2[i]:
                return True
            else:
                return False
        return False
    raise Exception('未知的判断类型')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test()
    # print(versioncodeCompare('1.0.2', '>>=', '1.0.1'))
    # print(format_json('{"asd": 123}'))
    # print(get_html('http://www.baidu.com'))
    # print(postForRequest('http://www.baidu.com'))
    pass