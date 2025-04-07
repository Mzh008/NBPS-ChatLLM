import redis
import hashlib
import logging
import json
from functools import wraps

# 初始化Redis连接
# Initialize Redis connection
redis_client = redis.Redis(host='localhost', port=6379, db=0)
CACHE_EXPIRATION = 60 * 60 * 24  # 24小时过期时间 (秒)

def get_cached_response(query):
    """
    检查Redis缓存中是否存在查询的响应
    Check if a response for this query exists in Redis cache
    
    :param query: 用户查询
    :return: 缓存的响应或None
    """
    try:
        # 生成查询的哈希键
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        # 获取缓存的响应
        cached_response = redis_client.get(query_hash)
        
        if cached_response:
            logging.info(f"缓存命中: {query[:30]}...")
            return cached_response.decode('utf-8')
        return None
    except Exception as e:
        logging.error(f"Redis缓存读取失败: {e}")
        return None

def cache_response(query, response):
    """
    将查询-响应对存储在Redis缓存中
    Store a query-response pair in Redis cache
    
    :param query: 用户查询
    :param response: 生成的响应
    """
    try:
        # 生成查询的哈希键
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        
        # 存储在Redis，带过期时间
        redis_client.setex(query_hash, CACHE_EXPIRATION, response)
        logging.info(f"已缓存响应: {query[:30]}...")
    except Exception as e:
        logging.error(f"Redis缓存写入失败: {e}")
