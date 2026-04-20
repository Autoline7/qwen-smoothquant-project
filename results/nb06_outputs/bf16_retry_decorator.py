# task: retry_decorator  (model: bf16)
# passed: False  note: expected ValueError, got UnboundLocalError: cannot access local variable 'e' where it is not associated with a value
# tokens: 91

from functools import wraps

def retry(max_attempts=3, exceptions=(Exception,)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempts += 1
            raise e
        return wrapper
    return decorator

# --- raw response ---
# ```python
# from functools import wraps
# 
# def retry(max_attempts=3, exceptions=(Exception,)):
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             attempts = 0
#             while attempts < max_attempts:
#                 try:
#                     return func(*args, **kwargs)
#                 except exceptions as e:
#                     attempts += 1
#             raise e
#         return wrapper
#     return decorator
# ```
