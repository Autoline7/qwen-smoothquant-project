# task: flatten_dict  (model: bf16)
# passed: True  note: 5/5 cases passed
# tokens: 122

def flatten_dict(d, sep='.'):
    def _flatten(sub_dict, parent_key, sep):
        items = []
        for k, v in sub_dict.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(_flatten(v, new_key, sep).items())
            elif isinstance(v, list):
                items.append((new_key, v))
            else:
                items.append((new_key, v))
        return dict(items)

    return _flatten(d, '', sep)

# --- raw response ---
# ```
# python
# def flatten_dict(d, sep='.'):
#     def _flatten(sub_dict, parent_key, sep):
#         items = []
#         for k, v in sub_dict.items():
#             new_key = f"{parent_key}{sep}{k}" if parent_key else k
#             if isinstance(v, dict):
#                 items.extend(_flatten(v, new_key, sep).items())
#             elif isinstance(v, list):
#                 items.append((new_key, v))
#             else:
#                 items.append((new_key, v))
#         return dict(items)
# 
#     return _flatten(d, '', sep)
# ```
