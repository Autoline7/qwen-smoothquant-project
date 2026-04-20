# task: csv_group_sum  (model: w8a8)
# passed: True  note: 3/3 cases passed
# tokens: 84

def csv_group_sum(rows, key_col, value_col):
    result = {}
    for row in rows:
        key = row[key_col]
        value = row[value_col]
        try:
            value = float(value)
            if key in result:
                result[key] += value
            else:
                result[key] = value
        except ValueError:
            continue
    return result

# --- raw response ---
# ```python
# def csv_group_sum(rows, key_col, value_col):
#     result = {}
#     for row in rows:
#         key = row[key_col]
#         value = row[value_col]
#         try:
#             value = float(value)
#             if key in result:
#                 result[key] += value
#             else:
#                 result[key] = value
#         except ValueError:
#             continue
#     return result
# ```
