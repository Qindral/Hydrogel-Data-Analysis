# elementwise on the whole DataFrame — apply column-wise using Series.map
# This avoids the deprecated DataFrame.applymap and keeps the same semantics.
result = df.apply(lambda col: col.map(my_func))