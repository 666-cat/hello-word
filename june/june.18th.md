
### 数学与三角函数
| 函数 | 用法 | 示例 |
| --- | --- | --- |
| `SUM(number1,[number2,...])` | 计算一组数值的总和 | `=SUM(A1:A10)` 计算 A1 到 A10 单元格的数值总和 |
| `AVERAGE(number1,[number2,...])` | 计算一组数值的平均值 | `=AVERAGE(B1:B5)` 求 B1 到 B5 单元格数值的平均值 |
| `MAX(number1,[number2,...])` | 返回一组数值中的最大值 | `=MAX(C1:C8)` 找出 C1 到 C8 单元格中的最大值 |
| `MIN(number1,[number2,...])` | 返回一组数值中的最小值 | `=MIN(D1:D6)` 得到 D1 到 D6 单元格中的最小值 |
| `ROUND(number,num_digits)` | 对数值进行四舍五入 | `=ROUND(3.14159, 2)` 将 3.14159 四舍五入到小数点后两位，结果为 3.14 |

### 文本函数
| 函数 | 用法 | 示例 |
| --- | --- | --- |
| `CONCATENATE(text1,[text2,...])` | 将多个文本字符串合并为一个 | `=CONCATENATE("Hello", " ", "World")` 结果为 "Hello World" |
| `LEFT(text,num_chars)` | 从文本字符串的左侧提取指定数量的字符 | `=LEFT("Excel", 3)` 返回 "Exc" |
| `RIGHT(text,num_chars)` | 从文本字符串的右侧提取指定数量的字符 | `=RIGHT("Function", 4)` 得到 "tion" |
| `MID(text,start_num,num_chars)` | 从文本字符串的指定位置开始提取指定数量的字符 | `=MID("Example", 3, 4)` 返回 "ampl" |
| `LEN(text)` | 返回文本字符串的字符个数 | `=LEN("Hello")` 结果为 5 |

### 逻辑函数
| 函数 | 用法 | 示例 |
| --- | --- | --- |
| `IF(logical_test,value_if_true,value_if_false)` | 根据逻辑测试的结果返回不同的值 | `=IF(A1>10, "大于10", "小于等于10")` |
| `AND(logical1,[logical2,...])` | 所有逻辑条件都为 TRUE 时返回 TRUE，否则返回 FALSE | `=AND(A1>5, A1<15)` |
| `OR(logical1,[logical2,...])` | 只要有一个逻辑条件为 TRUE 就返回 TRUE，全部为 FALSE 时才返回 FALSE | `=OR(B1="苹果", B1="香蕉")` |
| `NOT(logical)` | 对逻辑值取反 | `=NOT(C1>10)` |

### 查找与引用函数
| 函数 | 用法 | 示例 |
| --- | --- | --- |
| `VLOOKUP(lookup_value,table_array,col_index_num,[range_lookup])` | 在表格的首列查找指定的值，并返回该值所在行中指定列处的数值 | `=VLOOKUP("产品A", A1:D10, 3, FALSE)` |
| `HLOOKUP(lookup_value,table_array,row_index_num,[range_lookup])` | 与 VLOOKUP 类似，但在表格的首行查找指定的值，并返回该值所在列中指定行处的数值 |  |
| `INDEX(array,row_num,[column_num])` | 返回表格或数组中的指定值 | `=INDEX(A1:D10, 3, 2)` 返回 A1 到 D10 区域中第 3 行第 2 列的值 |
| `MATCH(lookup_value,lookup_array,[match_type])` | 在数组中查找指定的值，并返回其相对位置 | `=MATCH("目标值", A1:A10, 0)` |

### 日期与时间函数
| 函数 | 用法 | 示例 |
| --- | --- | --- |
| `TODAY()` | 返回当前日期 | 在单元格中输入 `=TODAY()` 显示当前日期 |
| `NOW()` | 返回当前日期和时间 | `=NOW()` 显示当前的日期和时间 |
| `YEAR(serial_number)` | 从日期中提取年份 | `=YEAR(A1)` 返回 A1 单元格中日期的年份 |
| `MONTH(serial_number)` | 从日期中提取月份 | `=MONTH(B1)` 返回 B1 单元格中日期的月份 |
| `DAY(serial_number)` | 从日期中提取日 | `=DAY(C1)` 返回 C1 单元格中日期的日 |