| 窗口函数 | 用法 | 示例 |
| ----------- | -------------- | ------------ |
| ROW_NUMBER() | 为结果集中的每一行分配一个唯一的行号，从 1 开始按指定的排序顺序依次递增。 | `ROW_NUMBER() OVER (PARTITION BY column1 ORDER BY column2)` <br> 按 `column1` 分组，在每个组内按 `column2` 排序并分配行号。 |
| RANK() | 为结果集中的每一行分配一个排名，相同值的行排名相同，排名会跳过后续的排名值。 | `RANK() OVER (PARTITION BY column1 ORDER BY column2)` <br> 按 `column1` 分组，在每个组内按 `column2` 排序并分配排名。 |
| DENSE_RANK() | 与 `RANK()` 类似，但相同值的行排名相同，排名不会跳过后续的排名值。 | `DENSE_RANK() OVER (PARTITION BY column1 ORDER BY column2)` <br> 按 `column1` 分组，在每个组内按 `column2` 排序并分配密集排名。 |
| NTILE(n) | 将结果集划分为 `n` 个大致相等的桶，并为每一行分配一个桶号。 | `NTILE(4) OVER (ORDER BY column1)` <br> 按 `column1` 排序，将结果集划分为 4 个桶并分配桶号。 |
| LAG(column, offset, default) | 访问当前行之前的第 `offset` 行的 `column` 值，如果不存在则返回 `default` 值。 | `LAG(salary, 1, 0) OVER (ORDER BY employee_id)` <br> 获取当前员工的前一个员工的薪水，如果是第一个员工则返回 0。 |
| LEAD(column, offset, default) | 访问当前行之后的第 `offset` 行的 `column` 值，如果不存在则返回 `default` 值。 | `LEAD(salary, 1, 0) OVER (ORDER BY employee_id)` <br> 获取当前员工的下一个员工的薪水，如果是最后一个员工则返回 0。 |
| FIRST_VALUE(column) | 返回分区内排序后的第一行的 `column` 值。 | `FIRST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC)` <br> 返回每个部门中薪水最高的员工的薪水。 |
| LAST_VALUE(column) | 返回分区内排序后的最后一行的 `column` 值。 | `LAST_VALUE(salary) OVER (PARTITION BY department ORDER BY salary DESC RANGE BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING)` <br> 返回每个部门中薪水最低的员工的薪水。 |
| SUM(column) | 在指定的窗口内计算 `column` 的累计总和。 | `SUM(sales) OVER (PARTITION BY product ORDER BY order_date)` <br> 按 `product` 分组，在每个组内按 `order_date` 计算 `sales` 的累计总和。 |
| AVG(column) | 在指定的窗口内计算 `column` 的平均值。 | `AVG(salary) OVER (PARTITION BY department)` <br> 计算每个部门的平均薪水。 |
| MIN(column) | 在指定的窗口内计算 `column` 的最小值。 | `MIN(price) OVER (PARTITION BY category)` <br> 计算每个类别的最低价格。 |
| MAX(column) | 在指定的窗口内计算 `column` 的最大值。 | `MAX(quantity) OVER (PARTITION BY order_id)` <br> 计算每个订单中的最大数量。 |
